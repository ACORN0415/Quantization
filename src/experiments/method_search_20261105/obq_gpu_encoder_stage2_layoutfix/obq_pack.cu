// Exact ordered OBQ on already rotated BF16 K. One warp owns one token/head row.
// No scale search, sorting, approximate division, fast math or factor construction.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void obq_pack_kernel(const __nv_bfloat16* z, const __half* scale,
                                const float* factor, unsigned char* packed,
                                const long rows) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long row = (long)blockIdx.x * 4 + warp;
  if (row >= rows) return;  // the whole warp takes the same branch
  const int head = row % 12;
  const long scale_row = (row / 12 / 16) * 12 + head;
  const float* u = factor + (long)head * 128 * 128;
  float w[4], s[4];
  int codes[4];
  #pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int k = slot * 32 + lane;
    w[slot] = __bfloat162float(z[row * 128 + k]);
    s[slot] = __half2float(scale[scale_row * 128 + k]);
    codes[slot] = 0;
  }
  // Deliberately not unrolled into 128 independent alternatives: preserve j order.
  #pragma unroll 1
  for (int j = 0; j < 128; ++j) {
    float local_error = 0.0f;
    if (lane == (j & 31)) {
      // Constant array subscripts keep the four values eligible for registers.
      const float wj = j < 32 ? w[0] : (j < 64 ? w[1] : (j < 96 ? w[2] : w[3]));
      const float sj = j < 32 ? s[0] : (j < 64 ? s[1] : (j < 96 ? s[2] : s[3]));
      const float rounded = nearbyintf(__fdiv_rn(wj, sj));
      const float code = fminf(7.0f, fmaxf(-8.0f, rounded));
      if (j < 32) codes[0] = (int)code;
      else if (j < 64) codes[1] = (int)code;
      else if (j < 96) codes[2] = (int)code;
      else codes[3] = (int)code;
      if (j < 127) {
        const float residual = __fsub_rn(wj, __fmul_rn(code, sj));
        local_error = __fdiv_rn(residual, u[j * 128 + j]);
      }
    }
    const float error = __shfl_sync(0xffffffff, local_error, j & 31);
    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      const int k = slot * 32 + lane;
      if (k > j) w[slot] = __fsub_rn(w[slot], __fmul_rn(error, u[j * 128 + k]));
    }
  }
  #pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int next = __shfl_down_sync(0xffffffff, codes[slot], 1);
    if ((lane & 1) == 0) {
      const int k = slot * 32 + lane;
      packed[row * 64 + k / 2] = (unsigned char)((codes[slot] & 15) | ((next & 15) << 4));
    }
  }
}

torch::Tensor obq_pack(torch::Tensor z, torch::Tensor scale, torch::Tensor factor) {
  TORCH_CHECK(z.is_cuda() && scale.is_cuda() && factor.is_cuda(), "all inputs must be CUDA tensors");
  TORCH_CHECK(z.device() == scale.device() && z.device() == factor.device(), "device mismatch");
  TORCH_CHECK(z.scalar_type() == at::kBFloat16 && scale.scalar_type() == at::kHalf && factor.scalar_type() == at::kFloat,
              "expected BF16 z, FP16 scale and FP32 factor");
  TORCH_CHECK(z.is_contiguous() && scale.is_contiguous() && factor.is_contiguous(), "contiguous inputs required");
  TORCH_CHECK(z.dim() == 5 && z.size(2) == 16 && z.size(3) == 12 && z.size(4) == 128, "z shape must be [frames,groups,16,12,128]");
  TORCH_CHECK(scale.dim() == 5 && scale.size(0) == z.size(0) && scale.size(1) == z.size(1) && scale.size(2) == 1 && scale.size(3) == 12 && scale.size(4) == 128, "scale shape mismatch");
  TORCH_CHECK(factor.dim() == 3 && factor.size(0) == 12 && factor.size(1) == 128 && factor.size(2) == 128, "factor shape mismatch");
  c10::cuda::CUDAGuard guard(z.device());
  const long rows = z.numel() / 128;
  auto out = torch::empty({z.numel() / 2}, z.options().dtype(torch::kUInt8));
  const auto stream = at::cuda::getCurrentCUDAStream(z.get_device());
  obq_pack_kernel<<<(rows + 3) / 4, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(z.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __half*>(scale.data_ptr<at::Half>()),
      factor.data_ptr<float>(), out.data_ptr<unsigned char>(), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("obq_pack", &obq_pack); }
