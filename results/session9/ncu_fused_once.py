"""S9 프로파일 3부 — Nsight Compute 용 단일 실행: 창 21f 대표 shape 에서 FA2 1회, fused INT2 λ0 1회, λ1 1회 (warm-up 후).
사용(관리자 권한 필요):
  sudo /usr/local/cuda/bin/ncu --set full --kernel-name regex:"_fused_kernel_v2|flash_fwd_kernel" --launch-skip 3 --launch-count 3 \
       -o /vfsdata/gpu_data/gpu/Self-Forcing/results/session9/profile/ncu_fused_w21_v2 \
       /vfsdata/gpu_data/gpu/sf-venv/bin/python /vfsdata/gpu_data/gpu/Self-Forcing/kernels/bench/ncu_fused_once.py
  결과 보기: ncu --import ncu_fused_w21.ncu-rep --page details   (메모리/연산 처리량, 레지스터/spill, occupancy, warp stall)"""
import os, sys, torch
sys.path.insert(0, "/vfsdata/gpu_data/gpu/Self-Forcing/kernels")
from triton_kv_quant import triton_quantize_kv
from triton_fused_lowbit_attn import fused_lowbit_attention_v2
import flash_attn
B, H, D, CHUNK, W = 1, 12, 128, 4680, 32760
torch.manual_seed(0)
q = torch.randn(B, CHUNK, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, W, H, D, device="cuda", dtype=torch.bfloat16); v = torch.randn_like(k)
seg = W - CHUNK; nb = (CHUNK + 15) // 16
def pack_segments(x, bits):   # chunk 별 양자화 후 연결 — 실제 fused 경로와 동일 (v1 은 통째 양자화로 scale 블록 수 불일치)
    pks, scs = zip(*[triton_quantize_kv(x[:, j * CHUNK:(j + 1) * CHUNK].contiguous(), bits) for j in range(x.shape[1] // CHUNK)])
    return torch.cat(pks, 2), torch.cat(scs, 2)
pk, ks = pack_segments(k[:, :seg].contiguous(), 2); pv, vs = pack_segments(v[:, :seg].contiguous(), 2)
sk, sv = k[:, seg:].contiguous(), v[:, seg:].contiguous()
def fa2(): return flash_attn.flash_attn_func(q, k, v, softmax_scale=1 / D ** 0.5)
def fused(lam): return fused_lowbit_attention_v2(q, pk, ks, pv, vs, sk, sv, key_bits=2, value_bits=2, bias_lambda=lam, seg_len=CHUNK, seg_nb=nb, block_m=128, block_n=32, num_warps=4, num_stages=2)
for _ in range(3): fa2(); fused(0.0); fused(1.0)            # warm-up (launch-skip 3 으로 건너뜀; 커널 이름 필터 기준 각 3회)
torch.cuda.synchronize()
fa2(); fused(0.0); fused(1.0); torch.cuda.synchronize(); print("done")
