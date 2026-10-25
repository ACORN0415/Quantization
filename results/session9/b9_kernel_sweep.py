"""S9 프로파일 2부 v2 (chunk 별 패킹으로 수정) — S7 B4 하네스 확장: 창 길이 {12,21,36,63}f × {FA2 bf16, fused INT2/K4V2 λ0/λ1, 복원→FA2(복원·FA2 분리), 패킹}.
한 프로세스·한 타이밍(CUDA 이벤트, warm-up 10, 반복 30, 중앙값). 사전 등록 Quantization/문서/지시서/S9_profile_prereg.md §2."""
import json, os, sys
import torch
sys.path.insert(0, os.path.expanduser("~/gpu/Self-Forcing/kernels"))
from triton_kv_quant import triton_quantize_kv, unpack_reference
from triton_fused_lowbit_attn import fused_lowbit_attention_v2
import flash_attn

B, H, D, CHUNK, FPF = 1, 12, 128, 4680, 1560
DEV = "cuda"; WARM, ITER = 10, 30
OUT = os.environ.get("B9_OUT", os.path.expanduser("~/gpu/Self-Forcing/results/session9/profile/b9_kernel_sweep.json"))

def timed(fn):
    for _ in range(WARM): fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(ITER)]
    for a, b in ev:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    ms = sorted(a.elapsed_time(b) for a, b in ev); return ms[len(ms) // 2]

def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

def pack_segments(x, bits):
    """실제 fused 경로(_fused_quantized_attention)와 같이 chunk 별로 양자화한 뒤 packed(dim 2 = L)·scale(dim 2 = nb) 을 연결한다.
    (v1 은 과거 전체를 한 번에 양자화해 scale 블록 수가 1,755 vs 커널 기대 1,758 로 어긋났다 — 2차 리뷰 지적.)"""
    pks, scs = [], []
    for j in range(x.shape[1] // CHUNK):
        pk, sc = triton_quantize_kv(x[:, j * CHUNK:(j + 1) * CHUNK].contiguous(), bits); pks.append(pk); scs.append(sc)
    return torch.cat(pks, 2), torch.cat(scs, 2), list(zip(pks, scs))

def dequant_segments(parts, bits):
    return torch.cat([unpack_reference(pk, sc, bits) for pk, sc in parts], 1)

torch.manual_seed(0); scale = 1.0 / D ** 0.5; res = {}
for W in (12, 21, 36, 63):
    window = W * FPF; seg = window - CHUNK; nb = (CHUNK + 15) // 16
    q = torch.randn(B, CHUNK, H, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, window, H, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, window, H, D, device=DEV, dtype=torch.bfloat16)
    r = {"window_tokens": window}
    ref = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale)
    r["fa2_bf16"] = timed(lambda: flash_attn.flash_attn_func(q, k, v, softmax_scale=scale))
    kc, vc = k[:, :seg].contiguous(), v[:, :seg].contiguous(); sk, sv = k[:, seg:].contiguous(), v[:, seg:].contiguous()
    r["pack_write_chunk_int2"] = timed(lambda: triton_quantize_kv(k[:, :CHUNK].contiguous(), 2))     # 쓰기 비용: chunk 1개 K
    r["pack_write_chunk_int4"] = timed(lambda: triton_quantize_kv(k[:, :CHUNK].contiguous(), 4))
    for kb, vb, tag in ((2, 2, "int2"), (4, 2, "k4v2")):
        pk, ks, kparts = pack_segments(kc, kb); pv, vs, vparts = pack_segments(vc, vb)
        assert ks.shape[2] == (seg // CHUNK) * nb, (ks.shape, nb)
        for lam, lt in ((0.0, "_l0"), (1.0, "_l1")):
            fn = lambda lam=lam, kb=kb, vb=vb, pk=pk, ks=ks, pv=pv, vs=vs: fused_lowbit_attention_v2(
                q, pk, ks, pv, vs, sk, sv, key_bits=kb, value_bits=vb, bias_lambda=lam, seg_len=CHUNK, seg_nb=nb,
                block_m=128, block_n=32, num_warps=4, num_stages=2)
            r[f"fused_{tag}{lt}"] = timed(fn)
            if lam == 0.0: r[f"fused_{tag}_l0_rel_vs_fa2_on_dequant"] = None   # 아래에서 채움
        # 복원 → FA2 (복원은 torch unpack_reference = 복원 비용의 상한; 출력 [B,L,H,D] bf16)
        def deq(kparts=kparts, vparts=vparts, kb=kb, vb=vb):
            return dequant_segments(kparts, kb), dequant_segments(vparts, vb)
        r[f"dequant_{tag}_torch"] = timed(lambda: deq())
        kd, vd = deq()
        kfull = torch.cat([kd, sk], 1); vfull = torch.cat([vd, sv], 1)
        r[f"fa2_after_dequant_{tag}"] = timed(lambda: flash_attn.flash_attn_func(q, kfull, vfull, softmax_scale=scale))
        r[f"dequant_plus_fa2_{tag}"] = r[f"dequant_{tag}_torch"] + r[f"fa2_after_dequant_{tag}"]
        out_f = fused_lowbit_attention_v2(q, pk, ks, pv, vs, sk, sv, key_bits=kb, value_bits=vb, bias_lambda=0.0, seg_len=CHUNK, seg_nb=nb,
                                          block_m=128, block_n=32, num_warps=4, num_stages=2)
        out_d = flash_attn.flash_attn_func(q, kfull, vfull, softmax_scale=scale)
        r[f"fused_{tag}_l0_rel_vs_fa2_on_dequant"] = rel(out_f, out_d)
    res[f"w{W}"] = r
    print(f"W{W}f ({window} tok): " + " | ".join(f"{k_} {v_:.3f}" for k_, v_ in r.items() if isinstance(v_, float) and "rel" not in k_), flush=True)
    print(f"   정확도 fused λ0 vs 복원→FA2: int2 {r['fused_int2_l0_rel_vs_fa2_on_dequant']:.2e} k4v2 {r['fused_k4v2_l0_rel_vs_fa2_on_dequant']:.2e}", flush=True)
    del q, k, v, ref; torch.cuda.empty_cache()
json.dump({"shapes": {"B": B, "H": H, "D": D, "chunk": CHUNK}, "method": f"CUDA events, median of {ITER} after {WARM} warm-up, one process",
           "commit": os.popen("git rev-parse HEAD").read().strip(), "res": res}, open(OUT, "w"), indent=1)
print("→", OUT)
