# -*- coding: utf-8 -*-
"""S9 프로파일 1부 분석 — prof_<tag>.json (torch.profiler key_averages) 을 분류별로 집계하고 fused − BF16 증분을 분해, 상한 공식 적용.
사용: analyse_profile.py DIR [--tags bf16 int2_l0 int2_l1 k4v2_l1]"""
import argparse, json, os, re
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--tags", nargs="+", default=["bf16", "int2_l0", "int2_l1", "k4v2_l1"])
a = ap.parse_args(); O = a.dir

def is_op(n):   # key_averages 는 aten/flash_attn 연산(op)과 그 아래 커널을 모두 포함해 device 시간을 둘 다 적는다 → 커널 행만 센다
    return n.startswith(("aten::", "flash_attn::", "torch::", "cudaLaunch", "cudaDevice", "cudaStream", "cudaMem", "cudaEvent", "ProfilerStep", "Optimizer", "autograd::", "nn.Module", "detach", "triton::", "c10::"))
def family(n):
    s = n.lower()
    if "_fused_kernel" in s or "fused_lowbit" in s or "fused_attn" in s: return "fused_attn"
    if "flash" in s or "fmha" in s: return "flash_attn"
    if "quant_pack" in s or "_quant_" in s or "quantize" in s or "pack" in s: return "quant_pack"
    if any(t in s for t in ("gemm", "cutlass", "addmm", "matmul", "mm_", "sgemm", "hgemm", "cublas")): return "gemm"
    if any(t in s for t in ("conv", "upsample", "interpolate", "cudnn")): return "vae_conv"
    if any(t in s for t in ("softmax",)): return "softmax"
    if any(t in s for t in ("norm", "layer_norm", "rms")): return "norm"
    if any(t in s for t in ("elementwise", "vectorized", "copy", "cat", "fill", "index", "reduce", "mul", "add", "silu", "gelu", "rope", "polar", "complex")): return "elementwise"
    if "memcpy" in s or "memset" in s: return "memcpy"
    return "other"

tot = {}; fam = {}; raw = {}
for t in a.tags:
    p = f"{O}/prof_{t}.json"
    if not os.path.exists(p): print("없음:", p); continue
    rows = json.load(open(p))[-1]                      # 마지막 프롬프트
    raw[t] = rows; f = {}
    rows = [r for r in rows if not is_op(r["name"])]
    for r in rows: f[family(r["name"])] = f.get(family(r["name"]), 0.0) + r["self_device_ms"]
    fam[t] = f; tot[t] = sum(f.values())
    wall = None
    sp = f"{O}/{t}.stats.json"
    if os.path.exists(sp): wall = json.load(open(sp))["runs"][-1]["seconds"] * 1e3
    print(f"\n## {t}: device 합 {tot[t]:.0f} ms" + (f", wall {wall:.0f} ms (CPU·launch·동기화 등 {wall - tot[t]:.0f} ms)" if wall else ""))
    for k, v in sorted(f.items(), key=lambda x: -x[1]): print(f"   {k:12s} {v:9.1f} ms {100*v/tot[t]:5.1f}%")
    top = sorted(rows, key=lambda r: -r["self_device_ms"])[:6]
    print("   상위 커널: " + " | ".join(f"{r['name'][:38]} {r['self_device_ms']:.0f}ms×{r['count']}" for r in top))
if "bf16" in fam:
    print("\n## fused − BF16 증분 (분류별 ms)")
    keys = sorted({k for f in fam.values() for k in f})
    for t in a.tags:
        if t == "bf16" or t not in fam: continue
        d = {k: fam[t].get(k, 0) - fam["bf16"].get(k, 0) for k in keys}; tot_d = tot[t] - tot["bf16"]
        attn_d = fam[t].get("fused_attn", 0) - fam["bf16"].get("flash_attn", 0)
        print(f"  {t}: 총 증분 {tot_d:+.0f} ms | attention 커널 증분(fused − flash) {attn_d:+.0f} ms ({100*attn_d/tot_d if tot_d else 0:.0f}%) | 패킹 {d.get('quant_pack',0):+.0f} | 그 외 {tot_d - attn_d - d.get('quant_pack',0):+.0f}")
        sp = f"{O}/{t}.stats.json"
        if os.path.exists(sp):
            wall = json.load(open(sp))["runs"][-1]["seconds"] * 1e3
            print(f"      상한: 완벽한 attention 커널(FA2 수준)이 얻을 수 있는 종단 이득 ≤ {attn_d:.0f} ms / wall {wall:.0f} ms = {100*attn_d/wall:.1f}% ; fused_attn 이 device 합의 {100*fam[t].get('fused_attn',0)/tot[t]:.1f}%")
json.dump({"family_ms": fam, "total_ms": tot}, open(f"{O}/profile1_summary.json", "w"), indent=1, ensure_ascii=False)
