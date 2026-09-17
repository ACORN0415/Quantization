"""Gate D — is the KV cache actually the bottleneck at 1.3B?

Answers the reviewer question the motivation section needs: on a 4090, where does
a chunk's time and memory go, which operations are compute- vs memory-bound, and
under what cache length / model size / stream count does the KV cache overtake
the weights.

Session 1 measured RTN INT4 KV at 19% lower peak memory but 24% higher latency.
Part 4 here localizes that 24%.

Sections:
  1  latency breakdown per chunk   (torch.profiler, by operator family)
  2  memory breakdown              (weights / KV / activations / VAE / text)
  3  roofline                      (GEMM FLOPs vs attention KV bytes)
  4  cost of INT4 KV               (where the extra 24% comes from)
  5  extrapolation                 (cache length x model size x streams)
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# RTX 4090, from the datasheet: BF16 tensor-core dense throughput and the
# memory bandwidth the roofline is drawn against.
GPU_BF16_TFLOPS = 165.2
GPU_BW_GBPS = 1008.0


def build(config_path, ckpt, device, kv_quant=None, kv_bits=4):
    from pipeline import CausalInferencePipeline
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load(config_path))
    pipe = CausalInferencePipeline(config, device=device)
    state = torch.load(ckpt, map_location="cpu")
    pipe.generator.load_state_dict(state["generator_ema"])
    del state
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.generator.to(device)
    if kv_quant:
        sys.path.insert(0, os.path.expanduser("~/gpu/kv-quant-longhorizon"))
        from kv_quant.factory import create_quantizer
        pipe.kv_quantizer = create_quantizer(
            kv_quant, bits=kv_bits, block_size=16,
            key_bits=kv_bits, value_bits=kv_bits,
            name=f"{kv_quant.upper()}_INT{kv_bits}")
    return pipe, config


def family(name):
    n = name.lower()
    if any(t in n for t in ("flash", "attention", "softmax", "bmm")):
        return "attention"
    if any(t in n for t in ("gemm", "addmm", "linear", "matmul", "mm")):
        return "gemm"
    if any(t in n for t in ("conv", "upsample", "interpolate")):
        return "vae"
    if any(t in n for t in ("quant", "round", "clamp")):
        return "quantize"
    return "other"


def latency_breakdown(pipe, cond, noise, nfb, frame_seq_length, out_dir, tag):
    """Per-chunk wall time, split by operator family."""
    from torch.profiler import profile, ProfilerActivity

    # warm up so autotuning and allocator growth do not land in the measurement
    _run_chunks(pipe, cond, noise, nfb, frame_seq_length, n_chunks=2)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        _run_chunks(pipe, cond, noise, nfb, frame_seq_length, n_chunks=3)
        torch.cuda.synchronize()

    agg = defaultdict(float)
    total = 0.0
    for e in prof.key_averages():
        t = e.self_device_time_total / 1e3      # ms
        if t <= 0:
            continue
        agg[family(e.key)] += t
        total += t
    rows = [{"tag": tag, "family": k, "ms": v, "pct": 100.0 * v / total}
            for k, v in sorted(agg.items(), key=lambda x: -x[1])]
    with open(os.path.join(out_dir, f"latency_{tag}.json"), "w") as f:
        json.dump({"total_ms": total, "per_family": rows}, f, indent=2)
    print(f"\n[latency] {tag}: {total:.0f} ms of device time over 3 chunks")
    for r in rows:
        print(f"  {r['family']:10s} {r['ms']:9.1f} ms  {r['pct']:5.1f}%")
    return rows


def _run_chunks(pipe, cond, noise, nfb, fsl, n_chunks):
    """Drive the chunk loop directly so profiling covers only generation."""
    nblk = pipe.num_transformer_blocks
    las = pipe.local_attn_size
    cache_size = las * fsl if las != -1 else 32760
    kv = [{
        "k": torch.zeros([1, cache_size, 12, 128], dtype=torch.bfloat16, device=noise.device),
        "v": torch.zeros([1, cache_size, 12, 128], dtype=torch.bfloat16, device=noise.device),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=noise.device),
        "local_end_index": torch.tensor([0], dtype=torch.long, device=noise.device),
        "quantizer": getattr(pipe, "kv_quantizer", None), "quant_state": None,
    } for _ in range(nblk)]
    xattn = [{"k": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=noise.device),
              "v": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=noise.device),
              "is_init": False} for _ in range(nblk)]
    start = 0
    for c in range(n_chunks):
        x = noise[:, start:start + nfb]
        for i, t in enumerate(pipe.denoising_step_list):
            ts = torch.ones([1, nfb], device=noise.device, dtype=torch.int64) * t
            _, pred = pipe.generator(noisy_image_or_video=x, conditional_dict=cond,
                                     timestep=ts, kv_cache=kv, crossattn_cache=xattn,
                                     current_start=start * fsl)
            if i < len(pipe.denoising_step_list) - 1:
                flat = pred.flatten(0, 1)
                x = pipe.scheduler.add_noise(
                    flat, torch.randn_like(flat),
                    pipe.denoising_step_list[i + 1] * torch.ones([nfb], device=noise.device, dtype=torch.long)
                ).unflatten(0, pred.shape[:2])
        ctx = torch.ones([1, nfb], device=noise.device, dtype=torch.int64) * pipe.args.context_noise
        pipe.generator(noisy_image_or_video=pred, conditional_dict=cond, timestep=ctx,
                       kv_cache=kv, crossattn_cache=xattn, current_start=start * fsl)
        start += nfb
    del kv, xattn


def memory_breakdown(pipe, fsl, out_dir):
    """Static split of what occupies memory, and KV share at several cache lengths."""
    def nbytes(mod):
        return sum(p.numel() * p.element_size() for p in mod.parameters()) + \
               sum(b.numel() * b.element_size() for b in mod.buffers())

    gen = nbytes(pipe.generator)
    vae = nbytes(pipe.vae)
    txt = nbytes(pipe.text_encoder)
    nblk = pipe.num_transformer_blocks
    rows = []
    for frames in (12, 21, 32, 63, 126):
        tok = frames * fsl
        kv = nblk * 2 * 1 * tok * 12 * 128 * 2      # bf16 K and V, all layers
        rows.append({"cache_frames": frames, "cache_tokens": tok,
                     "kv_bytes": kv, "kv_gb": kv / 1024 ** 3,
                     "kv_vs_generator": kv / gen})
    out = {"generator_gb": gen / 1024 ** 3, "vae_gb": vae / 1024 ** 3,
           "text_encoder_gb": txt / 1024 ** 3, "kv_by_cache_length": rows}
    with open(os.path.join(out_dir, "memory.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[memory] generator {gen/1024**3:.2f} GB | VAE {vae/1024**3:.2f} GB | "
          f"text encoder {txt/1024**3:.2f} GB")
    for r in rows:
        print(f"  cache {r['cache_frames']:3d} frames -> KV {r['kv_gb']:5.2f} GB "
              f"({r['kv_vs_generator']:.2f}x the generator weights)")
    return out


def roofline(pipe, fsl, nfb, out_dir):
    """Per chunk: GEMM FLOPs against attention's KV traffic.

    Arithmetic intensity below the machine balance (FLOPs per byte) means the
    operation is memory-bound, which is the case attention slides into as the
    cache grows.
    """
    m = pipe.generator.model
    nblk = pipe.num_transformer_blocks
    dim = 1536
    ffn = 8960
    q = nfb * fsl                                   # queries per chunk
    balance = GPU_BF16_TFLOPS * 1e12 / (GPU_BW_GBPS * 1e9)

    # 4 denoising steps + 1 context refresh
    calls = len(pipe.denoising_step_list) + 1
    gemm_flops = calls * nblk * 2 * q * (4 * dim * dim + 2 * dim * ffn)

    rows = []
    for frames in (12, 21, 32, 63):
        kvt = frames * fsl
        attn_flops = calls * nblk * 2 * 2 * q * kvt * dim      # QK^T and PV
        kv_bytes = calls * nblk * 2 * kvt * 12 * 128 * 2       # K and V read, bf16
        ai = attn_flops / kv_bytes
        rows.append({
            "cache_frames": frames,
            "attn_tflops": attn_flops / 1e12,
            "kv_read_gb": kv_bytes / 1024 ** 3,
            "arithmetic_intensity": ai,
            "bound": "compute" if ai > balance else "memory",
        })
    out = {"gemm_tflops_per_chunk": gemm_flops / 1e12,
           "machine_balance_flops_per_byte": balance,
           "attention": rows}
    with open(os.path.join(out_dir, "roofline.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[roofline] GEMM {gemm_flops/1e12:.2f} TFLOP per chunk; "
          f"machine balance {balance:.1f} FLOP/byte")
    for r in rows:
        print(f"  cache {r['cache_frames']:3d} frames: attention {r['attn_tflops']:.2f} TFLOP, "
              f"KV read {r['kv_read_gb']:.2f} GB, AI {r['arithmetic_intensity']:.1f} "
              f"-> {r['bound']}-bound")
    return out


def extrapolate(fsl, out_dir):
    """Where does the KV cache overtake the weights, and blow past a card?"""
    models = {  # name: (params, layers, heads, head_dim)
        "Wan2.1-1.3B": (1.3e9, 30, 12, 128),
        "LongLive-2.0-5B": (5.0e9, 30, 16, 128),
        "Wan2.1-14B": (14e9, 40, 40, 128),
    }
    rows = []
    for name, (params, layers, heads, hd) in models.items():
        w_gb = params * 2 / 1024 ** 3
        for frames in (12, 21, 32, 63, 126):
            tok = frames * fsl
            for streams in (1, 4, 8):
                kv_gb = layers * 2 * streams * tok * heads * hd * 2 / 1024 ** 3
                rows.append({
                    "model": name, "weights_gb": w_gb, "cache_frames": frames,
                    "streams": streams, "kv_gb": kv_gb,
                    "kv_over_weights": kv_gb / w_gb,
                    "total_gb": kv_gb + w_gb,
                    "fits_12gb": kv_gb + w_gb < 12, "fits_24gb": kv_gb + w_gb < 24,
                })
    import csv
    path = os.path.join(out_dir, "extrapolation.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[extrapolation] wrote {path} ({len(rows)} rows)")
    for r in rows:
        if r["kv_over_weights"] >= 1.0:
            print(f"  KV exceeds weights: {r['model']}, {r['cache_frames']} frames, "
                  f"{r['streams']} stream(s) -> KV {r['kv_gb']:.1f} GB vs weights {r['weights_gb']:.1f} GB")
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="configs/gateA_A1.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    ap.add_argument("--data_path", default="prompts/quant3/prompts10.txt")
    ap.add_argument("--out_dir", default="results/profile")
    ap.add_argument("--sections", nargs="+",
                    default=["latency", "memory", "roofline", "extrapolate"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    fsl, nfb = 1560, 3

    if "extrapolate" in args.sections:
        extrapolate(fsl, args.out_dir)

    need_model = {"latency", "memory", "roofline"} & set(args.sections)
    if not need_model:
        return

    from utils.dataset import TextDataset
    pipe, _ = build(args.config_path, args.checkpoint_path, device)
    ds = TextDataset(prompt_path=args.data_path)
    pipe.text_encoder.to(device)
    cond = pipe.text_encoder(text_prompts=[ds[0]["prompts"]])
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    if "memory" in args.sections:
        memory_breakdown(pipe, fsl, args.out_dir)
    if "roofline" in args.sections:
        roofline(pipe, fsl, nfb, args.out_dir)

    if "latency" in args.sections:
        noise = torch.cat([
            torch.randn([1, nfb, 16, 60, 104],
                        generator=torch.Generator(device=device).manual_seed(c),
                        device=device, dtype=torch.bfloat16) for c in range(21)], dim=1)
        latency_breakdown(pipe, cond, noise, nfb, fsl, args.out_dir, "bf16")

        # same again with the INT4 KV quantizer, to localize session 1's +24%
        del pipe
        torch.cuda.empty_cache()
        pipe2, _ = build(args.config_path, args.checkpoint_path, device,
                         kv_quant="RTN", kv_bits=4)
        pipe2.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        latency_breakdown(pipe2, cond, noise, nfb, fsl, args.out_dir, "int4_kv")


if __name__ == "__main__":
    main()
