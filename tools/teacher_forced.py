"""Step F — lockstep teacher-forced vs free-running error separation.

At every chunk the reference (BF16) and the quantized computation run over the
*same* context, so the quantized latent error is the error injected at that
chunk alone, with no room for trajectory divergence to accumulate:

    e_inj[t]       = TF error, measured here
    e_total[t]     = FR error, reused from session 1
    propagation[t] = e_total[t] - e_inj[t]

Design constraints (session 2 handoff, section 1):
  * both computations of a chunk use the same chunk-seeded noise generators; the
    generator is rebuilt from the same seed before the quantized recomputation
  * the reference cache is advanced by the BF16 computation only; the quantized
    pass writes exclusively into a working copy
  * weight/activation fake quantization is toggled on the single BF16 weight set
    rather than holding a second quantized copy
  * with quantization disabled the recomputation must be bit-exact against the
    reference (gate: --bitexact_gate)

Teacher-forced definition: slots 0..t-1 of the working cache are filled with the
reference BF16 K/V, that cache is quantized *once*, and chunk t is generated
from it. The chunk's own K/V is handled exactly as in the free-running path.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wan.modules.causal_model as causal_model
from fakequant import fq_active
from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed


def build_cache(kv_cache_size, dtype, device, num_blocks):
    return [{
        "k": torch.zeros([1, kv_cache_size, 12, 128], dtype=dtype, device=device),
        "v": torch.zeros([1, kv_cache_size, 12, 128], dtype=dtype, device=device),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "quantizer": None,
        "quant_state": None,
    } for _ in range(num_blocks)]


def build_crossattn_cache(dtype, device, num_blocks):
    return [{
        "k": torch.zeros([1, 512, 12, 128], dtype=dtype, device=device),
        "v": torch.zeros([1, 512, 12, 128], dtype=dtype, device=device),
        "is_init": False,
    } for _ in range(num_blocks)]


def anchor_work_cache(work, ref, quantizer, kv_cache_size, device):
    """Re-anchor the working cache to the reference BF16 context (slots 0..t-1).

    With a KV quantizer active the context is quantized once, here, so every
    denoising step of this chunk reads a clean context that has been quantized,
    rather than one that drifted across previous chunks. The UCSD patch keeps a
    live cache in `quant_state` and empties k/v, so re-anchoring means
    re-quantizing the reference contents from scratch each chunk.
    """
    for wc, rc in zip(work, ref):
        wc["global_end_index"].copy_(rc["global_end_index"])
        wc["local_end_index"].copy_(rc["local_end_index"])
        if quantizer is None:
            wc["quantizer"] = None
            wc["quant_state"] = None
            if wc["k"].numel() != rc["k"].numel():
                wc["k"] = torch.empty_like(rc["k"])
                wc["v"] = torch.empty_like(rc["v"])
            wc["k"].copy_(rc["k"])
            wc["v"].copy_(rc["v"])
        else:
            wc["quantizer"] = quantizer
            wc["quant_state"] = quantizer.quantize_kv(
                rc["k"], rc["v"], meta={"tensor_dtype": rc["k"].dtype})
            wc["k"] = rc["k"].new_empty(0)
            wc["v"] = rc["v"].new_empty(0)


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


def run_chunk(pipeline, kv_cache, crossattn_cache, cond, noise,
              chunk_idx, start_frame, nfb, base_seed, capture_new_kv=False):
    """One chunk: the denoising steps, then the clean-context cache refresh."""
    # Rebuilt from the same seed for both passes, so the quantized recomputation
    # sees exactly the reference run's re-noising draws.
    renoise_gen = torch.Generator(device=noise.device)
    renoise_gen.manual_seed(base_seed * 1000003 + 7919 + chunk_idx)

    noisy_input = noise[:, start_frame:start_frame + nfb]
    current_start = start_frame * pipeline.frame_seq_length

    denoised_pred = None
    for index, current_timestep in enumerate(pipeline.denoising_step_list):
        timestep = torch.ones([1, nfb], device=noise.device,
                              dtype=torch.int64) * current_timestep
        _, denoised_pred = pipeline.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=cond,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
        )
        if index < len(pipeline.denoising_step_list) - 1:
            next_timestep = pipeline.denoising_step_list[index + 1]
            flat_pred = denoised_pred.flatten(0, 1)
            renoise = torch.randn(flat_pred.shape, generator=renoise_gen,
                                  device=flat_pred.device, dtype=flat_pred.dtype)
            noisy_input = pipeline.scheduler.add_noise(
                flat_pred, renoise,
                next_timestep * torch.ones([nfb], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])

    # Clean-context refresh: the K/V written here is what the chunk hands on.
    context_timestep = torch.ones([1, nfb], device=noise.device,
                                  dtype=torch.int64) * pipeline.args.context_noise
    if capture_new_kv:
        causal_model.KV_NEW_CAPTURE = []
    pipeline.generator(
        noisy_image_or_video=denoised_pred,
        conditional_dict=cond,
        timestep=context_timestep,
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
        current_start=current_start,
    )
    captured = None
    if capture_new_kv:
        captured = causal_model.KV_NEW_CAPTURE
        causal_model.KV_NEW_CAPTURE = None
    return denoised_pred, captured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="configs/self_forcing_dmd_long.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    ap.add_argument("--data_path", default="prompts/quant3/prompts3.txt")
    ap.add_argument("--num_output_frames", type=int, default=63)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fakequant", default=None)
    ap.add_argument("--fakequant_group", type=int, default=128)
    ap.add_argument("--kv_quant", default=None)
    ap.add_argument("--kv_bits", type=int, default=4)
    ap.add_argument("--kv_block_size", type=int, default=16)
    ap.add_argument("--kv_quant_repo", default=os.path.expanduser("~/gpu/kv-quant-longhorizon"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--bitexact_gate", action="store_true",
                    help="Assert the second pass is bit-exact; only valid with no quantization")
    ap.add_argument("--max_prompts", type=int, default=3)
    ap.add_argument("--latent_dump_root", default=None,
                    help="Optional: dump TF chunk latents for re-analysis")
    args = ap.parse_args()

    out_dir = args.out_dir or f"results/tf_{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    torch.set_grad_enabled(False)
    set_seed(args.seed)
    device = torch.device("cuda")

    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load(args.config_path))
    pipeline = CausalInferencePipeline(config, device=device)
    state = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state["generator_ema" if args.use_ema else "generator"])
    del state
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device)

    # One-off text encoding; the 11 GB encoder goes back to CPU so both caches fit.
    dataset = TextDataset(prompt_path=args.data_path)
    prompts = [dataset[i]["prompts"] for i in range(min(len(dataset), args.max_prompts))]
    pipeline.text_encoder.to(device)
    conds = [pipeline.text_encoder(text_prompts=[p]) for p in prompts]
    pipeline.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    print(f"encoded {len(conds)} prompts; text encoder back on CPU")

    toggle = None
    if args.fakequant and args.fakequant not in ("bf16", "none"):
        from fakequant import apply_toggle_config
        toggle = apply_toggle_config(pipeline.generator.model, args.fakequant,
                                     group=args.fakequant_group)
    fq_active(False)

    kv_quantizer = None
    if args.kv_quant and args.kv_quant.upper() not in ("BF16", "NONE"):
        if args.kv_quant_repo not in sys.path:
            sys.path.insert(0, args.kv_quant_repo)
        from kv_quant.factory import create_quantizer
        kv_quantizer = create_quantizer(
            args.kv_quant, bits=args.kv_bits, block_size=args.kv_block_size,
            name=f"{args.kv_quant.upper()}_INT{args.kv_bits}")
        print(f"[kv_quant] {kv_quantizer.name()} block_size={args.kv_block_size}")

    is_plain = toggle is None and kv_quantizer is None
    if args.bitexact_gate and not is_plain:
        raise SystemExit("--bitexact_gate must run with no quantization configured")

    nfb = pipeline.num_frame_per_block
    assert args.num_output_frames % nfb == 0
    num_blocks = args.num_output_frames // nfb
    las = pipeline.local_attn_size
    kv_cache_size = las * pipeline.frame_seq_length if las != -1 else 32760
    nblk = pipeline.num_transformer_blocks
    print(f"chunks={num_blocks} frames/chunk={nfb} local_attn_size={las} "
          f"cache={kv_cache_size} tokens "
          f"({kv_cache_size // pipeline.frame_seq_length} frames)")

    per_chunk, mismatches, t0 = {}, 0, time.time()
    torch.cuda.reset_peak_memory_stats()

    for pi, (prompt, cond) in enumerate(zip(prompts, conds)):
        noise = torch.cat([
            torch.randn([1, nfb, 16, 60, 104],
                        generator=torch.Generator(device=device).manual_seed(
                            args.seed * 1000003 + c),
                        device=device, dtype=torch.bfloat16)
            for c in range(num_blocks)], dim=1)

        ref_cache = build_cache(kv_cache_size, torch.bfloat16, device, nblk)
        work_cache = build_cache(kv_cache_size, torch.bfloat16, device, nblk)
        ref_xattn = build_crossattn_cache(torch.bfloat16, device, nblk)
        work_xattn = build_crossattn_cache(torch.bfloat16, device, nblk)

        dump_dir = None
        if args.latent_dump_root:
            dump_dir = os.path.join(args.latent_dump_root, f"prompt{pi:03d}")
            os.makedirs(dump_dir, exist_ok=True)

        start_frame = 0
        for chunk_idx in range(num_blocks):
            # 1. Snapshot the context *before* the reference pass advances it.
            anchor_work_cache(work_cache, ref_cache, kv_quantizer, kv_cache_size, device)

            # 2. Reference pass — the only thing allowed to advance ref_cache.
            fq_active(False)
            ref_pred, ref_kv = run_chunk(pipeline, ref_cache, ref_xattn, cond, noise,
                                         chunk_idx, start_frame, nfb, args.seed,
                                         capture_new_kv=True)
            ref_kv_cpu = [(k.to("cpu", non_blocking=False), v.to("cpu", non_blocking=False))
                          for k, v in ref_kv]
            del ref_kv

            # 3. Teacher-forced quantized pass over the same context.
            fq_active(toggle is not None)
            q_pred, q_kv = run_chunk(pipeline, work_cache, work_xattn, cond, noise,
                                     chunk_idx, start_frame, nfb, args.seed,
                                     capture_new_kv=True)
            fq_active(False)

            # 4. Metrics.
            err_tf = rel(q_pred.float(), ref_pred.float())
            k_errs, v_errs = [], []
            for (rk, rv), (qk, qv) in zip(ref_kv_cpu, q_kv):
                k_errs.append(rel(qk.float(), rk.to(device).float()))
                v_errs.append(rel(qv.float(), rv.to(device).float()))
            del q_kv, ref_kv_cpu

            per_chunk.setdefault(chunk_idx, []).append({
                "err_tf": err_tf,
                "kv_new_k_rel": statistics.mean(k_errs),
                "kv_new_v_rel": statistics.mean(v_errs),
            })

            if args.bitexact_gate and not torch.equal(q_pred, ref_pred):
                mismatches += 1
                print(f"  MISMATCH prompt {pi} chunk {chunk_idx} "
                      f"max|d|={(q_pred - ref_pred).abs().max().item():.3e}")

            if dump_dir is not None:
                torch.save({"latent": q_pred.detach().float().cpu(),
                            "chunk_idx": chunk_idx},
                           os.path.join(dump_dir, f"chunk_{chunk_idx:04d}.pt"))

            start_frame += nfb

        print(f"prompt {pi}: done ({time.time() - t0:.0f}s elapsed, "
              f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB)")
        del ref_cache, work_cache, ref_xattn, work_xattn, noise
        torch.cuda.empty_cache()

    summary = {
        "tag": args.tag,
        "fakequant": args.fakequant or "bf16",
        "kv_quant": args.kv_quant or "bf16",
        "kv_bits": args.kv_bits,
        "seed": args.seed,
        "num_output_frames": args.num_output_frames,
        "local_attn_size": int(las),
        "eviction_chunk": kv_cache_size // pipeline.frame_seq_length // nfb,
        "seconds": time.time() - t0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "by_chunk": [{
            "chunk_idx": c,
            "n_prompts": len(v),
            "err_tf_mean": statistics.mean(x["err_tf"] for x in v),
            "err_tf_std": statistics.pstdev([x["err_tf"] for x in v]),
            "kv_new_k_rel_mean": statistics.mean(x["kv_new_k_rel"] for x in v),
            "kv_new_v_rel_mean": statistics.mean(x["kv_new_v_rel"] for x in v),
        } for c, v in sorted(per_chunk.items())],
    }
    path = os.path.join(out_dir, "tf.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {path}")

    if args.bitexact_gate:
        print(f"\nBIT-EXACT GATE: {'PASS' if mismatches == 0 else f'FAIL ({mismatches})'}")
        if mismatches:
            sys.exit(1)


if __name__ == "__main__":
    main()
