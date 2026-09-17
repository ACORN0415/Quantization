"""Gate E1 — how contaminated is each cache slot, by age?

The "clean-zero anchor" hypothesis says a sink helps at INT4 not because it
attracts attention, but because chunk 0 is the one chunk generated from pure
noise with no prior context, so its quantization error is nearly zero. Pinning
it permanently reserves 7-8% of the attention for uncontaminated content, while
an unpinned oldest slot keeps being refilled with chunks that carry accumulated
error.

If that is right, A1's contamination should rise monotonically with slot age
while A2's oldest slot (the pinned chunk 0) stays near zero.

Unlike the teacher-forced runs this is free-running on both sides: a BF16
reference and a quantized run advance in lockstep with their own caches, and
the caches are compared slot by slot at the requested chunks. Re-anchoring would
erase exactly the quantity being measured.
"""
import argparse
import json
import os
import statistics
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed


def build_cache(size, device, nblk, quantizer=None):
    return [{
        "k": torch.zeros([1, size, 12, 128], dtype=torch.bfloat16, device=device),
        "v": torch.zeros([1, size, 12, 128], dtype=torch.bfloat16, device=device),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "quantizer": quantizer, "quant_state": None,
    } for _ in range(nblk)]


def build_xattn(device, nblk):
    return [{"k": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
             "v": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
             "is_init": False} for _ in range(nblk)]


def live_contents(cache, quantizer, n):
    """The K/V a forward pass would actually see, for the first n tokens."""
    if quantizer is not None and cache.get("quant_state") is not None:
        k, v = quantizer.dequantize_kv(cache["quant_state"],
                                       meta={"tensor_dtype": torch.bfloat16})
        return k[:, :n].float(), v[:, :n].float()
    return cache["k"][:, :n].float(), cache["v"][:, :n].float()


def run_chunk(pipe, kv, xattn, cond, noise, chunk_idx, start, nfb, fsl, seed):
    g = torch.Generator(device=noise.device)
    g.manual_seed(seed * 1000003 + 7919 + chunk_idx)
    x = noise[:, start:start + nfb]
    pred = None
    for i, t in enumerate(pipe.denoising_step_list):
        ts = torch.ones([1, nfb], device=noise.device, dtype=torch.int64) * t
        _, pred = pipe.generator(noisy_image_or_video=x, conditional_dict=cond,
                                 timestep=ts, kv_cache=kv, crossattn_cache=xattn,
                                 current_start=start * fsl)
        if i < len(pipe.denoising_step_list) - 1:
            flat = pred.flatten(0, 1)
            rn = torch.randn(flat.shape, generator=g, device=noise.device, dtype=flat.dtype)
            x = pipe.scheduler.add_noise(
                flat, rn,
                pipe.denoising_step_list[i + 1] *
                torch.ones([nfb], device=noise.device, dtype=torch.long)
            ).unflatten(0, pred.shape[:2])
    ctx = torch.ones([1, nfb], device=noise.device, dtype=torch.int64) * pipe.args.context_noise
    pipe.generator(noisy_image_or_video=pred, conditional_dict=cond, timestep=ctx,
                   kv_cache=kv, crossattn_cache=xattn, current_start=start * fsl)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="configs/gateA_A1.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    ap.add_argument("--data_path", default="prompts/quant3/prompts10.txt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num_output_frames", type=int, default=63)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_prompts", type=int, default=10)
    ap.add_argument("--at_chunks", nargs="+", type=int, default=[8, 14, 20])
    ap.add_argument("--kv_quant", default="RTN")
    ap.add_argument("--kv_bits", type=int, default=4)
    ap.add_argument("--kv_quant_repo", default=os.path.expanduser("~/gpu/kv-quant-longhorizon"))
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or f"results/gateE/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    torch.set_grad_enabled(False)
    set_seed(args.seed)
    device = torch.device("cuda")
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load(args.config_path))
    pipe = CausalInferencePipeline(config, device=device)
    state = torch.load(args.checkpoint_path, map_location="cpu")
    pipe.generator.load_state_dict(state["generator_ema"])
    del state
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.generator.to(device)

    ds = TextDataset(prompt_path=args.data_path)
    prompts = [ds[i]["prompts"] for i in range(min(len(ds), args.max_prompts))]
    pipe.text_encoder.to(device)
    conds = [pipe.text_encoder(text_prompts=[p]) for p in prompts]
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    sys.path.insert(0, args.kv_quant_repo)
    from kv_quant.factory import create_quantizer
    quantizer = create_quantizer(args.kv_quant, bits=args.kv_bits, block_size=16,
                                 key_bits=args.kv_bits, value_bits=args.kv_bits,
                                 name=f"{args.kv_quant.upper()}_INT{args.kv_bits}")

    fsl, nfb = pipe.frame_seq_length, pipe.num_frame_per_block
    las = pipe.local_attn_size
    cache_size = las * fsl if las != -1 else 32760
    sink_tokens = int(getattr(pipe.generator.model, "sink_size", 0) or 0) * fsl
    chunk_tokens = nfb * fsl
    nblk = pipe.num_transformer_blocks
    n_slots = cache_size // chunk_tokens
    print(f"cache={cache_size} ({n_slots} slots), sink={sink_tokens} tokens, "
          f"quantizer={quantizer.name()}")

    per = {c: {i: {"k": [], "v": []} for i in range(n_slots)} for c in args.at_chunks}

    for pi, (prompt, cond) in enumerate(zip(prompts, conds)):
        noise = torch.cat([
            torch.randn([1, nfb, 16, 60, 104],
                        generator=torch.Generator(device=device).manual_seed(args.seed * 1000003 + c),
                        device=device, dtype=torch.bfloat16)
            for c in range(args.num_output_frames // nfb)], dim=1)

        ref_kv, ref_x = build_cache(cache_size, device, nblk), build_xattn(device, nblk)
        q_kv, q_x = build_cache(cache_size, device, nblk, quantizer), build_xattn(device, nblk)

        start = 0
        for c in range(args.num_output_frames // nfb):
            run_chunk(pipe, ref_kv, ref_x, cond, noise, c, start, nfb, fsl, args.seed)
            run_chunk(pipe, q_kv, q_x, cond, noise, c, start, nfb, fsl, args.seed)
            start += nfb

            if c in per:
                n = int(ref_kv[0]["local_end_index"].item())
                for slot in range(n_slots):
                    lo, hi = slot * chunk_tokens, min((slot + 1) * chunk_tokens, n)
                    if hi <= lo:
                        continue
                    ks, vs = [], []
                    for L in range(nblk):
                        rk, rv = live_contents(ref_kv[L], None, n)
                        qk, qv = live_contents(q_kv[L], quantizer, n)
                        ks.append(((qk[:, lo:hi] - rk[:, lo:hi]).norm()
                                   / rk[:, lo:hi].norm().clamp(min=1e-12)).item())
                        vs.append(((qv[:, lo:hi] - rv[:, lo:hi]).norm()
                                   / rv[:, lo:hi].norm().clamp(min=1e-12)).item())
                    # slot 0 is the oldest position in the cache
                    per[c][slot]["k"].append(statistics.mean(ks))
                    per[c][slot]["v"].append(statistics.mean(vs))
        print(f"prompt {pi} done")
        del noise, ref_kv, ref_x, q_kv, q_x
        torch.cuda.empty_cache()

    out = {"tag": args.tag, "config": args.config_path, "kv": quantizer.name(),
           "sink_tokens": sink_tokens, "n_slots": n_slots, "by_chunk": []}
    for c in sorted(per):
        entry = {"chunk": c, "slots": []}
        for slot in range(n_slots):
            k, v = per[c][slot]["k"], per[c][slot]["v"]
            if not k:
                continue
            entry["slots"].append({
                "slot": slot, "age_rank": n_slots - 1 - slot,   # 0 = newest
                "k_rel_mean": statistics.mean(k), "v_rel_mean": statistics.mean(v),
                "n_prompts": len(k),
            })
        out["by_chunk"].append(entry)

    with open(os.path.join(out_dir, "slot_contamination.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_dir}/slot_contamination.json")
    for e in out["by_chunk"]:
        print(f"\nchunk {e['chunk']} (slot 0 = oldest position)")
        for s in e["slots"]:
            print(f"  slot{s['slot']} (age {s['age_rank']}):  "
                  f"K {s['k_rel_mean']:.4f}   V {s['v_rel_mean']:.4f}")


if __name__ == "__main__":
    main()
