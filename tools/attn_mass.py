"""Gate B — where does attention mass actually go?

The session-2 claim is that the first chunk acts as an implicit attention sink,
and that losing it at eviction is what raises the model's sensitivity to cache
quantization. So far that rests on error numbers alone. This measures the
attention distribution directly.

flash-attn never materializes the attention matrix, and building it would cost
~3.7 GB per layer (4680 queries x 32760 keys x 12 heads). Instead this walks the
keys in column blocks with an online-softmax accumulator, carrying only the
running max, the running denominator, and the partial numerator for the token
ranges of interest — exact, and O(query x block) memory.

Reported per layer and head, for one generator call:
  mass(range) = mean over queries of  sum_{k in range} softmax(qk/sqrt(d))
Compare against the range's share of tokens: if chunk 0 holds 1/7 of the cache
but takes 30% of the mass, it is a sink.
"""
import argparse
import json
import os
from collections import defaultdict

import torch


@torch.no_grad()
def masses_online(q, k, ranges, block=4096):
    """Exact softmax mass over token ranges, without forming the full matrix.

    q: [B, Lq, H, D]   k: [B, Lk, H, D]   ranges: {name: (start, end)}
    returns {name: tensor[H]} — mean over queries of the mass in that range.
    """
    b, lq, h, d = q.shape
    lk = k.shape[1]
    scale = 1.0 / (d ** 0.5)
    qf = q.float().permute(0, 2, 1, 3).reshape(b * h, lq, d) * scale
    kf = k.float().permute(0, 2, 1, 3).reshape(b * h, lk, d)

    run_max = torch.full((b * h, lq), -float("inf"), device=q.device)
    run_den = torch.zeros((b * h, lq), device=q.device)
    run_num = {name: torch.zeros((b * h, lq), device=q.device) for name in ranges}

    for s in range(0, lk, block):
        e = min(s + block, lk)
        logits = torch.bmm(qf, kf[:, s:e].transpose(1, 2))      # [B*H, Lq, e-s]
        blk_max = logits.amax(dim=-1)
        new_max = torch.maximum(run_max, blk_max)
        # rescale what we already accumulated to the new max
        rescale = torch.exp(run_max - new_max).masked_fill(run_max == -float("inf"), 0.0)
        run_den = run_den * rescale
        for name in ranges:
            run_num[name] = run_num[name] * rescale
        p = torch.exp(logits - new_max.unsqueeze(-1))
        run_den = run_den + p.sum(dim=-1)
        for name, (rs, re) in ranges.items():
            lo, hi = max(rs, s), min(re, e)
            if hi > lo:
                run_num[name] = run_num[name] + p[:, :, lo - s:hi - s].sum(dim=-1)
        run_max = new_max
        del logits, p

    out = {}
    for name in ranges:
        mass = (run_num[name] / run_den.clamp(min=1e-20))        # [B*H, Lq]
        out[name] = mass.reshape(b, h, lq).mean(dim=(0, 2))      # [H]
    return out


CAPTURE = None   # set to a dict by the patched attention to hand q/k over


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from omegaconf import OmegaConf
    import wan.modules.causal_model as cm
    from pipeline import CausalInferencePipeline
    from utils.dataset import TextDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="configs/gateA_A1.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    ap.add_argument("--data_path", default="prompts/quant3/prompts3.txt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num_output_frames", type=int, default=63)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_prompts", type=int, default=3)
    ap.add_argument("--chunks", nargs="+", type=int, default=[6, 7, 14, 20],
                    help="chunk indices at which to record")
    ap.add_argument("--kv_quant", default=None)
    ap.add_argument("--kv_bits", type=int, default=4)
    ap.add_argument("--kv_quant_repo", default=os.path.expanduser("~/gpu/kv-quant-longhorizon"))
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or f"results/attn_mass/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load(args.config_path))
    pipeline = CausalInferencePipeline(config, device=device)
    state = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state["generator_ema"])
    del state
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device)

    ds = TextDataset(prompt_path=args.data_path)
    prompts = [ds[i]["prompts"] for i in range(min(len(ds), args.max_prompts))]
    pipeline.text_encoder.to(device)
    conds = [pipeline.text_encoder(text_prompts=[p]) for p in prompts]
    pipeline.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    if args.kv_quant:
        import sys as _s
        if args.kv_quant_repo not in _s.path:
            _s.path.insert(0, args.kv_quant_repo)
        from kv_quant.factory import create_quantizer
        pipeline.kv_quantizer = create_quantizer(
            args.kv_quant, bits=args.kv_bits, block_size=16,
            name=f"{args.kv_quant.upper()}_INT{args.kv_bits}")
        print(f"[kv_quant] {pipeline.kv_quantizer.name()}")

    fsl = pipeline.frame_seq_length
    nfb = pipeline.num_frame_per_block
    las = pipeline.local_attn_size
    sink_frames = getattr(pipeline.generator.model, "sink_size", 0)
    cache_tokens = las * fsl if las != -1 else 32760
    chunk_tokens = nfb * fsl
    print(f"cache={cache_tokens} tokens, chunk={chunk_tokens}, "
          f"sink={sink_frames} frames, capacity={cache_tokens // chunk_tokens} chunks")

    records = []

    def hook(q, k, current_start, local_end_index):
        """Called from the patched attention with the post-RoPE q and cache k."""
        cs_chunk = current_start // chunk_tokens
        if cs_chunk not in args.chunks:
            return
        n = int(local_end_index)
        ranges = {
            "slot0_chunk": (0, min(chunk_tokens, n)),
            "sink": (0, min(sink_frames * fsl, n)) if sink_frames else (0, 0),
            "newest_chunk": (max(0, n - chunk_tokens), n),
            "all": (0, n),
        }
        ranges = {kk: vv for kk, vv in ranges.items() if vv[1] > vv[0]}
        m = masses_online(q, k[:, :n], ranges)
        records.append({
            "chunk": cs_chunk,
            "tokens": n,
            "slot0_share": min(chunk_tokens, n) / n,
            **{f"{kk}_per_head": [float(x) for x in vv.cpu()] for kk, vv in m.items()},
        })

    cm.ATTN_MASS_HOOK = hook

    for pi, (prompt, cond) in enumerate(zip(prompts, conds)):
        noise = torch.cat([
            torch.randn([1, nfb, 16, 60, 104],
                        generator=torch.Generator(device=device).manual_seed(args.seed * 1000003 + c),
                        device=device, dtype=torch.bfloat16)
            for c in range(args.num_output_frames // nfb)], dim=1)
        cm.ATTN_MASS_PROMPT = pi
        try:
            pipeline.inference(noise=noise, text_prompts=[prompt],
                               return_latents=True, base_seed=args.seed)
        except torch.OutOfMemoryError:
            print("(VAE decode OOM ignored)")
        print(f"prompt {pi}: {len(records)} records so far")
        del noise
        torch.cuda.empty_cache()

    cm.ATTN_MASS_HOOK = None

    # fold: mean over prompts and generator calls, keeping layer index
    by_chunk = defaultdict(list)
    for r in records:
        by_chunk[r["chunk"]].append(r)
    summary = {"tag": args.tag, "config": args.config_path,
               "sink_frames": int(sink_frames), "cache_tokens": int(cache_tokens),
               "chunk_tokens": int(chunk_tokens), "by_chunk": []}
    for c in sorted(by_chunk):
        rs = by_chunk[c]
        entry = {"chunk": c, "n_records": len(rs),
                 "slot0_token_share": sum(r["slot0_share"] for r in rs) / len(rs)}
        for key in ("slot0_chunk", "sink", "newest_chunk"):
            vals = [sum(r[f"{key}_per_head"]) / len(r[f"{key}_per_head"])
                    for r in rs if f"{key}_per_head" in r]
            if vals:
                entry[f"{key}_mass_mean"] = sum(vals) / len(vals)
                entry[f"{key}_mass_max"] = max(vals)
        summary["by_chunk"].append(entry)

    with open(os.path.join(out_dir, "attn_mass.json"), "w") as f:
        json.dump({"summary": summary, "records": records}, f)
    print(f"wrote {out_dir}/attn_mass.json")
    for e in summary["by_chunk"]:
        share = e["slot0_token_share"]
        mass = e.get("slot0_chunk_mass_mean", float("nan"))
        print(f"  chunk {e['chunk']:2d}: first-chunk tokens {share*100:5.1f}% of cache, "
              f"attention mass {mass*100:5.1f}%  (ratio {mass/share:.2f}x)")


if __name__ == "__main__":
    main()
