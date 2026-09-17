"""Gate B — where does attention mass actually go?

The session-2 claim is that the first chunk acts as an implicit attention sink,
and that losing it at eviction is what raises sensitivity to cache quantization.
So far that rests on error numbers alone. This reads the distribution directly.

Design constraints (session 3 handoff, Gate B note):
  * the generation forward keeps using FlashAttention-2 unchanged
  * the hook receives the post-RoPE Q/K and recomputes softmax(QK^T/sqrt(d))
    purely for measurement, collapsing the key axis into per-source-chunk sums
    immediately so only scalars survive
  * before any measurement, one layer's recomputed softmax(QK^T/sqrt(d))V is
    checked against the FA2 output with torch.allclose(atol=1e-2)

The accumulation uses an online softmax over key blocks, so the full matrix is
never held: exact, and O(queries x block) instead of ~306 MB per head.

Baselines to compare the mass against, i.e. the chunk's share of cache tokens:
  Self-Forcing  1 chunk / 7  = 14.3 %   (21-frame cache, 3 frames per chunk)
  LongLive      sink 3 / 12  = 25.0 %
"""
import argparse
import json
import os
from collections import defaultdict

import torch


@torch.no_grad()
def masses_online(q, k, ranges, block=4096):
    """Exact softmax mass per token range, without forming the full matrix.

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
        logits = torch.bmm(qf, kf[:, s:e].transpose(1, 2))
        blk_max = logits.amax(dim=-1)
        new_max = torch.maximum(run_max, blk_max)
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
        mass = run_num[name] / run_den.clamp(min=1e-20)
        out[name] = mass.reshape(b, h, lq).mean(dim=(0, 2))
    return out


@torch.no_grad()
def verify_against_fa2(q, k, v, atol=1e-2, rtol=1e-2):
    """Recomputed softmax(QK^T/sqrt(d))V must match what the model actually ran.

    Guards against the wrong tensors, a wrong scale, or a mask mismatch.

    Mask: the model calls `attention(q, k, v)` with no `causal` argument, and
    `attention` defaults to causal=False — every cached token and every token of
    the current chunk is visible to every query (chunks are bidirectional
    internally). The recomputation therefore applies NO mask. Imposing a
    standard causal mask here would cut the current chunk to a lower triangle
    and silently disagree with FA2.

    Q and K are taken after norm_q/norm_k and after RoPE, matching exactly what
    is handed to FA2; capturing them earlier would compare different scales.

    Compared in fp32 with both atol and rtol, since FA2 accumulates in bf16.
    One head at a time keeps the probability matrix near 300 MB.
    """
    from wan.modules.attention import attention
    ref = attention(q, k, v).float()              # the FA2 path the model uses
    b, lq, h, d = q.shape
    scale = 1.0 / (d ** 0.5)
    max_abs, ok = 0.0, True
    for hi in range(h):
        qi = q[:, :, hi].float() * scale
        ki = k[:, :, hi].float()
        vi = v[:, :, hi].float()
        probs = torch.softmax(torch.bmm(qi, ki.transpose(1, 2)), dim=-1)
        out = torch.bmm(probs, vi)
        ri = ref[:, :, hi]
        max_abs = max(max_abs, (out - ri).abs().max().item())
        ok = ok and torch.allclose(out, ri, atol=atol, rtol=rtol)
        del probs, out
    return ok, max_abs


def slot_ranges(window_len, chunk_tokens, sink_tokens, global_end):
    """Token ranges for each cache slot, plus which global chunk each holds.

    Before eviction the cache is just the prefix, so slot i holds global chunk i.
    After eviction the sink region stays pinned to the first chunks while the
    rest of the window slides, so the mapping has to be built from the window's
    global end rather than assumed.
    """
    ranges, labels = {}, {}
    n_slots = (window_len + chunk_tokens - 1) // chunk_tokens
    n_sink_slots = sink_tokens // chunk_tokens
    # global index of the token sitting at local position 0 of the rolling part
    rolling_len = window_len - sink_tokens
    rolling_global_start = global_end - rolling_len
    for i in range(n_slots):
        lo = i * chunk_tokens
        hi = min(lo + chunk_tokens, window_len)
        if hi <= lo:
            continue
        name = f"slot{i}"
        ranges[name] = (lo, hi)
        if i < n_sink_slots:
            labels[name] = i                       # pinned sink chunk
        else:
            labels[name] = (rolling_global_start + (lo - sink_tokens)) // chunk_tokens
    # The current chunk's K/V are written into the cache before attention runs,
    # so the last slot is the query's own chunk. Queries attend to it too
    # (chunks are bidirectional internally) and it must stay in the denominator:
    # dropping it would inflate every past chunk's share.
    if ranges:
        last = f"slot{max(int(n[4:]) for n in ranges)}"
        labels[last] = "self"
    return ranges, labels


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
    ap.add_argument("--data_path", default="prompts/quant3/prompts10.txt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num_output_frames", type=int, default=63)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_prompts", type=int, default=3)
    ap.add_argument("--chunks", nargs="+", type=int, default=[6, 7, 14, 20])
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
    sink_frames = int(getattr(pipeline.generator.model, "sink_size", 0) or 0)
    cache_tokens = las * fsl if las != -1 else 32760
    chunk_tokens = nfb * fsl
    sink_tokens = sink_frames * fsl
    capacity = cache_tokens // chunk_tokens
    print(f"cache={cache_tokens} tokens ({capacity} chunks), chunk={chunk_tokens}, "
          f"sink={sink_frames} frames ({sink_tokens} tokens)")
    print(f"baseline: one chunk is {100.0 / capacity:.1f}% of the cache; "
          f"sink is {100.0 * sink_tokens / cache_tokens:.1f}%")

    n_layers = pipeline.num_transformer_blocks
    denoise_steps = [float(t) for t in pipeline.denoising_step_list]
    records = []
    state = {"verified": False, "verify_err": None,
             "chunk": None, "layer": 0, "step": 0}

    def hook(q, k, v, current_start, w0, local_end_index):
        # Each generator call walks all layers once, so counting hook calls
        # recovers which call we are in. Calls 0..3 are the denoising steps
        # (t = 1000/750/500/250 warped); call 4 is the clean-context refresh.
        cur = current_start // chunk_tokens
        if cur != state["chunk"]:
            state["chunk"], state["layer"], state["step"] = cur, 0, 0
        layer_idx = state["layer"]
        step_idx = state["step"]
        state["layer"] += 1
        if state["layer"] >= n_layers:
            state["layer"] = 0
            state["step"] += 1
        # One-time equivalence check before trusting any measurement.
        if not state["verified"]:
            ok, err = verify_against_fa2(q, k, v)
            state["verified"], state["verify_err"] = True, err
            print(f"[verify] recomputed softmax(QK^T/sqrt(d))V vs FA2: "
                  f"max|diff|={err:.4e} -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                raise SystemExit(f"FA2 equivalence check failed (max|diff|={err:.3e})")

        cur_chunk = cur
        if cur_chunk not in args.chunks:
            return
        window_len = int(local_end_index - w0)
        # The last token in the cache is the end of the current chunk, which is
        # what anchors the slot -> global chunk mapping. Using local_end_index
        # instead breaks once the cache is full and that pointer stops moving.
        ranges, labels = slot_ranges(window_len, chunk_tokens, sink_tokens,
                                     int(current_start) + chunk_tokens)
        if not ranges:
            return
        m = masses_online(q, k, ranges)
        records.append({
            "chunk": cur_chunk,
            "layer": layer_idx,
            "step": step_idx,
            "step_timestep": (denoise_steps[step_idx] if step_idx < len(denoise_steps)
                              else "context_refresh"),
            "window_len": window_len,
            "slots": {name: {"mass_per_head": [float(x) for x in m[name].cpu()],
                             "token_share": (ranges[name][1] - ranges[name][0]) / window_len,
                             "global_chunk": labels[name]}
                      for name in ranges},
        })

    cm.ATTN_MASS_HOOK = hook

    # Drive the chunk loop directly with the pre-encoded conditioning: calling
    # pipeline.inference() would re-encode the prompt and pull the text encoder
    # back onto a device it is no longer on. No VAE decode either — only the
    # attention distribution is wanted.
    nblk = pipeline.num_transformer_blocks
    cache_size = cache_tokens
    for pi, (prompt, cond) in enumerate(zip(prompts, conds)):
        noise = torch.cat([
            torch.randn([1, nfb, 16, 60, 104],
                        generator=torch.Generator(device=device).manual_seed(args.seed * 1000003 + c),
                        device=device, dtype=torch.bfloat16)
            for c in range(args.num_output_frames // nfb)], dim=1)
        kv = [{
            "k": torch.zeros([1, cache_size, 12, 128], dtype=torch.bfloat16, device=device),
            "v": torch.zeros([1, cache_size, 12, 128], dtype=torch.bfloat16, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "quantizer": getattr(pipeline, "kv_quantizer", None), "quant_state": None,
        } for _ in range(nblk)]
        xattn = [{"k": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
                  "v": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
                  "is_init": False} for _ in range(nblk)]
        start = 0
        for c in range(args.num_output_frames // nfb):
            g = torch.Generator(device=device)
            g.manual_seed(args.seed * 1000003 + 7919 + c)
            x = noise[:, start:start + nfb]
            pred = None
            for i, t in enumerate(pipeline.denoising_step_list):
                ts = torch.ones([1, nfb], device=device, dtype=torch.int64) * t
                _, pred = pipeline.generator(
                    noisy_image_or_video=x, conditional_dict=cond, timestep=ts,
                    kv_cache=kv, crossattn_cache=xattn, current_start=start * fsl)
                if i < len(pipeline.denoising_step_list) - 1:
                    flat = pred.flatten(0, 1)
                    rn = torch.randn(flat.shape, generator=g, device=device, dtype=flat.dtype)
                    x = pipeline.scheduler.add_noise(
                        flat, rn,
                        pipeline.denoising_step_list[i + 1] *
                        torch.ones([nfb], device=device, dtype=torch.long)
                    ).unflatten(0, pred.shape[:2])
            ctx = torch.ones([1, nfb], device=device, dtype=torch.int64) * pipeline.args.context_noise
            pipeline.generator(noisy_image_or_video=pred, conditional_dict=cond, timestep=ctx,
                               kv_cache=kv, crossattn_cache=xattn, current_start=start * fsl)
            start += nfb
        print(f"prompt {pi}: {len(records)} records")
        del noise, kv, xattn
        torch.cuda.empty_cache()

    cm.ATTN_MASS_HOOK = None

    by_chunk = defaultdict(list)
    for r in records:
        by_chunk[(r["chunk"], r["step"])].append(r)

    summary = {"tag": args.tag, "config": args.config_path,
               "sink_frames": sink_frames, "cache_tokens": cache_tokens,
               "chunk_tokens": chunk_tokens, "capacity_chunks": capacity,
               "fa2_verify_max_abs_diff": state["verify_err"], "by_chunk": []}
    for c, stp in sorted(by_chunk):
        rs = by_chunk[(c, stp)]
        slots = defaultdict(list)
        shares, gchunk = {}, {}
        for r in rs:
            for name, d in r["slots"].items():
                slots[name].append(sum(d["mass_per_head"]) / len(d["mass_per_head"]))
                shares[name] = d["token_share"]
                gchunk[name] = d["global_chunk"]
        entry = {"chunk": c, "step": stp,
                 "step_timestep": rs[0]["step_timestep"],
                 "n_records": len(rs), "slots": {}}
        for name in sorted(slots, key=lambda x: int(x[4:])):
            mean = sum(slots[name]) / len(slots[name])
            entry["slots"][name] = {
                "mass_mean": mean, "token_share": shares[name],
                "ratio": mean / shares[name] if shares[name] else float("nan"),
                "global_chunk": gchunk[name],
            }
        # No bucket may be missing: the slots partition the whole window, so
        # their masses must sum to 1 (the self bucket included).
        entry["mass_sum"] = sum(d["mass_mean"] for d in entry["slots"].values())
        entry["share_sum"] = sum(d["token_share"] for d in entry["slots"].values())
        summary["by_chunk"].append(entry)

    with open(os.path.join(out_dir, "attn_mass.json"), "w") as f:
        json.dump({"summary": summary, "records": records}, f)
    print(f"\nwrote {out_dir}/attn_mass.json")
    for e in summary["by_chunk"]:
        flag = "" if abs(e["mass_sum"] - 1.0) < 1e-3 else "   !! BUCKET MISSING"
        print(f"\nchunk {e['chunk']} step {e['step']} (t={e['step_timestep']})"
              f"  mass_sum={e['mass_sum']:.4f} share_sum={e['share_sum']:.4f}{flag}")
        for name, d in e["slots"].items():
            print(f"  {name:6s} gchunk={str(d['global_chunk']):>5s}  "
                  f"mass {d['mass_mean']*100:5.1f}%  share {d['token_share']*100:5.1f}%  "
                  f"ratio {d['ratio']:.2f}x")


if __name__ == "__main__":
    main()
