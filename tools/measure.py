"""Chunk-axis error curves: quantized runs vs the BF16 reference.

Reads the per-chunk latents dumped by the patched pipeline and reports, per
chunk index, the relative L2 error and the latent-space PSNR against BF16.

Layout expected (as produced by inference.py --latent_dump_root):
    <root>/<config>/prompt000/chunk_0000.pt, chunk_0001.pt, ...
"""
import argparse
import csv
import math
import os
import re
from collections import defaultdict

import torch


def load_chunks(prompt_dir):
    """-> {chunk_idx: tensor} for one prompt directory."""
    out = {}
    for fn in os.listdir(prompt_dir):
        m = re.fullmatch(r"chunk_(\d+)\.pt", fn)
        if not m:
            continue
        rec = torch.load(os.path.join(prompt_dir, fn), map_location="cpu")
        out[int(m.group(1))] = rec["latent"].float()
    return out


def rel_err(a, b):
    return (a - b).norm().item() / max(b.norm().item(), 1e-12)


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 0:
        return float("inf")
    # data range taken from the reference chunk itself
    rng = (b.max() - b.min()).item()
    return 20 * math.log10(max(rng, 1e-12)) - 10 * math.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Directory containing one subdirectory per config")
    ap.add_argument("--ref", default="bf16", help="Name of the reference config subdirectory")
    ap.add_argument("--out_csv", default="results/error_curves.csv")
    ap.add_argument("--out_png", default="results/error_curves.png")
    ap.add_argument("--local_attn_size", type=int, default=21,
                    help="Latent frames held in the rolling KV cache (for the boundary marker)")
    ap.add_argument("--num_frame_per_block", type=int, default=3)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Config subdirectories to skip entirely")
    ap.add_argument("--control_prefix", default="ctrl_",
                    help="Configs with this prefix are drawn as the chaotic-divergence baseline")
    args = ap.parse_args()

    ref_root = os.path.join(args.root, args.ref)
    if not os.path.isdir(ref_root):
        raise SystemExit(f"reference config dir not found: {ref_root}")

    configs = sorted(d for d in os.listdir(args.root)
                     if os.path.isdir(os.path.join(args.root, d))
                     and d != args.ref and d not in args.exclude)
    prompts = sorted(d for d in os.listdir(ref_root)
                     if os.path.isdir(os.path.join(ref_root, d)))
    print(f"reference={args.ref}  configs={configs}  prompts={prompts}")

    ref = {p: load_chunks(os.path.join(ref_root, p)) for p in prompts}

    rows = []
    for cfg in configs:
        # chunk_idx -> list over prompts
        errs, psnrs = defaultdict(list), defaultdict(list)
        for p in prompts:
            pdir = os.path.join(args.root, cfg, p)
            if not os.path.isdir(pdir):
                print(f"  ! {cfg}/{p} missing, skipped")
                continue
            got = load_chunks(pdir)
            for c, ref_t in sorted(ref[p].items()):
                if c not in got:
                    continue
                errs[c].append(rel_err(got[c], ref_t))
                psnrs[c].append(psnr(got[c], ref_t))

        for c in sorted(errs):
            e = torch.tensor(errs[c])
            s = torch.tensor(psnrs[c])
            rows.append({
                "config": cfg,
                "chunk_idx": c,
                "rel_err_mean": e.mean().item(),
                "rel_err_std": e.std(unbiased=False).item(),
                "psnr_mean": s.mean().item(),
                "n_prompts": len(errs[c]),
            })

    if not rows:
        raise SystemExit("no comparable chunks found")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    controls = [c for c in configs if c.startswith(args.control_prefix)]
    others = [c for c in configs if not c.startswith(args.control_prefix)]

    for cfg in others:
        pts = [r for r in rows if r["config"] == cfg]
        if not pts:
            continue
        xs = [r["chunk_idx"] for r in pts]
        ys = [r["rel_err_mean"] for r in pts]
        sd = [r["rel_err_std"] for r in pts]
        ax.errorbar(xs, ys, yerr=sd, marker="o", markersize=3.5,
                    capsize=2, linewidth=1.4, label=cfg)

    # Controls are BF16 everywhere, perturbed only in the initial noise. They
    # measure how fast this sampler diverges from *any* nudge, so a quantization
    # curve sitting on them carries no cache-specific error accumulation.
    for cfg in controls:
        pts = [r for r in rows if r["config"] == cfg]
        if not pts:
            continue
        xs = [r["chunk_idx"] for r in pts]
        ys = [r["rel_err_mean"] for r in pts]
        ax.plot(xs, ys, color="black", linestyle="--", linewidth=1.8,
                alpha=0.8, label=f"{cfg} (chaos floor)")

    # Above this, two same-norm signals are worse than uncorrelated.
    ax.axhline(2 ** 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(0.995, 2 ** 0.5, r"uncorrelated ($\sqrt{2}$) ", fontsize=8,
            va="bottom", ha="right", color="gray", transform=ax.get_yaxis_transform())

    # Where the rolling cache starts evicting the oldest chunk.
    if args.local_attn_size > 0 and args.num_frame_per_block > 0:
        boundary = args.local_attn_size / args.num_frame_per_block
        ax.axvline(boundary, color="k", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(boundary, ax.get_ylim()[1] * 0.97,
                f" KV eviction starts (chunk {boundary:g})",
                fontsize=8, va="top", alpha=0.75)

    ax.set_xlabel("chunk index")
    ax.set_ylabel(r"relative $L_2$ error vs BF16")
    ax.set_title("Quantization error along the frame (chunk) axis")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
