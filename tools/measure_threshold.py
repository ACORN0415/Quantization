"""Step I — where does the rollout break, as a function of cache error?

Session 1 bracketed it: 2.8% cache error (INT4) is safe, 13% (INT2) collapses.
This sweeps the gap two ways — bit-width, and block size at fixed bit-width —
and plots the injected cache error against what the rollout ends up at, so the
knee can be located and keys can be compared against values.
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

import torch


def cache_error(run_dir):
    """Steady-state (chunk >= 8) in-situ cache error, keys and values."""
    path = os.path.join(run_dir, "kv_err.json")
    if not os.path.exists(path):
        return None
    blob = json.load(open(path))
    agg = defaultdict(list)
    for r in blob["by_chunk"]:
        agg[r["chunk_idx"]].append((r["k_rel_mean"], r["v_rel_mean"]))
    tail = [c for c in sorted(agg) if c >= 8]
    if not tail:
        return None
    return (statistics.mean(statistics.mean(x[0] for x in agg[c]) for c in tail),
            statistics.mean(statistics.mean(x[1] for x in agg[c]) for c in tail))


def latent_error(latents_root, ref, cfg, chunk):
    ref_root = os.path.join(latents_root, ref)
    cfg_root = os.path.join(latents_root, cfg)
    if not os.path.isdir(cfg_root):
        return None
    vals = []
    for p in sorted(os.listdir(ref_root)):
        fa = os.path.join(ref_root, p, f"chunk_{chunk:04d}.pt")
        fb = os.path.join(cfg_root, p, f"chunk_{chunk:04d}.pt")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        a = torch.load(fa, map_location="cpu")["latent"].float()
        b = torch.load(fb, map_location="cpu")["latent"].float()
        vals.append(((b - a).norm() / a.norm()).item())
    return statistics.mean(vals) if vals else None


def iq_last(iq_csv, cfg):
    if not os.path.exists(iq_csv):
        return None
    best = None
    for row in csv.DictReader(open(iq_csv)):
        if row["config"] == cfg:
            c = int(row["chunk_idx"])
            if best is None or c > best[0]:
                best = (c, float(row["iq_mean"]))
    return best[1] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="tag=run_dir=latent_dir triples")
    ap.add_argument("--latents_root", default="results/latents63")
    ap.add_argument("--ref", default="bf16")
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--iq_csv", default="results/iq_curves.csv")
    ap.add_argument("--out_csv", default="results/threshold.csv")
    ap.add_argument("--out_png", default="results/threshold.png")
    ap.add_argument("--collapse_at", type=float, default=1.3,
                    help="latent rel_err above which session 1 saw visible collapse")
    args = ap.parse_args()

    rows = []
    for spec in args.runs:
        tag, run_dir, lat_dir = spec.split("=", 2)
        ce = cache_error(run_dir)
        if ce is None:
            print(f"  ! {tag}: no kv_err.json, skipped")
            continue
        le = latent_error(args.latents_root, args.ref, lat_dir, args.chunk)
        if le is None:
            print(f"  ! {tag}: no latents at {lat_dir}, skipped")
            continue
        rows.append({
            "config": tag,
            "k_cache_err": ce[0],
            "v_cache_err": ce[1],
            "mean_cache_err": (ce[0] + ce[1]) / 2,
            f"latent_rel_err_c{args.chunk}": le,
            "musiq_last": iq_last(args.iq_csv, lat_dir) or float("nan"),
        })
        print(f"{tag:14s} k={ce[0]:.4f} v={ce[1]:.4f} -> rel_err={le:.4f}")

    if not rows:
        raise SystemExit("nothing to plot")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat_key = f"latent_rel_err_c{args.chunk}"
    fig, (axk, axv) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, key, name in ((axk, "k_cache_err", "key"), (axv, "v_cache_err", "value")):
        for r in rows:
            ax.scatter(r[key], r[lat_key], s=60, zorder=3)
            ax.annotate(r["config"], (r[key], r[lat_key]), fontsize=7.5,
                        xytext=(5, 4), textcoords="offset points")
        ax.axhline(args.collapse_at, color="crimson", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel(f"in-situ {name} cache error (chunk >= 8)")
        ax.set_title(f"{name}s", fontsize=11)
        ax.grid(alpha=0.3)
    axk.set_ylabel(f"latent rel_err at chunk {args.chunk}")
    axk.text(axk.get_xlim()[0], args.collapse_at, " visible collapse in session 1",
             fontsize=8, va="bottom", color="crimson")
    fig.suptitle("Step I — cache error vs rollout damage", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
