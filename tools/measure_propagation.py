"""Step F — propagation = free-running error minus teacher-forced error.

TF re-anchors the context to the BF16 reference at every chunk, so its error is
what quantization injects at that chunk alone. FR lets the context drift. The
difference is what earlier chunks handed forward, with the chaotic-divergence
confound removed from the TF side by construction.
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

import torch


def fr_curve(latents_root, ref, cfg):
    """Per-chunk mean relative error of a free-running run against BF16."""
    ref_root = os.path.join(latents_root, ref)
    cfg_root = os.path.join(latents_root, cfg)
    if not os.path.isdir(cfg_root):
        return None
    per = defaultdict(list)
    for p in sorted(os.listdir(ref_root)):
        if not os.path.isdir(os.path.join(cfg_root, p)):
            continue
        for fn in sorted(os.listdir(os.path.join(ref_root, p))):
            if not fn.startswith("chunk_"):
                continue
            fb = os.path.join(cfg_root, p, fn)
            if not os.path.exists(fb):
                continue
            a = torch.load(os.path.join(ref_root, p, fn), map_location="cpu")["latent"].float()
            b = torch.load(fb, map_location="cpu")["latent"].float()
            per[int(fn[6:10])].append(((b - a).norm() / a.norm()).item())
    return {c: statistics.mean(v) for c, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf_dirs", nargs="+", required=True,
                    help="results/tf_<tag> directories holding tf.json")
    ap.add_argument("--latents_root", default="results/latents63")
    ap.add_argument("--ref", default="bf16")
    ap.add_argument("--map", nargs="+", default=[],
                    help="tag=fr_config pairs, e.g. kv_int4=kv_int4 w4a4=fq_w4a4")
    ap.add_argument("--out_csv", default="results/propagation.csv")
    ap.add_argument("--out_png", default="results/propagation.png")
    ap.add_argument("--eviction_chunk", type=int, default=7)
    args = ap.parse_args()

    mapping = dict(kv.split("=", 1) for kv in args.map)

    rows, series = [], {}
    for d in args.tf_dirs:
        path = os.path.join(d, "tf.json")
        if not os.path.exists(path):
            print(f"  ! {path} missing, skipped")
            continue
        tf = json.load(open(path))
        tag = tf["tag"]
        fr_cfg = mapping.get(tag, tag)
        fr = fr_curve(args.latents_root, args.ref, fr_cfg)
        if fr is None:
            print(f"  ! free-running run '{fr_cfg}' for tag '{tag}' not found, skipped")
            continue

        tf_by_chunk = {r["chunk_idx"]: r for r in tf["by_chunk"]}
        pts = []
        for c in sorted(set(tf_by_chunk) & set(fr)):
            e_tf = tf_by_chunk[c]["err_tf_mean"]
            e_fr = fr[c]
            pts.append((c, e_tf, e_fr, e_fr - e_tf))
            rows.append({
                "config": tag,
                "chunk_idx": c,
                "err_TF_mean": e_tf,
                "err_TF_std": tf_by_chunk[c]["err_tf_std"],
                "err_FR_mean": e_fr,
                "propagation": e_fr - e_tf,
                "kv_new_k_rel_mean": tf_by_chunk[c]["kv_new_k_rel_mean"],
                "kv_new_v_rel_mean": tf_by_chunk[c]["kv_new_v_rel_mean"],
            })
        series[tag] = pts
        print(f"{tag}: {len(pts)} chunks (FR source: {fr_cfg})")

    if not rows:
        raise SystemExit("nothing to write")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    for tag, pts in sorted(series.items()):
        xs = [p[0] for p in pts]
        ax1.plot(xs, [p[1] for p in pts], marker="o", markersize=3, linewidth=1.4, label=tag)
        ax2.plot(xs, [p[2] for p in pts], marker="o", markersize=3, linewidth=1.4, label=tag)
        ax3.plot(xs, [p[3] for p in pts], marker="o", markersize=3, linewidth=1.6, label=tag)

    for ax, title in ((ax1, "teacher-forced: injected per chunk"),
                      (ax2, "free-running: injected + propagated"),
                      (ax3, "propagation = FR - TF")):
        ax.axvline(args.eviction_chunk, color="k", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("chunk index")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax3.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
    ax1.set_ylabel(r"relative $L_2$ error vs BF16")
    ax1.text(args.eviction_chunk, ax1.get_ylim()[1] * 0.97,
             f" eviction (chunk {args.eviction_chunk})", fontsize=8, va="top", alpha=0.75)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
