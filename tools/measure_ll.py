"""Step H — Self-Forcing vs LongLive, same quantization grid, different cache design.

Self-Forcing: no sink, 21-frame window, eviction from chunk 7.
LongLive:     3-frame sink + 9-frame window, eviction from chunk 4.

Both curves are read against their own noise-perturbation control, which is the
chaotic-divergence floor for that model. Ratio to floor is the comparable
quantity; absolute rel_err is not, because the two models diverge at different
rates.
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

import torch


def curve(latents_root, ref, cfg):
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
    return {k: statistics.mean(v) for k, v in per.items()} or None


def cache_err(path):
    if not os.path.exists(path):
        return None
    blob = json.load(open(path))
    agg = defaultdict(list)
    for r in blob["by_chunk"]:
        agg[r["chunk_idx"]].append((r["k_rel_mean"], r["v_rel_mean"]))
    tail = [c for c in sorted(agg) if c >= 5]
    if not tail:
        return None
    return (statistics.mean(statistics.mean(x[0] for x in agg[c]) for c in tail),
            statistics.mean(statistics.mean(x[1] for x in agg[c]) for c in tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ll_root", default="results/latents_ll")
    ap.add_argument("--sf_root", default=os.path.expanduser("~/gpu/Self-Forcing/results/latents63"))
    ap.add_argument("--ll_configs", nargs="+",
                    default=["ctrl1em3", "kv_int4", "kv_int2", "w4a4", "sink_bf16_int2"])
    ap.add_argument("--sf_configs", nargs="+",
                    default=["ctrl_perturb1em3", "kv_int4", "kv_int2", "fq_w4a4"])
    ap.add_argument("--ll_evict", type=int, default=4)
    ap.add_argument("--sf_evict", type=int, default=7)
    ap.add_argument("--out_csv", default="results/longlive_vs_selfforcing.csv")
    ap.add_argument("--out_png", default="results/longlive_vs_selfforcing.png")
    args = ap.parse_args()

    ll = {c: curve(args.ll_root, "bf16", c) for c in args.ll_configs}
    sf = {c: curve(args.sf_root, "bf16", c) for c in args.sf_configs}
    ll = {k: v for k, v in ll.items() if v}
    sf = {k: v for k, v in sf.items() if v}

    ll_floor = ll.get("ctrl1em3")
    sf_floor = sf.get("ctrl_perturb1em3")
    if not (ll_floor and sf_floor):
        raise SystemExit("missing a control run")

    rows = []
    for model, data, floor, ev in (("LongLive", ll, ll_floor, args.ll_evict),
                                   ("Self-Forcing", sf, sf_floor, args.sf_evict)):
        for cfg, d in data.items():
            for c in sorted(d):
                rows.append({
                    "model": model, "config": cfg, "chunk_idx": c,
                    "rel_err": d[c],
                    "floor": floor.get(c, float("nan")),
                    "ratio_to_floor": d[c] / floor[c] if floor.get(c) else float("nan"),
                    "eviction_chunk": ev,
                })

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (title, data, floor, ev) in zip(
            axes[:2],
            (("Self-Forcing  (no sink, window 21, evict @7)", sf, sf_floor, args.sf_evict),
             ("LongLive  (sink 3 + window 9, evict @4)", ll, ll_floor, args.ll_evict))):
        for cfg, d in sorted(data.items()):
            cs = sorted(d)
            style = (dict(linestyle="--", color="black", linewidth=1.8)
                     if cfg.startswith("ctrl") else
                     dict(marker="o", markersize=3, linewidth=1.4))
            ax.plot(cs, [d[c] for c in cs], label=cfg, **style)
        ax.axvline(ev, color="k", linestyle=":", linewidth=1.2, alpha=0.6)
        ax.set_xlabel("chunk index")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 2.0)
    axes[0].set_ylabel(r"relative $L_2$ error vs BF16")

    # Ratio to each model's own floor — the comparable quantity.
    ax = axes[2]
    for cfg, d, sty in (("SF INT2", sf.get("kv_int2"), dict(color="tab:red")),
                        ("LL INT2", ll.get("kv_int2"), dict(color="tab:blue")),
                        ("SF INT4", sf.get("kv_int4"), dict(color="tab:red", linestyle="--")),
                        ("LL INT4", ll.get("kv_int4"), dict(color="tab:blue", linestyle="--"))):
        if not d:
            continue
        floor = sf_floor if cfg.startswith("SF") else ll_floor
        cs = [c for c in sorted(d) if floor.get(c)]
        ax.plot(cs, [d[c] / floor[c] for c in cs], marker="o", markersize=3,
                linewidth=1.5, label=cfg, **sty)
    ax.axhline(1.0, color="gray", linewidth=1, alpha=0.7)
    ax.set_xlabel("chunk index")
    ax.set_ylabel("rel_err / own chaos floor")
    ax.set_title("damage above each model's own floor", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png}")

    print("\n--- chunk 20 요약 ---")
    for model, data, floor in (("Self-Forcing", sf, sf_floor), ("LongLive", ll, ll_floor)):
        for cfg, d in sorted(data.items()):
            if 20 in d and floor.get(20):
                print(f"{model:13s} {cfg:16s} rel_err={d[20]:.4f}  "
                      f"floor={floor[20]:.4f}  ratio={d[20]/floor[20]:.2f}")


if __name__ == "__main__":
    main()
