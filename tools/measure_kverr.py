"""Plot the in-situ KV cache quantization error per chunk.

Unlike the latent L2 curve, this is immune to trajectory divergence: it compares
the cache contents right before storing against what comes back out, so it
isolates what the cache itself loses.
"""
import argparse, csv, json, os, statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="dirs containing kv_err.json")
ap.add_argument("--out_csv", default="results/kv_cache_error.csv")
ap.add_argument("--out_png", default="results/kv_cache_error.png")
ap.add_argument("--local_attn_size", type=int, default=21)
ap.add_argument("--num_frame_per_block", type=int, default=3)
args = ap.parse_args()

series, rows = {}, []
for d in args.runs:
    path = os.path.join(d, "kv_err.json")
    if not os.path.exists(path):
        print(f"  ! {path} missing, skipped"); continue
    blob = json.load(open(path))
    label = f"{blob['kv_quant']}_INT{blob['kv_bits']}"
    agg = defaultdict(list)
    for r in blob["by_chunk"]:
        agg[r["chunk_idx"]].append((r["k_rel_mean"], r["v_rel_mean"]))
    ks = {c: statistics.mean(x[0] for x in v) for c, v in agg.items()}
    vs = {c: statistics.mean(x[1] for x in v) for c, v in agg.items()}
    series[label] = (ks, vs)
    for c in sorted(ks):
        rows.append({"config": label, "chunk_idx": c,
                     "k_rel_mean": ks[c], "v_rel_mean": vs[c],
                     "n_prompts": len(agg[c])})

os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
with open(args.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {args.out_csv} ({len(rows)} rows)")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (axk, axv) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for label, (ks, vs) in sorted(series.items()):
    cs = sorted(ks)
    axk.plot(cs, [ks[c] for c in cs], marker="o", markersize=3.5, linewidth=1.5, label=label)
    axv.plot(cs, [vs[c] for c in cs], marker="o", markersize=3.5, linewidth=1.5, label=label)
boundary = args.local_attn_size / args.num_frame_per_block
for ax, name in ((axk, "keys"), (axv, "values")):
    ax.axvline(boundary, color="k", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("chunk index"); ax.set_title(f"KV cache quantization error - {name}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
axk.set_ylabel(r"relative $L_2$ error (before store vs after reload)")
axk.text(boundary, axk.get_ylim()[1] * 0.97, f" eviction starts (chunk {boundary:g})",
         fontsize=8, va="top", alpha=0.75)
fig.tight_layout(); fig.savefig(args.out_png, dpi=150)
print(f"wrote {args.out_png}")
