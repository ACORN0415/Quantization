"""Step J — a reference-free quality curve alongside the latent L2 curve.

Latent L2 against BF16 saturates at the chaotic-divergence floor and says
nothing about quality above it: session 1 had W4A16 at rel_err 1.24 with a
perfectly good video and RTN INT2 at 1.85 with a broken one. MUSIQ needs no
reference, so it separates "went somewhere else" from "fell apart".

Runs as post-processing on the videos already written — nothing is regenerated.
"""
import argparse
import csv
import json
import os
import statistics

import torch


def chunk_frame_indices(num_latent_frames, frames_per_chunk, num_pixel_frames):
    """Middle pixel frame of each chunk.

    The VAE maps latent frame 0 to pixel frame 0 and every later latent frame to
    4 pixel frames, so chunk c covers latent [c*k, (c+1)*k).
    """
    out = []
    n_chunks = num_latent_frames // frames_per_chunk
    for c in range(n_chunks):
        lat_mid = c * frames_per_chunk + frames_per_chunk / 2.0
        pix = 0 if lat_mid < 1 else int(round((lat_mid - 1) * 4 + 1))
        out.append((c, min(max(pix, 0), num_pixel_frames - 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", default="results",
                    help="Directory holding <config>/videos/<idx>-0_ema.mp4")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--metric", default="musiq", help="pyiqa metric name")
    ap.add_argument("--frames_per_chunk", type=int, default=3)
    ap.add_argument("--num_latent_frames", type=int, default=63)
    ap.add_argument("--out_csv", default="results/iq_curves.csv")
    ap.add_argument("--out_png", default="results/iq_curves.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import pyiqa
    from torchvision.io import read_video

    device = torch.device(args.device)
    metric = pyiqa.create_metric(args.metric, device=device)
    print(f"metric={args.metric} lower_better={metric.lower_better}")

    rows, series = [], {}
    for cfg in args.configs:
        vdir = os.path.join(args.video_root, cfg, "videos")
        if not os.path.isdir(vdir):
            print(f"  ! {vdir} missing, skipped")
            continue
        vids = sorted(f for f in os.listdir(vdir) if f.endswith(".mp4"))
        if not vids:
            print(f"  ! no videos in {vdir}, skipped")
            continue

        per_chunk = {}
        for vf in vids:
            video, _, _ = read_video(os.path.join(vdir, vf), pts_unit="sec",
                                     output_format="TCHW")
            n_pix = video.shape[0]
            for c, fi in chunk_frame_indices(args.num_latent_frames,
                                             args.frames_per_chunk, n_pix):
                frame = video[fi].unsqueeze(0).float().to(device) / 255.0
                with torch.no_grad():
                    score = metric(frame).item()
                per_chunk.setdefault(c, []).append(score)
            del video

        series[cfg] = {c: statistics.mean(v) for c, v in per_chunk.items()}
        for c in sorted(per_chunk):
            rows.append({
                "config": cfg,
                "chunk_idx": c,
                "iq_mean": statistics.mean(per_chunk[c]),
                "iq_std": statistics.pstdev(per_chunk[c]),
                "n_videos": len(per_chunk[c]),
            })
        vals = list(series[cfg].values())
        print(f"{cfg}: {len(vals)} chunks, mean {statistics.mean(vals):.2f}, "
              f"first {vals[0]:.2f} -> last {vals[-1]:.2f}")

    if not rows:
        raise SystemExit("nothing measured")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for cfg, d in sorted(series.items()):
        cs = sorted(d)
        style = dict(linestyle="--", color="black", linewidth=1.8) if cfg.startswith("ctrl") \
            else dict(marker="o", markersize=3.5, linewidth=1.4)
        ax.plot(cs, [d[c] for c in cs], label=cfg, **style)
    ax.axvline(7, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(7, ax.get_ylim()[1] * 0.99, " eviction (chunk 7)", fontsize=8, va="top", alpha=0.75)
    ax.set_xlabel("chunk index")
    ax.set_ylabel(f"{args.metric} (higher = better)")
    ax.set_title("Reference-free image quality along the frame axis")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
