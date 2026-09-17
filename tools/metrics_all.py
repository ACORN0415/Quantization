"""Gate C — triangulate the metrics.

Session 2 showed that latent L2 and MUSIQ each fail in the opposite direction:
L2 saturates at the chaotic-divergence floor and cannot see quality loss, while
MUSIQ has no reference and rewards a sharp but entirely different scene. Neither
sees temporal defects at all, since MUSIQ reads one frame at a time.

This computes five per-chunk metrics over the same runs so a claim can be
checked against all of them:

  rel_err   latent L2 against the BF16 run           (reference, saturates)
  musiq     no-reference perceptual quality          (single frame)
  flow_err  optical-flow warping error between       (temporal)
            adjacent frames, via RAFT
  lpips     against the BF16 frame                   (reference, perceptual)
  clip      frame-to-prompt alignment                (did it leave the prompt?)

Everything runs as post-processing on videos that already exist.
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

import torch


def chunk_frame_index(chunk_idx, frames_per_chunk, num_pixel_frames):
    """Middle pixel frame of a chunk.

    The VAE maps latent frame 0 to pixel frame 0 and each later latent frame to
    four pixel frames.
    """
    lat_mid = chunk_idx * frames_per_chunk + frames_per_chunk / 2.0
    pix = 0 if lat_mid < 1 else int(round((lat_mid - 1) * 4 + 1))
    return min(max(pix, 0), num_pixel_frames - 1)


def load_video(path):
    from torchvision.io import read_video
    v, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")
    return v


def flow_warp_error(frames, device, raft, pairs_per_chunk=2):
    """Mean warping error between adjacent frames.

    Warps frame t+1 back to t with the estimated flow and measures the residual.
    A model that flickers or breaks motion continuity scores higher.
    """
    import torch.nn.functional as F
    errs = []
    for i in range(min(pairs_per_chunk, frames.shape[0] - 1)):
        a = frames[i:i + 1].to(device).float() / 127.5 - 1.0
        b = frames[i + 1:i + 2].to(device).float() / 127.5 - 1.0
        with torch.no_grad():
            flow = raft(a, b)[-1]
        _, _, h, w = a.shape
        gy, gx = torch.meshgrid(torch.arange(h, device=device),
                                torch.arange(w, device=device), indexing="ij")
        grid = torch.stack((gx, gy), dim=0).float()[None] + flow
        grid[:, 0] = grid[:, 0] / max(w - 1, 1) * 2 - 1
        grid[:, 1] = grid[:, 1] / max(h - 1, 1) * 2 - 1
        warped = F.grid_sample(b, grid.permute(0, 2, 3, 1), align_corners=True)
        errs.append((warped - a).abs().mean().item())
    return statistics.mean(errs) if errs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="tag=video_dir pairs")
    ap.add_argument("--ref", default=None,
                    help="tag of the BF16 reference run (for lpips)")
    ap.add_argument("--prompts", default="prompts/quant3/prompts10.txt")
    ap.add_argument("--frames_per_chunk", type=int, default=3)
    ap.add_argument("--num_latent_frames", type=int, default=63)
    ap.add_argument("--out_csv", default="results/metrics_all.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="metrics to skip: musiq lpips clip flow")
    args = ap.parse_args()

    device = torch.device(args.device)
    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    n_chunks = args.num_latent_frames // args.frames_per_chunk

    import pyiqa
    metrics = {}
    if "musiq" not in args.skip:
        metrics["musiq"] = pyiqa.create_metric("musiq", device=device)
    if "lpips" not in args.skip and args.ref:
        metrics["lpips"] = pyiqa.create_metric("lpips", device=device)
    if "clip" not in args.skip:
        metrics["clip"] = pyiqa.create_metric("clipscore", device=device)
    raft = None
    if "flow" not in args.skip:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()

    runs = dict(spec.split("=", 1) for spec in args.runs)

    # Reference frames, kept on CPU: one per (prompt, chunk).
    ref_frames = {}
    if args.ref and args.ref in runs:
        rdir = runs[args.ref]
        for pi, vf in enumerate(sorted(f for f in os.listdir(rdir) if f.endswith(".mp4"))):
            v = load_video(os.path.join(rdir, vf))
            for c in range(n_chunks):
                ref_frames[(pi, c)] = v[chunk_frame_index(c, args.frames_per_chunk, v.shape[0])].clone()
            del v
        print(f"reference frames cached: {len(ref_frames)}")

    rows = []
    for tag, vdir in runs.items():
        if not os.path.isdir(vdir):
            print(f"  ! {vdir} missing, skipped")
            continue
        vids = sorted(f for f in os.listdir(vdir) if f.endswith(".mp4"))
        if not vids:
            print(f"  ! no videos in {vdir}")
            continue
        per = defaultdict(lambda: defaultdict(list))

        for pi, vf in enumerate(vids):
            v = load_video(os.path.join(vdir, vf))
            n_pix = v.shape[0]
            prompt = prompts[pi] if pi < len(prompts) else ""
            for c in range(n_chunks):
                fi = chunk_frame_index(c, args.frames_per_chunk, n_pix)
                frame = v[fi].unsqueeze(0).to(device).float() / 255.0
                with torch.no_grad():
                    if "musiq" in metrics:
                        per[c]["musiq"].append(metrics["musiq"](frame).item())
                    if "clip" in metrics and prompt:
                        per[c]["clip"].append(
                            metrics["clip"](frame, caption_list=[prompt]).item())
                    if "lpips" in metrics and (pi, c) in ref_frames:
                        rf = ref_frames[(pi, c)].unsqueeze(0).to(device).float() / 255.0
                        per[c]["lpips"].append(metrics["lpips"](frame, rf).item())
                if raft is not None:
                    lo = max(0, fi - 1)
                    per[c]["flow_err"].append(
                        flow_warp_error(v[lo:lo + 3], device, raft))
                del frame
            del v
            torch.cuda.empty_cache()

        for c in sorted(per):
            row = {"config": tag, "chunk_idx": c,
                   "n_prompts": len(per[c].get("musiq") or per[c].get("clip") or [])}
            for name, vals in per[c].items():
                vals = [x for x in vals if x == x]  # drop NaN
                if not vals:
                    continue
                row[f"{name}_mean"] = statistics.mean(vals)
                row[f"{name}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                # 95% CI half-width of the mean
                if len(vals) > 1:
                    row[f"{name}_ci95"] = 1.96 * statistics.stdev(vals) / len(vals) ** 0.5
            rows.append(row)
        print(f"{tag}: {len(per)} chunks over {len(vids)} prompts")

    if not rows:
        raise SystemExit("nothing measured")

    fields = sorted({k for r in rows for k in r},
                    key=lambda k: (k not in ("config", "chunk_idx", "n_prompts"), k))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} rows, {len(fields)} columns)")


if __name__ == "__main__":
    main()
