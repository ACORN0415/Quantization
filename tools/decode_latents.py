"""Decode saved per-chunk latents into videos.

The sweep wrote latents per configuration but, until the `--output_folder`
override was wired up, all runs shared one video directory. Rather than
regenerating (4 min per config), this reassembles the dumped chunks and runs
only the VAE, which takes seconds.
"""
import argparse
import os

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import CausalInferencePipeline


def load_full_latent(prompt_dir):
    chunks = []
    for fn in sorted(os.listdir(prompt_dir)):
        if not fn.startswith("chunk_"):
            continue
        chunks.append(torch.load(os.path.join(prompt_dir, fn),
                                 map_location="cpu")["latent"])
    if not chunks:
        return None
    return torch.cat(chunks, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="configs/gateA_A1.yaml")
    ap.add_argument("--latent_roots", nargs="+", required=True,
                    help="tag=dir pairs, e.g. kv_int4=results/latents_ll/kv_int4")
    ap.add_argument("--out_root", default="results")
    ap.add_argument("--fps", type=int, default=16)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load(args.config_path))

    # Only the VAE is needed; build the pipeline and drop the rest.
    pipeline = CausalInferencePipeline(config, device=device)
    pipeline.text_encoder.to("cpu")
    del pipeline.generator
    torch.cuda.empty_cache()
    pipeline.vae.to(device=device, dtype=torch.bfloat16)

    for spec in args.latent_roots:
        tag, root = spec.split("=", 1)
        if not os.path.isdir(root):
            print(f"  ! {root} missing, skipped")
            continue
        out_dir = os.path.join(args.out_root, tag, "videos")
        os.makedirs(out_dir, exist_ok=True)
        for pi, p in enumerate(sorted(os.listdir(root))):
            pdir = os.path.join(root, p)
            if not os.path.isdir(pdir):
                continue
            lat = load_full_latent(pdir)
            if lat is None:
                continue
            lat = lat.to(device=device, dtype=torch.bfloat16)
            video = pipeline.vae.decode_to_pixel(lat, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            frames = (255.0 * rearrange(video, "b t c h w -> b t h w c")).cpu()
            write_video(os.path.join(out_dir, f"{pi}-0_decoded.mp4"),
                        frames[0], fps=args.fps)
            pipeline.vae.model.clear_cache()
            del lat, video, frames
            torch.cuda.empty_cache()
        print(f"{tag}: wrote videos to {out_dir}")


if __name__ == "__main__":
    main()
