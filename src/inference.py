# Modified from Self-Forcing 33593df; see vendor/UPSTREAM.md
import argparse
import json
import time
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
from torch.utils.data import DataLoader, SequentialSampler

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--rng", choices=["release", "chunk"], default="release")
parser.add_argument("--drain_rng", action="store_true")
parser.add_argument("--kv_method", choices=["rtn", "kivi"])
parser.add_argument("--kv_bits", type=int, default=4)
parser.add_argument("--key_bits", type=int)
parser.add_argument("--value_bits", type=int)
parser.add_argument("--group", type=int, default=16)
parser.add_argument("--storage", choices=["rolling", "chunk"], default="rolling")
parser.add_argument("--residual_tokens", type=int, default=0)
parser.add_argument("--unsigned_codes", action="store_true", help="Use uint8 KIVI codes instead of historical int8")
parser.add_argument("--latent_dump_root")
parser.add_argument("--kv_error_log")
parser.add_argument("--stats_path")
parser.add_argument("--no_video", action="store_true", help="Decode normally but omit mp4 encoding")
parser.add_argument("--max_prompts", type=int)
args = parser.parse_args()

device = torch.device("cuda")
local_rank = 0
set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

pipeline = CausalInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

if args.kv_method:
    from kvq import Quantizer, RollingStore, ChunkStore
    quantizer = Quantizer(args.kv_method, args.kv_bits if args.key_bits is None else args.key_bits,
                          args.kv_bits if args.value_bits is None else args.value_bits, args.group,
                          legacy_unsigned_int8=not args.unsigned_codes)
    store_cls = RollingStore if args.storage == "rolling" else ChunkStore
    sink = int(getattr(config.model_kwargs, "sink_size", 0)) * pipeline.frame_seq_length
    pipeline.store_factory = lambda capacity: store_cls(quantizer, capacity, sink, args.residual_tokens)
    pipeline.kv_records = [] if args.kv_error_log else None

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

run_stats = []
for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    if args.max_prompts is not None and i >= args.max_prompts:
        break
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    prompts = [extended_prompt if extended_prompt is not None else prompt] * args.num_samples
    initial_latent = None
    if args.rng == "release":
        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    if args.rng == "chunk":
        n = pipeline.num_frame_per_block
        if pipeline.independent_first_frame or args.num_output_frames % n:
            raise ValueError("chunk RNG requires fixed-size chunks")
        sampled_noise = torch.cat([
            torch.randn([args.num_samples, n, 16, 60, 104],
                        generator=torch.Generator(device=device).manual_seed(args.seed * 1000003 + c),
                        device=device, dtype=torch.bfloat16)
            for c in range(args.num_output_frames // n)], dim=1)

    # Generate 81 frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
        base_seed=args.seed if args.rng == "chunk" else None,
        latent_dump_dir=(os.path.join(args.latent_dump_root, f"prompt{idx:03d}")
                         if args.latent_dump_root else None),
        drain_rng=args.drain_rng,
        prompt_index=idx,
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            if not args.no_video:
                write_video(output_path, video[seed_idx], fps=16)

    torch.cuda.synchronize()
    cache_memory = None
    if args.kv_method:
        from kvq.memory import memory_report
        cache_memory = memory_report([
            {"state": getattr(c["store"], "state", None),
             "residual": getattr(c["store"], "residual", None),
             "segments": getattr(c["store"], "segments", None),
             "pending": getattr(c["store"], "pending", None)}
            for c in pipeline.kv_cache1])
    run_stats.append({"prompt": idx, "seconds": time.time() - started,
                      "cache_memory": cache_memory,
                      "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3})

for path, payload in ((args.kv_error_log, getattr(pipeline, "kv_records", [])),
                      (args.stats_path, {"args": vars(args), "runs": run_stats})):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
