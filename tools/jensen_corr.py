"""Gate F1 — per-slot key noise against the attention mass that slot gains.

Jensen: E[exp(s+eps)] = exp(s)exp(sigma^2/2), so a slot whose keys carry more
quantization noise should attract mass the BF16 run did not give it. Session 4's
mass measurement shows the oldest slot gaining the most, which a uniform bias
cannot explain — under a uniform shift the slot with the largest existing mass
would gain most in absolute terms. So the bias must differ by slot, and this
measures whether it does.

Two quantities per slot, at a chosen chunk:
  sigma2   the squared key perturbation actually applied there
           (dequantized cache minus the BF16 reference, per element)
  delta2   the quantizer's own step size, which is what the TUM term uses
and correlates each against delta-mass from the attention measurement.
"""
import argparse
import json
import os
import statistics
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset


def build_cache(size, device, nblk, quantizer=None):
    return [{
        "k": torch.zeros([1, size, 12, 128], dtype=torch.bfloat16, device=device),
        "v": torch.zeros([1, size, 12, 128], dtype=torch.bfloat16, device=device),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "quantizer": quantizer, "quant_state": None,
    } for _ in range(nblk)]


def build_xattn(device, nblk):
    return [{"k": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
             "v": torch.zeros([1, 512, 12, 128], dtype=torch.bfloat16, device=device),
             "is_init": False} for _ in range(nblk)]


def run_chunk(pipe, kv, xattn, cond, noise, c, start, nfb, fsl, seed):
    g = torch.Generator(device=noise.device)
    g.manual_seed(seed * 1000003 + 7919 + c)
    x = noise[:, start:start + nfb]
    pred = None
    for i, t in enumerate(pipe.denoising_step_list):
        ts = torch.ones([1, nfb], device=noise.device, dtype=torch.int64) * t
        _, pred = pipe.generator(noisy_image_or_video=x, conditional_dict=cond,
                                 timestep=ts, kv_cache=kv, crossattn_cache=xattn,
                                 current_start=start * fsl)
        if i < len(pipe.denoising_step_list) - 1:
            flat = pred.flatten(0, 1)
            rn = torch.randn(flat.shape, generator=g, device=noise.device, dtype=flat.dtype)
            x = pipe.scheduler.add_noise(
                flat, rn, pipe.denoising_step_list[i + 1] *
                torch.ones([nfb], device=noise.device, dtype=torch.long)
            ).unflatten(0, pred.shape[:2])
    ctx = torch.ones([1, nfb], device=noise.device, dtype=torch.int64) * pipe.args.context_noise
    pipe.generator(noisy_image_or_video=pred, conditional_dict=cond, timestep=ctx,
                   kv_cache=kv, crossattn_cache=xattn, current_start=start * fsl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="configs/gateA_A1.yaml")
    ap.add_argument("--checkpoint_path", default="checkpoints/self_forcing_dmd.pt")
    ap.add_argument("--data_path", default="prompts/quant3/prompts10.txt")
    ap.add_argument("--at_chunk", type=int, default=20)
    ap.add_argument("--max_prompts", type=int, default=3)
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 3, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gateF/jensen_sigma.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load(args.config_path))
    pipe = CausalInferencePipeline(config, device=device)
    st = torch.load(args.checkpoint_path, map_location="cpu")
    pipe.generator.load_state_dict(st["generator_ema"]); del st
    pipe = pipe.to(dtype=torch.bfloat16); pipe.generator.to(device)

    ds = TextDataset(prompt_path=args.data_path)
    prompts = [ds[i]["prompts"] for i in range(min(len(ds), args.max_prompts))]
    pipe.text_encoder.to(device)
    conds = [pipe.text_encoder(text_prompts=[p]) for p in prompts]
    pipe.text_encoder.to("cpu"); torch.cuda.empty_cache()

    sys.path.insert(0, os.path.expanduser("~/gpu/kv-quant-longhorizon"))
    from kv_quant.factory import create_quantizer

    fsl, nfb = pipe.frame_seq_length, pipe.num_frame_per_block
    las = pipe.local_attn_size
    cache_size = las * fsl if las != -1 else 32760
    chunk_tokens = nfb * fsl
    nblk = pipe.num_transformer_blocks
    n_slots = cache_size // chunk_tokens

    out = {"at_chunk": args.at_chunk, "n_slots": n_slots, "by_bits": {}}

    for bits in args.bits:
        qz = create_quantizer("RTN", bits=bits, block_size=16,
                              key_bits=bits, value_bits=bits, name=f"RTN_INT{bits}")
        acc = {i: [] for i in range(n_slots)}
        for pi, (prompt, cond) in enumerate(zip(prompts, conds)):
            noise = torch.cat([
                torch.randn([1, nfb, 16, 60, 104],
                            generator=torch.Generator(device=device).manual_seed(args.seed * 1000003 + c),
                            device=device, dtype=torch.bfloat16)
                for c in range(21)], dim=1)
            ref_kv, ref_x = build_cache(cache_size, device, nblk), build_xattn(device, nblk)
            q_kv, q_x = build_cache(cache_size, device, nblk, qz), build_xattn(device, nblk)
            start = 0
            for c in range(args.at_chunk + 1):
                run_chunk(pipe, ref_kv, ref_x, cond, noise, c, start, nfb, fsl, args.seed)
                run_chunk(pipe, q_kv, q_x, cond, noise, c, start, nfb, fsl, args.seed)
                start += nfb
            n = int(ref_kv[0]["local_end_index"].item())
            for slot in range(n_slots):
                lo, hi = slot * chunk_tokens, min((slot + 1) * chunk_tokens, n)
                if hi <= lo:
                    continue
                s2 = []
                for L in range(nblk):
                    rk = ref_kv[L]["k"][:, lo:hi].float()
                    dk, _ = qz.dequantize_kv(q_kv[L]["quant_state"],
                                             meta={"tensor_dtype": torch.bfloat16})
                    s2.append(((dk[:, lo:hi].to(rk.device).float() - rk) ** 2).mean().item())
                acc[slot].append(statistics.mean(s2))
            del noise, ref_kv, ref_x, q_kv, q_x
            torch.cuda.empty_cache()
            print(f"  INT{bits} prompt {pi} done")
        out["by_bits"][str(bits)] = {str(s): statistics.mean(v) for s, v in acc.items() if v}
        print(f"INT{bits}: " + "  ".join(f"slot{s}={v:.4f}"
                                         for s, v in out["by_bits"][str(bits)].items()))

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
