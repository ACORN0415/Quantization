"""Measure the zero-point magnitudes the fixed asymmetric quantizer produces on
real post-RoPE keys, to see how much headroom fp16 storage (max 65504) has.
"""
import sys, torch
from omegaconf import OmegaConf
sys.path.insert(0, ".")
sys.path.insert(0, "/home/devel/gpu/kv-quant-longhorizon")

import wan.modules.causal_model as cm
from kv_quant.utils import _reshape_blocks
from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset

torch.set_grad_enabled(False)
device = torch.device("cuda")
config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                         OmegaConf.load("configs/self_forcing_dmd_long.yaml"))
pipe = CausalInferencePipeline(config, device=device)
sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
pipe.generator.load_state_dict(sd["generator_ema"]); del sd
pipe = pipe.to(dtype=torch.bfloat16); pipe.generator.to(device)

ds = TextDataset(prompt_path="prompts/quant3/prompts3.txt")
# pipeline.inference() encodes internally, so the encoder has to stay on device.
pipe.text_encoder.to(device)
pipe.vae.to(device)

cm.KV_NEW_CAPTURE = []
g = torch.Generator(device=device); g.manual_seed(0)
noise = torch.randn([1, 12, 16, 60, 104], generator=g, device=device, dtype=torch.bfloat16)
try:
    pipe.inference(noise=noise, text_prompts=[ds[0]["prompts"]],
                   return_latents=True, base_seed=0)
except torch.OutOfMemoryError:
    print("(VAE decode OOM ignored)")
caps = cm.KV_NEW_CAPTURE; cm.KV_NEW_CAPTURE = None
print(f"captured {len(caps)} (layer, call) key tensors\n")

qmin, qmax = 0, 15
allz = []
for k, _ in caps:
    xb, _ = _reshape_blocks(k.float(), 16)
    mn = xb.amin(dim=(2,), keepdim=True); mx = xb.amax(dim=(2,), keepdim=True)
    scale = ((mx - mn) / (qmax - qmin)).clamp_min(1e-8)
    allz.append(torch.round(qmin - mn / scale).abs().flatten().cpu())
    del xb, mn, mx, scale
z = torch.cat(allz)
print(f"|zp| over {z.numel():,} groups (post-RoPE keys, INT4, block 16):")
for q in [0.5, 0.9, 0.99, 0.999, 0.9999]:
    idx = torch.randperm(z.numel())[:2_000_000]
    print(f"  p{q*100:7.2f} : {torch.quantile(z[idx].float(), q).item():10.1f}")
print(f"  max      : {z.max().item():10.1f}")
print(f"\nfp16 max = 65504  ->  headroom = {65504 / max(z.max().item(), 1):.0f}x")
print(f"groups exceeding fp16 max: {(z > 65504).sum().item()}")
