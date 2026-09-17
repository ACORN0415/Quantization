"""Measure FP8 activation quantization error: per-tensor vs per-token scaling.

Answers whether the per-tensor FP8 scale used by fakequant.fq_fp8_e4m3 is
needlessly lossy on this model's activations (i.e. whether they carry the
outlier channels that transformers are known for).
"""
import sys
import torch
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from pipeline import CausalInferencePipeline

torch.set_grad_enabled(False)

def fp8_per_tensor(x):
    s = x.abs().amax().clamp(min=1e-8) / 448.0
    return (x / s).to(torch.float8_e4m3fn).to(x.dtype) * s

def fp8_per_token(x):
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    return (x / s).to(torch.float8_e4m3fn).to(x.dtype) * s

config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                         OmegaConf.load("configs/self_forcing_dmd_long.yaml"))
device = torch.device("cuda")
pipeline = CausalInferencePipeline(config, device=device)
state = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
pipeline.generator.load_state_dict(state["generator_ema"])
pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.text_encoder.to(device); pipeline.generator.to(device); pipeline.vae.to(device)

records = {}
def make_hook(name):
    def hook(mod, inputs):
        x = inputs[0]
        if name in records or x.numel() == 0:
            return
        xf = x.float()
        records[name] = {
            "shape": tuple(x.shape),
            "pt": ((fp8_per_tensor(xf) - xf).norm() / xf.norm()).item(),
            "ptok": ((fp8_per_token(xf) - xf).norm() / xf.norm()).item(),
            # outlier signal: how peaked is the largest channel vs the median row max
            "amax": xf.abs().amax().item(),
            "row_amax_med": xf.abs().amax(dim=-1).median().item(),
        }
    return hook

targets = ("blocks.0.self_attn.q", "blocks.0.ffn.0", "blocks.0.ffn.2",
           "blocks.15.self_attn.q", "blocks.15.ffn.0", "blocks.15.ffn.2",
           "blocks.29.self_attn.q", "blocks.29.ffn.0", "blocks.29.ffn.2")
handles = []
for n, m in pipeline.generator.model.named_modules():
    if isinstance(m, torch.nn.Linear) and any(n.endswith(t) for t in targets):
        handles.append(m.register_forward_pre_hook(make_hook(n)))
print(f"hooked {len(handles)} layers")

prompt = open("prompts/quant3/prompts3.txt").readline().strip()
g = torch.Generator(device=device); g.manual_seed(0)
noise = torch.randn([1, 6, 16, 60, 104], generator=g, device=device, dtype=torch.bfloat16)
try:
    pipeline.inference(noise=noise, text_prompts=[prompt], return_latents=True, base_seed=0)
except torch.OutOfMemoryError:
    # The hooks already fired during the diffusion loop; only VAE decode ran out.
    print("(VAE decode OOM ignored - activation records already collected)")

print(f"\n{'layer':32s} {'shape':>22s} {'fp8_tensor':>10s} {'fp8_token':>10s} {'amax/med':>9s}")
for n in sorted(records):
    r = records[n]
    ratio = r["amax"] / max(r["row_amax_med"], 1e-8)
    print(f"{n[:32]:32s} {str(r['shape']):>22s} {r['pt']:10.4f} {r['ptok']:10.4f} {ratio:9.1f}")
