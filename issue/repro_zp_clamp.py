"""Minimal reproduction: zero-point clamping in quantize_asym.

Run from the repo root:  python repro_zp_clamp.py
"""
import torch
from kv_quant.utils import quantize_asym, dequantize_asym, quantize_sym, dequantize_sym

torch.manual_seed(0)

def rel(a, b):
    return ((a - b).norm() / b.norm()).item()

# Block layout produced by _reshape_blocks: [B, n_blocks, block_size, heads, dim]
x = torch.randn(1, 8, 16, 12, 128) * 0.1 + 5.0     # a group that never crosses zero

q, s, z = quantize_asym(x, bits=4, reduce_dims=(2,))
asym = dequantize_asym(q, s, z, dtype=torch.float32)

q2, s2 = quantize_sym(x, bits=4, reduce_dims=(2,))
sym = dequantize_sym(q2, s2, dtype=torch.float32)

print(f"asymmetric (KIVI/QAQ path): rel_err = {rel(asym, x):.4f}")
print(f"symmetric  (RTN path)     : rel_err = {rel(sym,  x):.4f}")

# why
qmin, qmax = 0, 15
x_min = x.amin(dim=(2,), keepdim=True)
x_max = x.amax(dim=(2,), keepdim=True)
scale = ((x_max - x_min) / (qmax - qmin)).clamp_min(1e-8)
zp_true = torch.round(qmin - x_min / scale)
zp_used = zp_true.clamp(qmin, qmax)
print(f"\ntrue zero point range   : [{zp_true.min():.0f}, {zp_true.max():.0f}]")
print(f"after .clamp(0, 15)     : [{zp_used.min():.0f}, {zp_used.max():.0f}]")
print(f"groups altered by clamp : {(zp_true != zp_used).float().mean() * 100:.1f}%")
