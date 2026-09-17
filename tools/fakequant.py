"""Fake (simulate-only) quantization hooks for the Self Forcing DiT blocks.

No low-bit kernels: values are quantized and immediately de-quantized in BF16
so the arithmetic error is injected while the compute stays BF16. Used to
measure how weight/activation quantization error propagates along the chunk
axis through the KV cache.

Scope: Linear layers inside the DiT blocks only. The text encoder and the VAE
are left untouched.
"""
from typing import List, Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# quantizers
# --------------------------------------------------------------------------
def fq_int(x: torch.Tensor, bits: int, group: Optional[int] = 128) -> torch.Tensor:
    """Per-group symmetric integer fake-quantization along the last dim."""
    if bits >= 16:
        return x
    shape = x.shape
    if group and shape[-1] % group == 0:
        xq = x.reshape(-1, group)
    else:
        xq = x.reshape(-1, shape[-1])
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    s = xq.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    out = (torch.round(xq / s).clamp(qmin, qmax) * s)
    return out.reshape(shape).to(x.dtype)


def fq_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 round-trip with a per-tensor scale into the representable range."""
    amax = x.abs().amax().clamp(min=1e-8)
    # E4M3 max representable magnitude
    scale = amax / 448.0
    return ((x / scale).to(torch.float8_e4m3fn).to(x.dtype) * scale)


def quantize(x: torch.Tensor, spec: str, group: Optional[int] = 128) -> torch.Tensor:
    """spec is one of: 'bf16' (no-op), 'fp8', 'int8', 'int4', 'int2', ..."""
    if spec in (None, "bf16", "none", "16"):
        return x
    if spec == "fp8":
        return fq_fp8_e4m3(x)
    if spec.startswith("int"):
        return fq_int(x, int(spec[3:]), group=group)
    raise ValueError(f"unknown quant spec: {spec}")


# --------------------------------------------------------------------------
# hook installation
# --------------------------------------------------------------------------
# Linear layers inside a Wan DiT block. self_attn/cross_attn expose q,k,v,o;
# the feed-forward is an nn.Sequential, so its Linears end in .0 / .2.
DEFAULT_PATTERNS = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
                    "ffn.")

KV_PATTERNS = ("self_attn.k", "self_attn.v")


class FakeQuantizer:
    """Installs pre-forward hooks on selected nn.Linear modules.

    Weights are quantized once, in place. Activations are quantized on every
    forward.
    """

    def __init__(self, model: nn.Module, w_spec: str = "bf16", a_spec: str = "bf16",
                 group: Optional[int] = 128,
                 include: Optional[tuple] = None, exclude: Optional[tuple] = None):
        self.model = model
        self.w_spec = w_spec
        self.a_spec = a_spec
        self.group = group
        self.include = DEFAULT_PATTERNS if include is None else include
        self.exclude = exclude or ()
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.targets: List[str] = []

    def _match(self, name: str) -> bool:
        if not any(p in name for p in self.include):
            return False
        if any(p in name for p in self.exclude):
            return False
        return True

    def install(self) -> "FakeQuantizer":
        for name, mod in self.model.named_modules():
            if not isinstance(mod, nn.Linear) or not self._match(name):
                continue
            self.targets.append(name)

            if self.w_spec not in (None, "bf16"):
                with torch.no_grad():
                    mod.weight.copy_(quantize(mod.weight.data, self.w_spec, self.group))

            if self.a_spec not in (None, "bf16"):
                a_spec, group = self.a_spec, self.group

                def pre_hook(module, inputs, a_spec=a_spec, group=group):
                    x = inputs[0]
                    return (quantize(x, a_spec, group),) + inputs[1:]

                self.handles.append(mod.register_forward_pre_hook(pre_hook))

        print(f"[fakequant] W={self.w_spec} A={self.a_spec} group={self.group} "
              f"-> {len(self.targets)} Linear layers, {len(self.handles)} activation hooks")
        return self

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# Named configurations from the session plan (§5).
CONFIGS = {
    "bf16":        dict(w_spec="bf16", a_spec="bf16"),
    "w8a8":        dict(w_spec="fp8",  a_spec="fp8"),
    "w4a16":       dict(w_spec="int4", a_spec="bf16"),
    "w4a8":        dict(w_spec="int4", a_spec="fp8"),
    "w4a4":        dict(w_spec="int4", a_spec="int4"),
    # Added beyond the session plan: every specified config drives the latent
    # metric past the divergence ceiling within ~8 chunks. INT8 weights perturb
    # ~20x less, so the early chunks stay interpretable.
    "w8a16":       dict(w_spec="int8", a_spec="bf16"),
    # ablation: is the damage the error that enters the KV cache, or everything else?
    "kv_only_w4a4": dict(w_spec="int4", a_spec="int4", include=KV_PATTERNS),
    "no_kv_w4a4":   dict(w_spec="int4", a_spec="int4", exclude=KV_PATTERNS),
}


def apply_config(model: nn.Module, name: str, group: Optional[int] = 128) -> Optional[FakeQuantizer]:
    if name in (None, "bf16"):
        print("[fakequant] bf16 baseline, no hooks installed")
        return None
    if name not in CONFIGS:
        raise ValueError(f"unknown config {name}; choose from {sorted(CONFIGS)}")
    return FakeQuantizer(model, group=group, **CONFIGS[name]).install()
