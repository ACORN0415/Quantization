"""Independent grouping candidate, validated on CPU only; see its study note.

No packing, cache lifecycle, attention kernel, or official implementation is
included. This module is deliberately unregistered in the frozen experiment.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kvq.quant import Quantized


SOURCE_COMMIT = "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6"
# Smallest positive FP16 subnormal: the persisted scale must not become zero.
MIN_SCALE = 2.0 ** -24


@dataclass
class GroupedQuantized(Quantized):
    value: bool = False
    arithmetic: str = "fp32"

    def restore(self):
        # Reconstruct from persisted metadata, with one final BF16 conversion.
        dtype = self.dtype if self.arithmetic == "input" else torch.float32
        x = self.codes.to(dtype) * self.scale.to(dtype) + self.offset.to(dtype)
        if self.value:
            return x.reshape(self.shape).to(self.dtype)
        return x.reshape(self.shape[0], -1, *self.shape[2:])[:, :self.shape[1]].to(self.dtype)


class KiviGroupedCandidate:
    """[B,L,H,D] adapter with K token groups and V per-head D groups.

    Finite floating inputs only; restores BF16 regardless of input dtype.
    K token tails are zero padded before extrema; V requires D % group == 0.
    """

    def __init__(self, key_bits=4, value_bits=4, group=16, arithmetic="fp32"):
        if any(type(b) is not int or b not in (2, 4, 8) for b in (key_bits, value_bits)):
            raise ValueError("key/value bits must be 2, 4, or 8")
        if type(group) is not int or group <= 0:
            raise ValueError("group must be a positive integer")
        if arithmetic not in ("input", "fp32"):
            raise ValueError("arithmetic must be input or fp32")
        self.key_bits, self.value_bits, self.group = key_bits, value_bits, group
        self.arithmetic = arithmetic

    def encode(self, x, *, value=False):
        if (x.ndim != 4 or not x.is_floating_point()
                or any(n == 0 for n in x.shape) or not torch.isfinite(x).all()):
            raise ValueError("expected finite nonempty floating [B,L,H,D]")
        shape = tuple(x.shape)
        b, length, heads, channels = shape
        if self.arithmetic == "input" and x.dtype != torch.bfloat16:
            raise ValueError("input arithmetic comparison requires BF16 input")
        data = x if self.arithmetic == "input" else x.float()
        if value:
            if channels % self.group:
                raise ValueError("value head dimension D must be divisible by group")
            data = data.reshape(b, length, heads, channels // self.group, self.group)
            axis = -1
        else:
            pad = (-length) % self.group
            if pad:
                data = torch.cat((data, data.new_zeros(b, pad, heads, channels)), dim=1)
            data = data.reshape(b, -1, self.group, heads, channels)
            axis = 2
        bits = self.value_bits if value else self.key_bits
        levels = (1 << bits) - 1
        offset = data.amin(axis, keepdim=True)
        floor = 1e-8 if self.arithmetic == "input" else MIN_SCALE
        scale = ((data.amax(axis, keepdim=True) - offset) / levels).clamp_min(floor)
        codes = ((data - offset) / scale).clamp(0, levels).round().to(torch.uint8)
        scale16, offset16 = scale.to(torch.float16), offset.to(torch.float16)
        if not torch.isfinite(scale16).all() or not torch.isfinite(offset16).all():
            raise ValueError("FP16 metadata overflow; input range unsupported")
        return GroupedQuantized(codes, scale16, offset16, shape, torch.bfloat16,
                                bits, value, self.arithmetic)

    def encode_pair(self, k, v):
        if k.shape != v.shape:
            raise ValueError("key/value shapes must match")
        return self.encode(k), self.encode(v, value=True)
