"""Independent tensor implementation of the quantization equations in the reconstruction protocol.

Arithmetic deliberately stays in the input dtype. Codes are simulated in byte
containers, not packed. KIVI uses unsigned byte codes by default. Historical signed-int8 wraparound
is available only through the explicit legacy_unsigned_int8 compatibility flag.
"""
from dataclasses import dataclass
import math

import torch


@dataclass
class Quantized:
    codes: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor | None
    shape: tuple
    dtype: torch.dtype
    bits: int

    def restore(self):
        values = self.codes.to(self.dtype) * self.scale.to(self.dtype)
        if self.offset is not None:
            values = values + self.offset.to(self.dtype)
        return values.reshape(self.shape[0], -1, *self.shape[2:])[:, :self.shape[1]]

    def bytes(self):
        meta = self.scale.numel() * self.scale.element_size()
        if self.offset is not None:
            meta += self.offset.numel() * self.offset.element_size()
        return {"resident": self.codes.numel() * self.codes.element_size() + meta,
                "packed_estimate": math.ceil(self.codes.numel() * self.bits / 8) + meta}


class Quantizer:
    def __init__(self, method="rtn", key_bits=4, value_bits=4, group=16,
                 legacy_unsigned_int8=False):
        self.method = method.lower()
        if self.method not in ("rtn", "kivi"):
            raise ValueError("method must be rtn or kivi")
        if not all(isinstance(b, int) and 2 <= b <= 8 for b in (key_bits, value_bits)):
            raise ValueError("key/value bits must be integers in 2..8")
        if not isinstance(group, int) or group <= 0:
            raise ValueError("group must be a positive integer")
        self.key_bits, self.value_bits, self.group = key_bits, value_bits, group
        self.legacy_unsigned_int8 = legacy_unsigned_int8

    def encode(self, x, *, value=False):
        if x.ndim != 4 or x.shape[1] == 0 or not x.is_floating_point():
            raise ValueError("expected nonempty floating [B,L,H,D]")
        shape = tuple(x.shape)
        pad = (-shape[1]) % self.group
        if pad:
            x = torch.cat((x, x.new_zeros(shape[0], pad, shape[2], shape[3])), 1)
        x = x.reshape(shape[0], -1, self.group, shape[2], shape[3])
        bits = self.value_bits if value else self.key_bits
        if self.method == "rtn":
            high = (1 << (bits - 1)) - 1
            scale = (x.abs().amax(2, keepdim=True) / high).clamp_min(1e-8)
            codes = (x / scale).round().clamp(-high - 1, high).to(torch.int8)
            offset = None
        else:
            dims = (3, 4) if value else (2,)
            offset = x.amin(dims, keepdim=True)
            high = (1 << bits) - 1
            scale = ((x.amax(dims, keepdim=True) - offset) / high).clamp_min(1e-8)
            code_dtype = torch.int8 if self.legacy_unsigned_int8 else torch.uint8
            codes = ((x - offset) / scale).round().clamp(0, high).to(code_dtype)
            offset = offset.to(torch.float16)
        return Quantized(codes, scale.to(torch.float16), offset, shape, x.dtype, bits)

    def encode_pair(self, k, v):
        return self.encode(k), self.encode(v, value=True)
