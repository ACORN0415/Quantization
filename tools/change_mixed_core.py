"""Actually packed, fixed-original-group mixed precision for development diagnostics.

Input is finite [1,N,heads,channels]. Group maxima are computed once over the
original token order, independent of the precision mask. A single stored FP16
maximum per original group/feature defines scales M/qmax(bits) in FP32. Selecting
low/high token streams changes byte layout, never the calibration groups.

Bit streams are LSB-first: scalar i starts at bit i*bits, and lower bits occupy
lower positions in each byte. Only real tokens are coded; final zero padding is
used to calculate the last group maximum. The mixed mask is itself one-bit
packed. Optional shared packed_mask retains the exact same caller tensor, so a
parent can share it across K/V/layers and account unique storages only once.

These are pure PyTorch packing/unpacking operations, not fused attention kernels.
restore() returns FP32 at original positions. bytes() counts actual tensor
storage payload, including packed mask and FP16 maxima, not Python/header bytes.
"""
from dataclasses import dataclass
import math

import torch


MIN_MAXIMUM = 2.0**-24


def _bit_width(bits, *, mask=False):
    allowed = (1, 2, 3, 4) if mask else (2, 3, 4)
    if type(bits) is not int or bits not in allowed:
        raise ValueError(f'bits must be one of {allowed}')


def pack_bits(codes, bits):
    """Pack integer scalar codes into an actual contiguous uint8 byte stream."""
    _bit_width(bits, mask=True)
    if not isinstance(codes, torch.Tensor) or codes.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
        raise ValueError('codes must be an integer or boolean tensor')
    flat = codes.reshape(-1)
    if flat.numel() and not bool(((flat >= 0) & (flat < (1 << bits))).all()):
        raise ValueError('code outside bit range')
    count = flat.numel()
    if not count:
        return torch.empty(0, dtype=torch.uint8, device=codes.device)
    if bits in (1, 2, 4):
        per_byte = 8 // bits
        extra = (-count) % per_byte
        x = flat.to(torch.int32)
        if extra:
            x = torch.cat((x, x.new_zeros(extra)))
        shifts = torch.arange(per_byte, device=x.device, dtype=torch.int32)*bits
        packed = (x.reshape(-1, per_byte) << shifts).sum(1, dtype=torch.int32)
        return packed.to(torch.uint8)
    # Three-bit fields cross bytes. Contributions never overlap, so integer
    # addition equals bitwise OR and cannot overflow a byte (sum <=255).
    bit = torch.arange(count, device=codes.device, dtype=torch.int64)*bits
    byte = bit // 8
    shift = (bit % 8).to(torch.int32)
    x = flat.to(torch.int32)
    out = torch.zeros(math.ceil(count*bits/8), device=codes.device, dtype=torch.int32)
    out.scatter_add_(0, byte, (x << shift) & 255)
    crossing = (shift+bits > 8) & (byte+1 < out.numel())
    out.scatter_add_(0, byte[crossing]+1, (x >> (8-shift))[crossing])
    return out.to(torch.uint8)


def unpack_bits(packed, bits, count):
    """Inverse of pack_bits; exactly count scalar codes, trailing bits ignored."""
    _bit_width(bits, mask=True)
    if (not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8
            or packed.ndim != 1 or type(count) is not int or count < 0
            or packed.numel() != math.ceil(count*bits/8)):
        raise ValueError('expected uint8 byte vector of exactly ceil(count*bits/8)')
    if not count:
        return torch.empty(0, dtype=torch.uint8, device=packed.device)
    bit = torch.arange(count, device=packed.device, dtype=torch.int64)*bits
    byte = bit // 8
    shift = (bit % 8).to(torch.int32)
    x = packed[byte].to(torch.int32) >> shift
    crossing = shift+bits > 8
    if bool(crossing.any()):
        high = packed[byte[crossing]+1].to(torch.int32)
        x[crossing] |= high << (8-shift[crossing])
    return (x & ((1 << bits)-1)).to(torch.uint8)


def _frame(x, group):
    if (not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[0] != 1
            or any(n <= 0 for n in x.shape) or not x.is_floating_point()
            or not bool(torch.isfinite(x).all())):
        raise ValueError('expected finite nonempty floating [1,N,heads,channels]')
    if type(group) is not int or group <= 0:
        raise ValueError('group must be a positive integer')


def _group_maximum(x, group):
    count = x.shape[1]
    data = x.float()
    padding = (-count) % group
    if padding:
        data = torch.cat((data, data.new_zeros(1, padding, *data.shape[2:])), 1)
    maximum = data.reshape(1, -1, group, *x.shape[2:]).abs().amax(2, keepdim=True)
    maximum = maximum.clamp_min(MIN_MAXIMUM).to(torch.float16)
    if not bool(torch.isfinite(maximum).all()):
        raise ValueError('FP16 group maximum overflow')
    return maximum


def _token_maximum(maximum, count, group):
    indices = torch.arange(count, device=maximum.device)//group
    return maximum.index_select(1, indices)[:, :, 0].float()


def _encode_values(values, maxima, bits):
    qmax = (1 << (bits-1))-1
    scale = maxima/qmax
    codes = (values.float()/scale).round().clamp(-qmax, qmax).add(qmax).to(torch.uint8)
    return pack_bits(codes, bits)


def _storage_bytes(tensor):
    return tensor.untyped_storage().nbytes()


@dataclass
class PackedMixed:
    group_max: torch.Tensor
    low_stream: torch.Tensor
    high_stream: torch.Tensor | None
    packed_mask: torch.Tensor | None
    shape: tuple
    group: int
    low_bits: int
    high_bits: int | None
    input_dtype: torch.dtype
    high_tokens: int = 0

    def restore(self):
        _, count, heads, dim = self.shape
        maximum = _token_maximum(self.group_max, count, self.group)
        if self.packed_mask is None:
            qmax = (1 << (self.low_bits-1))-1
            codes = unpack_bits(self.low_stream, self.low_bits, count*heads*dim)
            return (codes.reshape(self.shape).float()-qmax)*(maximum/qmax)
        mask = unpack_bits(self.packed_mask, 1, count).bool()
        high = mask.nonzero(as_tuple=False).flatten()
        low = (~mask).nonzero(as_tuple=False).flatten()
        out = maximum.new_empty(self.shape)
        for indices, stream, bits in ((low, self.low_stream, self.low_bits),
                                      (high, self.high_stream, self.high_bits)):
            if not indices.numel():
                continue
            qmax = (1 << (bits-1))-1
            codes = unpack_bits(stream, bits, indices.numel()*heads*dim)
            restored = (codes.reshape(1, indices.numel(), heads, dim).float()-qmax)
            restored *= maximum.index_select(1, indices)/qmax
            out.index_copy_(1, indices, restored)
        return out

    def bytes(self):
        """Actual tensor storage; external sharing must be deduplicated by owner."""
        tensors = (self.group_max, self.low_stream, self.high_stream, self.packed_mask)
        seen = set(); resident = 0
        for tensor in tensors:
            if tensor is None:
                continue
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                resident += storage.nbytes(); seen.add(key)
        return dict(resident=resident, packed_estimate=resident,
                    actual_packed_bytes=resident, group_max_bytes=_storage_bytes(self.group_max),
                    mask_bytes=0 if self.packed_mask is None else _storage_bytes(self.packed_mask),
                    code_bytes=_storage_bytes(self.low_stream)+(0 if self.high_stream is None else _storage_bytes(self.high_stream)))


def encode_uniform(x, bits=3, group=16):
    _frame(x, group); _bit_width(bits)
    maximum = _group_maximum(x, group)
    token_maximum = _token_maximum(maximum, x.shape[1], group)
    stream = _encode_values(x, token_maximum, bits)
    return PackedMixed(maximum, stream, None, None, tuple(x.shape), group,
                       bits, None, x.dtype)


def encode_mixed(x, high_mask, low_bits=2, high_bits=4, group=16, packed_mask=None):
    _frame(x, group); _bit_width(low_bits); _bit_width(high_bits)
    if (not isinstance(high_mask, torch.Tensor) or high_mask.dtype != torch.bool
            or tuple(high_mask.shape) != (x.shape[1],) or high_mask.device != x.device):
        raise ValueError('high_mask must be same-device bool [N]')
    if packed_mask is None:
        packed_mask = pack_bits(high_mask, 1)
    else:
        if (packed_mask.device != x.device
                or not torch.equal(unpack_bits(packed_mask, 1, x.shape[1]).bool(), high_mask)):
            raise ValueError('shared packed_mask does not encode high_mask')
    maximum = _group_maximum(x, group)
    token_maximum = _token_maximum(maximum, x.shape[1], group)
    high = high_mask.nonzero(as_tuple=False).flatten()
    low = (~high_mask).nonzero(as_tuple=False).flatten()
    low_stream = _encode_values(x.index_select(1, low), token_maximum.index_select(1, low), low_bits)
    high_stream = _encode_values(x.index_select(1, high), token_maximum.index_select(1, high), high_bits)
    return PackedMixed(maximum, low_stream, high_stream, packed_mask, tuple(x.shape), group,
                       low_bits, high_bits, x.dtype, high.numel())
