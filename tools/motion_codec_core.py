"""Development-only pairwise spatial residual codec, not a TempoKV reproduction.

Frames are [B, height*width, heads, channels] at their original token positions.
The caller MUST pass the reconstructed (not original) reference to both encode
and decode. Each target depends on one independently encoded anchor; no reference
chains, eviction, attention integration, RoPE handling, or GPU kernels are hidden
here. Spatial translation is externally selected and shared across all features.
Byte counts describe tensor payloads, excluding Python/container headers. Codes
occupy uint8 containers; packed_estimate is hypothetical, never deployed memory.
FP32 arithmetic + persisted FP16 scales intentionally differs from legacy BF16 RTN.
"""
from dataclasses import dataclass
import math

import torch


MIN_SCALE = 2.0 ** -24


def _validate(x):
    if (x.ndim != 4 or not x.is_floating_point() or any(n == 0 for n in x.shape)
            or not torch.isfinite(x).all()):
        raise ValueError("expected finite nonempty floating [B,spatial,heads,channels]")


def _nbytes(x):
    return x.numel() * x.element_size()


@dataclass
class TensorCode:
    codes: torch.Tensor
    scale: torch.Tensor
    shape: tuple
    dtype: torch.dtype
    bits: int
    group: int

    def restore(self):
        qmax = (1 << (self.bits-1))-1
        x = (self.codes.float()-qmax) * self.scale.float()
        return x.reshape(self.shape[0], -1, *self.shape[2:])[:, :self.shape[1]].to(self.dtype)

    def bytes(self):
        metadata = _nbytes(self.scale)
        return dict(resident=_nbytes(self.codes)+metadata,
                    packed_estimate=math.ceil(self.codes.numel()*self.bits/8)+metadata,
                    code_container_bytes=_nbytes(self.codes), scale_bytes=metadata)


def make_anchor(x, bits=4, group=16):
    """Direct symmetric grouped RTN; grouping is along flattened spatial tokens."""
    _validate(x)
    if type(bits) is not int or bits not in (2, 4, 8):
        raise ValueError("bits must be 2, 4, or 8")
    if type(group) is not int or group <= 0:
        raise ValueError("group must be a positive integer")
    shape = tuple(x.shape)
    data = x.float()
    pad = (-shape[1]) % group
    if pad:
        data = torch.cat((data, data.new_zeros(shape[0], pad, shape[2], shape[3])), 1)
    data = data.reshape(shape[0], -1, group, shape[2], shape[3])
    qmax = (1 << (bits-1))-1
    scale = (data.abs().amax(2, keepdim=True)/qmax).clamp_min(MIN_SCALE)
    scale16 = scale.to(torch.float16)
    if not torch.isfinite(scale16).all():
        raise ValueError("FP16 scale overflow")
    codes = (data/scale).round().clamp(-qmax, qmax).add(qmax).to(torch.uint8)
    return TensorCode(codes, scale16, shape, x.dtype, bits, group)


def _validate_translation(reference, height, width, dy, dx):
    _validate(reference)
    if (type(height) is not int or type(width) is not int or height <= 0
            or width <= 0 or height*width != reference.shape[1]):
        raise ValueError("height*width must equal spatial token count")
    if any(type(v) is not int or not -32768 <= v <= 32767 for v in (dy, dx)):
        raise ValueError("translation must fit signed int16")


def translate_reference(reference, height, width, dy=0, dx=0):
    """Source coordinate for destination (y,x) is (y-dy,x-dx).

    No wraparound: invalid destinations predict zero, so their residual stores
    absolute values. The validity mask is derived from dimensions/translation
    and therefore needs no stored mask. Original token ordering is unchanged.
    """
    _validate_translation(reference, height, width, dy, dx)
    b, _, heads, dim = reference.shape
    src = reference.reshape(b, height, width, heads, dim)
    dst = torch.zeros_like(src)
    y0, y1 = max(0, dy), min(height, height+dy)
    x0, x1 = max(0, dx), min(width, width+dx)
    if y1 > y0 and x1 > x0:
        dst[:, y0:y1, x0:x1] = src[:, y0-dy:y1-dy, x0-dx:x1-dx]
    return dst.reshape_as(reference)


def matched_border_reference(reference, height, width, dy=0, dx=0):
    """Same-position predictor with exactly a translated predictor's zero border.

    Interior values do not move. This control separates correspondence changes
    from disabling prediction in newly visible border regions. The mask is
    inferred from an externally persisted (dy,dx) int16 pair (4 bytes); callers
    must count that metadata in addition to any residual-code metadata.
    """
    # Validate geometry without data-dependent validity tests: true zero features
    # must not be mistaken for invalid tokens.
    _validate_translation(reference, height, width, dy, dx)
    b, _, heads, dim = reference.shape
    src = reference.reshape(b, height, width, heads, dim)
    dst = torch.zeros_like(src)
    y0, y1 = max(0, dy), min(height, height+dy)
    x0, x1 = max(0, dx), min(width, width+dx)
    if y1 > y0 and x1 > x0:
        dst[:, y0:y1, x0:x1] = src[:, y0:y1, x0:x1]
    return dst.reshape_as(reference)


@dataclass
class ResidualCode:
    residual: TensorCode | torch.Tensor
    translation: torch.Tensor
    height: int
    width: int
    dtype: torch.dtype
    shape: tuple
    lossless: bool
    correction_indices: torch.Tensor | None = None
    correction_values: torch.Tensor | None = None

    def bytes(self):
        if self.lossless:
            payload = _nbytes(self.residual)
            if self.correction_indices is not None:
                payload += _nbytes(self.correction_indices)+_nbytes(self.correction_values)
            result = dict(resident=payload, packed_estimate=payload,
                          code_container_bytes=payload, scale_bytes=0)
        else:
            result = self.residual.bytes()
        result = dict(result)
        map_bytes = _nbytes(self.translation)
        result['resident'] += map_bytes
        result['packed_estimate'] += map_bytes
        result['map_bytes'] = map_bytes
        result['reference_bytes_included'] = 0
        return result


def encode_residual(target, reconstructed_reference, *, height, width,
                    dy=0, dx=0, bits=4, group=16, lossless=False):
    """Encode against an explicitly supplied reconstructed reference.

    lossless=True is an expensive diagnostic identity path: FP64 residual plus
    explicit exact escapes if finite-precision cancellation requires them. Its
    bytes are fully counted and it is never a compression performance result.
    """
    _validate(target); _validate(reconstructed_reference)
    if target.shape != reconstructed_reference.shape or target.device != reconstructed_reference.device:
        raise ValueError("target/reference shapes and devices must match")
    prediction = translate_reference(reconstructed_reference, height, width, dy, dx)
    translation = torch.tensor([dy, dx], dtype=torch.int16, device=target.device)
    if lossless:
        residual = target.double()-prediction.double()
        if not torch.isfinite(residual).all():
            raise ValueError("FP64 lossless residual overflow")
        restored = (prediction.double()+residual).to(target.dtype)
        indices = torch.nonzero((restored != target).reshape(-1), as_tuple=False).flatten()
        values = target.reshape(-1)[indices].clone()
        return ResidualCode(residual, translation, height, width, target.dtype,
                            tuple(target.shape), True, indices, values)
    residual = make_anchor(target.float()-prediction.float(), bits, group)
    return ResidualCode(residual, translation, height, width, target.dtype,
                        tuple(target.shape), False)


def decode_residual(code, reconstructed_reference):
    _validate(reconstructed_reference)
    if tuple(reconstructed_reference.shape) != code.shape:
        raise ValueError("reference shape differs from encoded frame")
    dy, dx = (int(v) for v in code.translation.tolist())
    prediction = translate_reference(reconstructed_reference, code.height, code.width, dy, dx)
    if code.lossless:
        restored = (prediction.double()+code.residual).to(code.dtype)
        if code.correction_indices is not None and code.correction_indices.numel():
            restored.reshape(-1)[code.correction_indices] = code.correction_values
        return restored
    return (prediction.float()+code.residual.restore()).to(code.dtype)
