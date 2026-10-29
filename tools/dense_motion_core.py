"""Local correspondence and affine prediction for a development KV codec probe.

This file does not load research captures or integrate attention. Frames remain
[B, height*width, heads, channels] in destination token order. A map stores int8
(source_y-destination_y, source_x-destination_x) per destination, shared across
heads/features. Every source must be in bounds: no border zeroing, escape,
wraparound, or token merging occurs. Callers must pass reconstructed references,
count the map once at its actual ownership scope, and retain/count anchor codes.
An affine predictor is a stronger same-position baseline, NOT TempoKV itself.
"""
from dataclasses import dataclass

import torch


def _frame(x):
    if (x.ndim != 4 or any(v <= 0 for v in x.shape) or not x.is_floating_point()
            or not bool(torch.isfinite(x).all())):
        raise ValueError('expected finite nonempty floating [B,N,H,D]')


def _geometry(reference, height, width):
    _frame(reference)
    if (type(height) is not int or type(width) is not int or height <= 0
            or width <= 0 or height*width != reference.shape[1]):
        raise ValueError('height*width must equal spatial token count')


def _coordinates(reference, height, width):
    y = torch.arange(height, device=reference.device).repeat_interleave(width)
    x = torch.arange(width, device=reference.device).repeat(height)
    return y, x


def gather_reference(reference, offsets, height, width):
    """Return source samples in original destination ordering, without zero borders."""
    _geometry(reference, height, width)
    b, n, heads, dim = reference.shape
    if (offsets.dtype != torch.int8 or tuple(offsets.shape) != (b, n, 2)
            or offsets.device != reference.device):
        raise ValueError('expected same-device int8 [B,N,2] source offsets')
    y, x = _coordinates(reference, height, width)
    sy = y[None] + offsets[..., 0].long()
    sx = x[None] + offsets[..., 1].long()
    if not bool(((sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)).all()):
        raise ValueError('map contains out-of-bounds source')
    indices = (sy*width+sx)[..., None, None].expand(b, n, heads, dim)
    return reference.gather(1, indices)


def choose_map(target, reconstructed_reference, height, width, radius=2):
    """Minimize current full-feature K prediction MSE at each destination token.

    Candidate offsets are a square integer window and always include identity.
    Identity wins exact ties, then smaller displacement/lexicographic order.
    This minimizes predictor MSE, not quantized residual or attention error.
    Current target K is available at encode time; no future query is inspected.
    """
    _geometry(reconstructed_reference, height, width); _frame(target)
    if target.shape != reconstructed_reference.shape or target.device != reconstructed_reference.device:
        raise ValueError('target/reference shapes and devices must match')
    if type(radius) is not int or not 0 <= radius <= 127:
        raise ValueError('radius must be integer in 0..127')
    reference = reconstructed_reference
    b, n, _, _ = reference.shape
    y, x = _coordinates(reference, height, width)
    target32 = target.float()
    best = (target32-reference.float()).square().mean((-2, -1))
    if not bool(torch.isfinite(best).all()):
        raise ValueError('nonfinite FP32 prediction MSE')
    chosen = torch.zeros((b, n, 2), device=reference.device, dtype=torch.int8)
    shifts = sorted(((dy, dx) for dy in range(-radius, radius+1)
                     for dx in range(-radius, radius+1) if dy or dx),
                    key=lambda s: (s[0]*s[0]+s[1]*s[1], s))
    for dy, dx in shifts:
        sy, sx = y+dy, x+dx
        valid = (sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)
        indices = sy.clamp(0, height-1)*width+sx.clamp(0, width-1)
        prediction = reference[:, indices]
        cost = (target32-prediction.float()).square().mean((-2, -1))
        if not bool(torch.isfinite(cost).all()):
            raise ValueError('nonfinite FP32 candidate prediction MSE')
        improve = valid[None] & (cost < best)
        best = torch.where(improve, cost, best)
        chosen[..., 0] = torch.where(improve, torch.full_like(chosen[..., 0], dy), chosen[..., 0])
        chosen[..., 1] = torch.where(improve, torch.full_like(chosen[..., 1], dx), chosen[..., 1])
    return chosen


def map_bytes(offsets):
    """Actual int8 map payload, excluding anchors and external format headers."""
    if offsets.dtype != torch.int8 or offsets.ndim != 3 or offsets.shape[-1] != 2:
        raise ValueError('expected int8 [B,N,2] offsets')
    return offsets.numel()*offsets.element_size()


@dataclass
class AffinePredictor:
    """Per-head/channel FP16 a,b; evaluate in FP32 at any original destination."""
    a: torch.Tensor
    b: torch.Tensor
    training_tokens: int
    ridge: float

    def predict(self, reference):
        _frame(reference)
        if (tuple(reference.shape[2:]) != tuple(self.a.shape)
                or self.a.device != reference.device):
            raise ValueError('predictor feature shape/device mismatch')
        return reference.float()*self.a.float()+self.b.float()

    def bytes(self):
        payload = self.a.numel()*self.a.element_size()+self.b.numel()*self.b.element_size()
        return dict(resident=payload, packed_estimate=payload, coefficient_bytes=payload)


class AffineStats:
    """Streaming FP64 training-only sufficient statistics per head/channel.

    The caller selects fitting/evaluation slices. update() never retains frames.
    fit() uses a = cov(x,y)/(var(x)+ridge), b = mean(y)-a*mean(x), where ridge
    has feature-variance units. Fitting a baseline and fitting aligned references
    require separate instances; do not update with evaluation targets.
    """
    def __init__(self):
        self.count = 0
        self.sums = None

    def update(self, reconstructed_reference, target):
        _frame(reconstructed_reference); _frame(target)
        if (target.shape != reconstructed_reference.shape
                or target.device != reconstructed_reference.device):
            raise ValueError('training shapes and devices must match')
        x, y = reconstructed_reference.double(), target.double()
        additions = (x.sum((0, 1)), y.sum((0, 1)), x.square().sum((0, 1)), (x*y).sum((0, 1)))
        if self.sums is None:
            self.sums = [v.clone() for v in additions]
        else:
            if self.sums[0].shape != additions[0].shape or self.sums[0].device != x.device:
                raise ValueError('training feature shape/device changed')
            for total, value in zip(self.sums, additions): total.add_(value)
        self.count += x.shape[0]*x.shape[1]
        return self

    def fit(self, ridge=1e-4):
        if self.count < 2 or self.sums is None:
            raise ValueError('at least two training tokens required')
        if type(ridge) not in (int, float) or not 0 <= ridge < float('inf'):
            raise ValueError('ridge must be finite and nonnegative')
        sx, sy, sxx, sxy = self.sums
        mx, my = sx/self.count, sy/self.count
        var = (sxx/self.count-mx.square()).clamp_min(0)
        cov = sxy/self.count-mx*my
        denominator = var+ridge
        a = torch.where(denominator > 0, cov/denominator.clamp_min(torch.finfo(torch.float64).tiny),
                        torch.zeros_like(cov))
        b = my-a*mx
        a16, b16 = a.to(torch.float16), b.to(torch.float16)
        if not bool(torch.isfinite(a16).all() and torch.isfinite(b16).all()):
            raise ValueError('affine coefficient FP16 overflow')
        return AffinePredictor(a16, b16, self.count, float(ridge))
