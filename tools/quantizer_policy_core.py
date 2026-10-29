"""Cross-study adapters. Previously registered numerical files stay frozen."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/scripts')]
import torch
from kvq.quant import Quantizer, Quantized
from kvq.memory import memory_report
from kvq.store import RollingStore
from mechanism_core import make as original_make


def make_store(method, arm, group=16, bits=4, capacity=32760, identity=None):
    q = identity if identity is not None else Quantizer(method, bits, bits, group)
    return original_make(arm, group, bits, capacity, identity=q)


def storage_bytes(store):
    """Count KIVI offsets as well as scales; never double count code padding."""
    payload = store.state if isinstance(store, RollingStore) else [s['state'] for s in store.segments]
    pending = getattr(store, 'pending', None)
    residual = getattr(store, 'residual', None)

    def walk(value):
        if isinstance(value, Quantized):
            yield value
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from walk(child)

    valid = padding = scales = offsets = 0
    for q in walk(payload):
        n = math.prod(q.shape)
        valid += n * q.codes.element_size()
        padding += (q.codes.numel() - n) * q.codes.element_size()
        scales += q.scale.numel() * q.scale.element_size()
        if q.offset is not None:
            offsets += q.offset.numel() * q.offset.element_size()
    report = memory_report([payload, pending, residual])
    pending_bytes = memory_report(pending)['resident_tensor_bytes']
    residual_bytes = memory_report(residual)['resident_tensor_bytes']
    report.update(valid_code_bytes=valid, padding_code_bytes=padding,
                  scale_bytes=scales, offset_bytes=offsets,
                  bf16_pending_bytes=pending_bytes, bf16_residual_bytes=residual_bytes)
    if report['resident_tensor_bytes'] != valid + padding + scales + offsets + pending_bytes + residual_bytes:
        raise RuntimeError('cross-study storage accounting mismatch')
    return report


def interaction(rows):
    """Positive = KIVI advantage over RTN grows under chunk-local repeat."""
    cells = {(r['method'], r['arm']): float(r['late_raw_musiq']) for r in rows}
    if len(rows) != 4 or len(cells) != 4 or set(cells) != {(m, a) for m in ('rtn', 'kivi') for a in ('L', 'C_repeat')}:
        raise ValueError('exactly one result per factorial cell required')
    if not all(math.isfinite(v) for v in cells.values()):
        raise ValueError('finite factorial cells required')
    return (cells['kivi', 'C_repeat'] - cells['rtn', 'C_repeat']) - (cells['kivi', 'L'] - cells['rtn', 'L'])
