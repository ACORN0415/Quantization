"""Count tensor storage bytes; packed estimates never describe actual residency."""
from dataclasses import fields, is_dataclass
import torch
from .quant import Quantized


def memory_report(value):
    seen = set()
    resident = 0
    packed = 0

    def visit(obj):
        nonlocal resident, packed
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            key = (str(obj.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                resident += storage.nbytes()
                packed += storage.nbytes()
        elif isinstance(obj, Quantized):
            before = resident
            for f in fields(obj):
                visit(getattr(obj, f.name))
            if resident > before:
                packed -= obj.codes.untyped_storage().nbytes()
                packed += (obj.codes.numel() * obj.bits + 7) // 8
        elif isinstance(obj, dict):
            for x in obj.values(): visit(x)
        elif isinstance(obj, (tuple, list)):
            for x in obj: visit(x)

    visit(value)
    return {"resident_tensor_bytes": resident, "hypothetical_packed_bytes": packed}
