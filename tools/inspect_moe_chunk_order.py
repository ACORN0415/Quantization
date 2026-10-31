"""CPU-only review of chunk-local versus global MoE accumulation order.

This models the control flow in RobustGEMQ at fe6fe6dc679495a53795b2a7eed4efb7e655e29c,
vllm_moe_dispatch.py:111-172. It does not execute that Triton kernel.
Synthetic witnesses are not observations of model damage.
"""
import argparse
import hashlib
import json
import struct
import time
from datetime import datetime, timezone
from pathlib import Path


def rounded(value, kind):
    return struct.unpack(kind, struct.pack(kind, value))[0]


def fp16_bits(value):
    return struct.pack('<e', value).hex()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def reduce_chunks(values, gates, experts, chunk_size, canonical=False):
    sorted_slots = sorted(range(len(experts)), key=lambda slot: experts[slot])
    inverse = {slot: rank for rank, slot in enumerate(sorted_slots)}
    loop = sorted_slots if canonical else list(range(len(experts)))
    accumulator = 0.0
    effective_order = []
    for start in range(0, len(experts), chunk_size):
        for slot in loop:
            inside = start <= inverse[slot] < start + chunk_size
            value = values[slot] if inside else 0.0
            product = rounded(value * gates[slot], '<f')
            # Products of these chosen FP16 operands are exactly FP32-representable.
            assert product == value * gates[slot]
            accumulator = rounded(accumulator + product, '<f')
            if inside:
                effective_order.append(slot)
    result = rounded(accumulator, '<e')
    return dict(fp32=accumulator, fp16=result, fp16_little_endian_hex=fp16_bits(result),
                effective_slot_order=effective_order)


def arithmetic_check(spec):
    experts = spec['synthetic_selected_experts']
    gates = spec['synthetic_gates']
    assert sum(gates) == 1.0 and len(set(experts)) == len(experts)
    results = []
    for raw_values in spec['synthetic_expert_outputs']:
        values = [rounded(v, '<e') for v in raw_values]
        assert values == raw_values
        assert [rounded(g, '<e') for g in gates] == gates
        local = {str(c): reduce_chunks(values, gates, experts, c) for c in [1, 2, 4, 8]}
        canonical = {str(c): reduce_chunks(values, gates, experts, c, True) for c in [1, 2, 4, 8]}
        assert local['2']['fp16_little_endian_hex'] != local['8']['fp16_little_endian_hex']
        assert len({r['fp16_little_endian_hex'] for r in canonical.values()}) == 1
        results.append(dict(values=values, gates=gates, experts=experts,
                            chunk_local_slot_order=local, canonical_expert_order=canonical,
                            exact_real_sum=sum(v*g for v,g in zip(values,gates)),
                            canonical_changes_original_slot_reference=(
                                canonical['8']['fp16_little_endian_hex'] !=
                                local['8']['fp16_little_endian_hex'])))
    return results


def routing_check(path):
    import torch
    torch.set_num_threads(2)
    captures = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    rows = []
    # These are assignment counts, not the upstream user-facing token chunk option.
    capacities = [128, 512, 1024, 4096]
    for item in captures:
        indices = item['indices'].reshape(-1, 8).long()
        tokens, top_k = indices.shape
        flat = indices.flatten()
        assert all(len(set(row)) == top_k for row in indices.tolist())
        order = torch.argsort(flat, stable=True)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel())
        inverse = inverse.reshape(tokens, top_k)
        conditions = []
        for capacity in capacities:
            chunk_ids = inverse // capacity
            # Within each chunk the kernel visits slots in ascending order.
            actual_slot_order = torch.argsort(chunk_ids, dim=1, stable=True)
            canonical_slot_order = torch.arange(top_k)[None].expand(tokens, -1)
            reordered = (actual_slot_order != canonical_slot_order).any(dim=1)
            split = chunk_ids.min(dim=1).values != chunk_ids.max(dim=1).values
            conditions.append(dict(assignment_capacity=capacity,
                                   split_tokens=int(split.sum()),
                                   tokens_with_changed_slot_order=int(reordered.sum())))
        rows.append(dict(document=item['id'], split=item['split'], tokens=tokens,
                         top_k=top_k, active_experts=int(torch.unique(flat).numel()),
                         full_fp16_expert_output_bytes=tokens * top_k * 2048 * 2,
                         fp32_output_accumulator_bytes=tokens * 2048 * 4,
                         conditions=conditions))
    aggregate = []
    for j, capacity in enumerate(capacities):
        aggregate.append(dict(assignment_capacity=capacity,
                              tokens=sum(r['tokens'] for r in rows),
                              split_tokens=sum(r['conditions'][j]['split_tokens'] for r in rows),
                              tokens_with_changed_slot_order=sum(
                                  r['conditions'][j]['tokens_with_changed_slot_order'] for r in rows)))
    return dict(source=str(path), source_sha256=digest(path), documents=rows,
                aggregate=aggregate,
                limitation='BF16 fixed-routing captures from the closed allocation study; not RobustGEMQ routing, not a quantized serving workload; no expert outputs evaluated')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    spec = json.loads(args.scope.read_text())['cpu_check_registered_before_execution']
    result = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope_sha256=digest(args.scope), code_sha256=digest(Path(__file__)),
                  synthetic=arithmetic_check(spec), routing=routing_check(args.captures),
                  gpu_calls=0, model_forwards=0)
    result['elapsed_seconds'] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output),
                          arithmetic_witnesses=len(result['synthetic']),
                          routing=result['routing']['aggregate'],
                          elapsed_seconds=result['elapsed_seconds'], gpu_calls=0), indent=2))


if __name__ == '__main__':
    main()
