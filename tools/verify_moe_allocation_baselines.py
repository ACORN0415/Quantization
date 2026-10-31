"""Verify CPU objective identities and reference packing; no model or CUDA use."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from moe_allocation_objectives import (
    BITS, diagonal_scores, fixed_multiset_candidates, gram_scores,
    incremental_scores, lexicographic_choice, pack_reference,
    reference_payload_bytes, unpack_reference,
)


def direct_reference(bank, candidates, offset):
    # Independent scalar accumulation. This is a correctness audit, not B's
    # implementation or a deliberately slow performance baseline.
    result = []
    for candidate in candidates:
        squares = []
        for t in range(bank.shape[0]):
            for d in range(bank.shape[-1]):
                value = math.fsum([float(offset[t, d])] + [
                    float(bank[t, e, BITS.index(bit), d])
                    for e, bit in enumerate(candidate)
                ])
                squares.append(value * value)
        result.append(math.fsum(squares) / bank.shape[0])
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20261031)
    candidates = fixed_multiset_candidates()
    assert len(candidates) == 12
    cases = []
    for index in range(24):
        delta = rng.normal(size=(29, 4, 3, 9))
        gates = rng.uniform(0.01, 0.5, size=(29, 4))
        gates[rng.random((29, 4)) < 0.6] = 0
        bank = delta * gates[:, :, None, None]
        offset = rng.normal(scale=0.3, size=(29, 9)) if index % 2 else np.zeros((29, 9))
        cases.append((f'sparse_{index:02d}', bank, offset))

    bank = rng.normal(size=(20, 4, 3, 5))
    for t in range(20):
        bank[t, np.arange(4) != t % 4] = 0
    cases.append(('mutually_exclusive_experts', bank, np.zeros((20, 5))))
    coefficients = np.array([[3, 2, 1], [-3, -2, -1],
                             [2, 1, 0.5], [1, 0.5, 0.25]], dtype=np.float64)
    cases.append(('synthetic_cancellation_changes_choice', coefficients[None, :, :, None],
                  np.zeros((1, 1))))
    cases.append(('all_zero_tie', np.zeros((7, 4, 3, 5)), np.zeros((7, 5))))
    repeated = rng.normal(size=(7, 4, 1, 5))
    cases.append(('precision_independent_tie', np.repeat(repeated, 3, axis=2), np.zeros((7, 5))))
    # Dyadic values exercise strong cancellation without prescribing an
    # unrealistic bitwise equality between different floating-point reductions.
    strong = np.array([[1024, 512, 256], [-1024, -512, -256],
                       [0.5, 0.25, 0.125], [-0.5, -0.25, -0.125]])
    cases.append(('strong_cancellation', strong[None, :, :, None], np.zeros((1, 1))))

    records = []
    for name, bank, offset in cases:
        reference = direct_reference(bank, candidates, offset)
        a = diagonal_scores(bank, candidates, offset)
        b, b_trace = incremental_scores(bank, candidates, offset)
        c, c_trace = gram_scores(bank, candidates, offset)
        np.testing.assert_allclose(b, reference, atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(c, reference, atol=1e-10, rtol=1e-9)
        selected = {key: lexicographic_choice(values, candidates)[0]
                    for key, values in [('A', a), ('B', b), ('C', c), ('reference', reference)]}
        assert selected['B'] == selected['C'] == selected['reference']
        reverse_b, _ = incremental_scores(bank, candidates[::-1], offset)
        np.testing.assert_allclose(reverse_b[::-1], reference, atol=1e-10, rtol=1e-9)
        assert lexicographic_choice(reverse_b, candidates[::-1])[0] == selected['reference']
        # Both B and C may stream token tiles and release their local error bank.
        streamed_b = np.zeros(12)
        streamed_c = np.zeros(12)
        for indices in np.array_split(np.arange(len(bank)), min(3, len(bank))):
            tile_b, _ = incremental_scores(bank[indices], candidates, offset[indices])
            tile_c, _ = gram_scores(bank[indices], candidates, offset[indices])
            streamed_b += tile_b * len(indices) / len(bank)
            streamed_c += tile_c * len(indices) / len(bank)
        np.testing.assert_allclose(streamed_b, reference, atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(streamed_c, reference, atol=1e-10, rtol=1e-9)
        if name == 'mutually_exclusive_experts':
            np.testing.assert_allclose(a, reference, atol=1e-10, rtol=1e-9)
        if name == 'synthetic_cancellation_changes_choice':
            assert selected['A'] != selected['B']
            assert reference[candidates.index(selected['B'])] < reference[candidates.index(selected['A'])]
        if name.endswith('_tie'):
            assert selected['reference'] == min(candidates)
        records.append(dict(name=name, candidate_count=len(candidates),
                            max_absolute_B_reference=float(np.max(np.abs(b-reference))),
                            max_absolute_C_reference=float(np.max(np.abs(c-reference))),
                            selected=selected, B_trace=b_trace, C_trace=c_trace))

    packing = []
    for bits in BITS:
        for columns in (1, 9, 10, 11, 15, 16, 17, 31, 32, 33):
            codes = rng.integers(0, 2**bits, size=(3, columns), dtype=np.uint8)
            packed = pack_reference(codes, bits)
            np.testing.assert_array_equal(unpack_reference(packed, bits, columns), codes)
            for row in range(len(codes)):
                per_word = 32 // bits
                for word in range(packed.shape[0]):
                    expected = sum(int(codes[row, col]) * 2**((col % per_word)*bits)
                                   for col in range(word*per_word, min((word+1)*per_word, columns)))
                    actual = int(packed[word, row]) & 0xffffffff
                    assert actual == expected
            packing.append(dict(bits=bits, columns=columns, bytes=packed.nbytes))
        full_word = np.full((1, 32//bits), 2**bits-1, dtype=np.uint8)
        assert (int(pack_reference(full_word, bits)[0, 0]) & 0xffffffff) == (1 << ((32//bits)*bits))-1

    shapes = [(1024, 2048), (1024, 2048), (2048, 1024)]
    per_expert = {bit: reference_payload_bytes(shapes, bit) for bit in BITS}
    # Allocate the reference tensors to check actual nbytes, not just the formula.
    for bits in BITS:
        observed = 0
        for rows, columns in shapes:
            packed = pack_reference(np.zeros((rows, columns), dtype=np.uint8), bits)
            metadata = np.empty((2, rows*columns//128), dtype=np.uint16)
            observed += packed.nbytes + metadata.nbytes
        assert observed == per_expert[bits]
    candidate_bytes = [sum(per_expert[b] for b in candidate) for candidate in candidates]
    assert len(set(candidate_bytes)) == 1
    all_three_bytes = 4 * per_expert[3]
    assert all_three_bytes != candidate_bytes[0]

    source_paths = [Path(__file__), Path(__file__).with_name('moe_allocation_objectives.py')]
    result = dict(
        schema_version=1, status='PASS', scope='CPU synthetic identities and reference layout only',
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        seed=20261031, dtype='float64', atol=1e-10, rtol=1e-9,
        objective_cases=len(records), candidate_scores_per_method=len(records)*len(candidates),
        objective_cases_detail=records, reference_packing_cases=packing,
        reference_payload=dict(expert_shapes=shapes, group_size=128,
                               metadata_bytes_per_group=4, per_expert_bytes=per_expert,
                               candidate_multiset=[2, 3, 3, 4], candidate_count=12,
                               candidate_bytes=candidate_bytes[0], all_three_bytes=all_three_bytes,
                               extra_bytes_all_three=all_three_bytes-candidate_bytes[0],
                               excludes='Other deployment metadata, file container, common BF16 weights'),
        elapsed_seconds=time.perf_counter()-started, model_forwards=0, gpu_calls=0,
        real_MoE_accuracy_measured=False, performance_claim=False,
        source_sha256={str(path.name): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status', 'objective_cases', 'candidate_scores_per_method',
                                          'elapsed_seconds', 'reference_payload')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
