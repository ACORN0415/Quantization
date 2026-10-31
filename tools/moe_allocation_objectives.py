"""CPU reference objectives for a fixed-routing MoE allocation diagnostic.

This module neither loads a model nor defines a quantizer. Inputs are a cached
bank u[t,e,p,d] = g[t,e] * (expert_quantized - expert_reference), evaluated in
float64. Exactness refers to this real-valued aggregation, not BF16 block
accumulation. The small reference implementation is not a performance result.
"""
from itertools import permutations
import numpy as np


BITS = (2, 3, 4)


def fixed_multiset_candidates(multiset=(2, 3, 3, 4)):
    return sorted(set(permutations(multiset)))


def _checked(bank, candidates, offset):
    bank = np.asarray(bank, dtype=np.float64)
    if bank.ndim != 4 or bank.shape[2] != len(BITS):
        raise ValueError('Expected [tokens, experts, three precisions, dimension]')
    t, e, _, d = bank.shape
    if not t or not d or not np.isfinite(bank).all():
        raise ValueError('Empty/nonfinite bank')
    if not candidates or any(len(c) != e or any(b not in BITS for b in c) for c in candidates):
        raise ValueError('Invalid candidate shape or precision')
    offset = np.zeros((t, d), dtype=np.float64) if offset is None else np.asarray(offset, dtype=np.float64)
    if offset.shape != (t, d) or not np.isfinite(offset).all():
        raise ValueError('Invalid fixed error offset')
    return bank, offset


def diagonal_scores(bank, candidates, offset=None):
    """A: omit cross-expert terms. A fixed offset adds only a constant."""
    u, fixed = _checked(bank, candidates, offset)
    diag = np.einsum('tepd,tepd->ep', u, u) / len(u)
    constant = np.vdot(fixed, fixed) / len(u)
    return np.array([constant + sum(diag[e, BITS.index(b)] for e, b in enumerate(c))
                     for c in candidates])


def incremental_scores(bank, candidates, offset=None):
    """B: cache the residual, update changed experts on affected token rows.

    A new candidate never re-runs expert forward passes. Both residual and its
    squared norm are updated, including all cross terms with unchanged experts.
    """
    u, fixed = _checked(bank, candidates, offset)
    active = np.any(u != 0, axis=(2, 3))
    previous = candidates[0]
    residual = fixed.copy()
    for e, bit in enumerate(previous):
        residual += u[:, e, BITS.index(bit)]
    square_sum = float(np.vdot(residual, residual))
    scores = [square_sum / len(u)]
    changed_rows = [int(active.any(axis=1).sum())]
    for candidate in candidates[1:]:
        changed = [e for e, (old, new) in enumerate(zip(previous, candidate)) if old != new]
        rows = np.flatnonzero(active[:, changed].any(axis=1)) if changed else np.array([], dtype=int)
        delta = np.zeros((len(rows), u.shape[-1]), dtype=np.float64)
        for e in changed:
            delta += u[rows, e, BITS.index(candidate[e])]
            delta -= u[rows, e, BITS.index(previous[e])]
        old_residual = residual[rows]
        square_sum += 2 * float(np.vdot(old_residual, delta)) + float(np.vdot(delta, delta))
        residual[rows] = old_residual + delta
        scores.append(square_sum / len(u))
        changed_rows.append(len(rows))
        previous = candidate
    return np.asarray(scores), dict(updated_token_rows=changed_rows)


def gram_scores(bank, candidates, offset=None):
    """C: precompute only usable precision cross terms on co-active tokens."""
    u, fixed = _checked(bank, candidates, offset)
    t, experts, precisions, _ = u.shape
    active = np.any(u != 0, axis=(2, 3))
    gram = np.zeros((experts * precisions, experts * precisions), dtype=np.float64)
    linear = np.zeros(experts * precisions, dtype=np.float64)
    for e in range(experts):
        block = slice(e * precisions, (e+1) * precisions)
        for p in range(precisions):
            gram[e*precisions+p, e*precisions+p] = np.vdot(u[:, e, p], u[:, e, p])
            linear[e*precisions+p] = np.vdot(fixed, u[:, e, p])
        for f in range(e+1, experts):
            rows = np.flatnonzero(active[:, e] & active[:, f])
            other = slice(f * precisions, (f+1) * precisions)
            # No pair can contribute unless both experts are selected on a token.
            if len(rows):
                left = u[rows, e].transpose(1, 0, 2).reshape(precisions, -1)
                right = u[rows, f].transpose(1, 0, 2).reshape(precisions, -1)
                gram[block, other] = left @ right.T
                gram[other, block] = gram[block, other].T
    constant = float(np.vdot(fixed, fixed))
    scores = []
    for c in candidates:
        indices = [e*precisions + BITS.index(bit) for e, bit in enumerate(c)]
        scores.append((constant + 2 * linear[indices].sum() +
                       gram[np.ix_(indices, indices)].sum()) / t)
    return np.asarray(scores), dict(gram_bytes=gram.nbytes, linear_bytes=linear.nbytes)


def lexicographic_choice(scores, candidates, atol=1e-10, rtol=1e-9):
    """Fixed tie rule for this FP64 diagnostic; never infer quality from a tie."""
    scores = np.asarray(scores)
    if len(scores) != len(candidates) or not np.isfinite(scores).all():
        raise ValueError('Invalid scores')
    best = float(scores.min())
    band = atol + rtol * abs(best)
    tied = [c for c, loss in zip(candidates, scores) if loss <= best + band]
    return min(tied), dict(best_value=best, tie_count=len(tied), band=band)


def pack_reference(codes, bits):
    """CPU reference for GEMQ's 32-bit word layout, not its GPU kernel."""
    codes = np.asarray(codes)
    if codes.ndim != 2 or bits not in BITS or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError('Invalid codes or precision')
    if np.any(codes < 0) or np.any(codes >= 2**bits):
        raise ValueError('Out-of-range code')
    per_word = 32 // bits
    rows, cols = codes.shape
    packed = np.zeros((rows, (cols + per_word - 1) // per_word), dtype=np.uint32)
    for slot in range(per_word):
        values = codes[:, slot::per_word].astype(np.uint32)
        packed[:, :values.shape[1]] |= values << (slot * bits)
    return packed.view(np.int32).T


def unpack_reference(packed, bits, columns):
    per_word = 32 // bits
    words = np.asarray(packed).T.view(np.uint32)
    output = np.empty((len(words), columns), dtype=np.uint8)
    for column in range(columns):
        output[:, column] = (words[:, column // per_word] >>
                              ((column % per_word) * bits)) & (2**bits - 1)
    return output


def reference_payload_bytes(shapes, bits, group_size=128):
    """Codes + two 2-byte per-group arrays; other metadata must be counted later."""
    total = 0
    for rows, cols in shapes:
        if cols % group_size:
            raise ValueError('Group size must divide the input dimension')
        total += rows * ((cols + 32//bits - 1) // (32//bits)) * 4
        total += (rows * cols // group_size) * 2 * 2
    return total
