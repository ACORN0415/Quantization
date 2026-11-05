"""CPU-only K-only paired Gaussian response summaries; no model/data loading.

All contractions and exponent preparation use FP64. Persistent summaries are
FP32 with a per-chunk/per-feature FP64 common offset. One FP32 Omega is shared
by query, original-key and decoded-key features. No experimental gate is set.
"""
from dataclasses import dataclass
import math
import torch

LOG_MAX64 = math.log(torch.finfo(torch.float64).max)


def require_cpu_float(x, name):
    if not isinstance(x, torch.Tensor) or x.device.type != 'cpu' or not x.is_floating_point():
        raise ValueError(f'{name}: a floating CPU tensor is required')


def tensor_bytes(x):
    return x.numel() * x.element_size()


def _compensated_add(total, residual, term):
    """Elementwise Neumaier sum; callers keep native terms until the end.

    This improves signed floating-point accumulation, not feature accuracy or
    exact real arithmetic. Products/exponentials are still rounded in FP64.
    """
    updated = total + term
    residual = residual + torch.where(
        total.abs() >= term.abs(), (total-updated)+term, (term-updated)+total)
    return updated, residual


class PreparationError(ValueError):
    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.reason, self.diagnostics = reason, dict(diagnostics)


@dataclass
class ChunkState:
    chunk_id: str
    dz: torch.Tensor       # [H,R], FP32, scaled by exp(-offset)
    dn: torch.Tensor       # [H,R,Dv], FP32, same offset
    offset: torch.Tensor   # [H,R], FP64, shared original/decoded feature offset

    def tensor_payload_bytes(self):
        return sum(tensor_bytes(x) for x in (self.dz, self.dn, self.offset))


@dataclass
class ReadResult:
    output: torch.Tensor
    fallback: torch.Tensor            # [H,Q]
    log_denominator_ratio: torch.Tensor
    diagnostics: dict


class PairedResponse:
    def __init__(self, omega):
        require_cpu_float(omega, 'omega')
        if omega.ndim != 3 or min(omega.shape) < 1 or omega.dtype != torch.float32:
            raise ValueError('omega must be FP32 [H,D,R] with positive dimensions')
        if not torch.isfinite(omega).all():
            raise ValueError('omega must be finite')
        self.omega = omega.detach().clone().contiguous()  # the only persistent map
        self.heads, self.dim, self.rank = self.omega.shape
        self.states = {}

    @classmethod
    def gaussian(cls, heads, dim, rank, seed):
        if min(heads, dim, rank) < 1:
            raise ValueError('positive dimensions required')
        gen = torch.Generator(device='cpu').manual_seed(int(seed))
        # IID standard Gaussian rounded to FP32, not normalized/QR directions.
        return cls(torch.randn(heads, dim, rank, generator=gen, dtype=torch.float32))

    def _logs(self, x):
        # phi(x)=exp(Omega^T x/d^.25 - ||x/d^.25||^2/2)/sqrt(R).
        z = x.double() / math.sqrt(math.sqrt(self.dim))
        return torch.bmm(z, self.omega.double()) - .5*z.square().sum(-1, keepdim=True) - .5*math.log(self.rank)

    @torch.no_grad()
    def summarize(self, chunk_id, key, decoded_key, decoded_value):
        diagnostics = {'feature_nonfinite': 0, 'feature_exp_underflow': 0,
                       'naive_exp_subtraction_collision': 0,
                       'nonzero_log_difference_lost': 0,
                       'state_cast_underflow': 0, 'state_nonfinite': 0}
        for name, x in (('key', key), ('decoded_key', decoded_key), ('decoded_value', decoded_value)):
            require_cpu_float(x, name)
        if key.ndim != 3 or key.shape != decoded_key.shape or key.shape[0] != self.heads or key.shape[-1] != self.dim or key.shape[1] < 1:
            raise ValueError('key shapes must agree: [H,N,D], N>0')
        if decoded_value.ndim != 3 or decoded_value.shape[:2] != key.shape[:2] or decoded_value.shape[-1] < 1:
            raise ValueError('decoded_value must have shape [H,N,Dv], Dv>0')
        if not all(bool(torch.isfinite(x).all()) for x in (key, decoded_key, decoded_value)):
            raise PreparationError('nonfinite_input', diagnostics)
        a, b = self._logs(key), self._logs(decoded_key)
        diagnostics['feature_nonfinite'] = int((~torch.isfinite(a)).sum() + (~torch.isfinite(b)).sum())
        if diagnostics['feature_nonfinite']:
            raise PreparationError('nonfinite_feature_log', diagnostics)
        offset = torch.maximum(a.amax(1), b.amax(1))
        aa, bb = a-offset[:, None, :], b-offset[:, None, :]
        pa, pb = aa.exp(), bb.exp()
        diagnostics['feature_exp_underflow'] = int(((pa == 0) & torch.isfinite(aa)).sum() + ((pb == 0) & torch.isfinite(bb)).sum())
        diagnostics['naive_exp_subtraction_collision'] = int(((pa == pb) & (pa > 0) & (a != b)).sum())
        # Stable subtraction for nearby logs. Both expm1 arguments are <=0,
        # so neither branch evaluates an overflowing positive exponential.
        delta = torch.where(a >= b,
                            -pa * torch.expm1((b-a).clamp_max(0)),
                            pb * torch.expm1((a-b).clamp_max(0)))
        diagnostics['nonzero_log_difference_lost'] = int(((a != b) & (delta == 0)).sum())
        dz64 = delta.sum(1)
        dn64 = torch.bmm(delta.transpose(1, 2), decoded_value.double())
        dz, dn = dz64.float(), dn64.float()
        diagnostics['state_cast_underflow'] = int(((dz64 != 0) & (dz == 0)).sum() + ((dn64 != 0) & (dn == 0)).sum())
        diagnostics['state_nonfinite'] = int((~torch.isfinite(dz)).sum() + (~torch.isfinite(dn)).sum())
        if diagnostics['state_nonfinite']:
            raise PreparationError('unrepresentable_FP32_state', diagnostics)
        state = ChunkState(str(chunk_id), dz.contiguous(), dn.contiguous(), offset.contiguous())
        diagnostics['tensor_payload_bytes'] = state.tensor_payload_bytes()
        return state, diagnostics

    def _check_state(self, state, dv=None):
        if not isinstance(state, ChunkState):
            raise ValueError('ChunkState required')
        for name, x, dtype in (('dz', state.dz, torch.float32), ('dn', state.dn, torch.float32), ('offset', state.offset, torch.float64)):
            require_cpu_float(x, name)
            if x.dtype != dtype:
                raise ValueError(f'{name}: wrong persistent dtype')
        if state.dz.shape != (self.heads, self.rank) or state.offset.shape != state.dz.shape:
            raise ValueError('wrong state shape')
        if state.dn.ndim != 3 or state.dn.shape[:2] != state.dz.shape or state.dn.shape[-1] < 1 or (dv is not None and state.dn.shape[-1] != dv):
            raise ValueError('wrong value dimension')

    def append(self, state):
        self._check_state(state)
        if state.chunk_id in self.states:
            raise ValueError('duplicate chunk_id')
        if self.states and next(iter(self.states.values())).dn.shape[-1] != state.dn.shape[-1]:
            raise ValueError('value dimension changed')
        if not all(bool(torch.isfinite(x).all()) for x in (state.dz, state.dn, state.offset)):
            raise ValueError('cannot append nonfinite state')
        self.states[state.chunk_id] = state

    def evict(self, chunk_id):
        del self.states[str(chunk_id)]

    def tensor_payload_bytes(self):
        # Logical tensor payload only. No allocator/Python metadata or workspaces.
        return tensor_bytes(self.omega) + sum(s.tensor_payload_bytes() for s in self.states.values())

    @torch.no_grad()
    def read(self, query, base_output, base_logz, *, min_denominator_ratio):
        """Caller supplies the native base for exactly the currently retained KV.

        Numeric failure returns base_output exactly. min_denominator_ratio is
        an external safety parameter, not a fixed real-data success criterion.
        Underflows are counted; loss of the scaled native base triggers fallback.
        """
        for name, x in (('query', query), ('base_output', base_output), ('base_logz', base_logz)):
            require_cpu_float(x, name)
        if query.ndim != 3 or query.shape[0] != self.heads or query.shape[-1] != self.dim:
            raise ValueError('query must be [H,Q,D]')
        if base_output.ndim != 3 or base_output.shape[:2] != query.shape[:2] or base_output.shape[-1] < 1 or base_logz.shape != query.shape[:2]:
            raise ValueError('base output/logZ shapes disagree')
        if not math.isfinite(min_denominator_ratio) or min_denominator_ratio <= 0:
            raise ValueError('external min_denominator_ratio must be finite and positive')
        if not torch.isfinite(base_output).all():
            raise ValueError('no finite base_output to fall back to')
        shape = query.shape[:2]
        d = {'state_nonfinite': 0, 'query_log_nonfinite': 0, 'normalized_log_nonfinite': 0,
             'potential_unscaled_exp_overflow': 0, 'coefficient_exp_underflow': 0,
             'base_exp_underflow': 0, 'unsafe_denominator_queries': 0,
             'nonfinite_output_queries': 0, 'output_cast_nonfinite_queries': 0,
             'fallback_queries': 0, 'exact_zero_bypass': False}

        def all_fallback(reason):
            d['fallback_reason'] = reason
            d['fallback_queries'] = math.prod(shape)
            return ReadResult(base_output.clone(), torch.ones(shape, dtype=torch.bool), torch.full(shape, float('nan'), dtype=torch.float64), d)

        states = list(self.states.values())
        for s in states:
            self._check_state(s, base_output.shape[-1])
            d['state_nonfinite'] += sum(int((~torch.isfinite(x)).sum()) for x in (s.dz, s.dn, s.offset))
        if d['state_nonfinite']:
            return all_fallback('nonfinite_state')
        # No corrected path at all for no-intervention/empty state: preserves bits.
        if not states or all(torch.count_nonzero(s.dz) == 0 and torch.count_nonzero(s.dn) == 0 for s in states):
            d['exact_zero_bypass'] = True
            return ReadResult(base_output.clone(), torch.zeros(shape, dtype=torch.bool), torch.zeros(shape, dtype=torch.float64), d)
        if not torch.isfinite(query).all() or not torch.isfinite(base_logz).all():
            return all_fallback('nonfinite_query_or_base_logz')
        b = self._logs(query)
        d['query_log_nonfinite'] = int((~torch.isfinite(b)).sum())
        if d['query_log_nonfinite']:
            return all_fallback('nonfinite_query_feature_log')
        terms, masks = [], []
        for s in states:
            active = (s.dz != 0) | (s.dn != 0).any(-1)
            t = b + s.offset[:, None, :] - base_logz.double()[:, :, None]
            d['normalized_log_nonfinite'] += int((~torch.isfinite(t) & active[:, None, :]).sum())
            terms.append(t.masked_fill(~active[:, None, :], float('-inf')))
            masks.append(active[:, None, :])
        if d['normalized_log_nonfinite']:
            return all_fallback('nonfinite_normalized_log_coefficient')
        # Common scale across all chunks/features for each query; independent
        # per-chunk offsets are never silently added in incompatible coordinates.
        m = torch.stack([t.amax(-1) for t in terms]).amax(0).clamp_min(0)
        base_scale = (-m).exp()
        d['base_exp_underflow'] = int((base_scale == 0).sum())
        # Signed corrections can dwarf the native term and then cancel. Never
        # seed a naive running sum with the native value. Compensate across both
        # features and chunks (a chunk-local bmm can already lose small terms).
        den_sum = torch.zeros_like(base_scale)
        den_residual = torch.zeros_like(base_scale)
        num_sum = torch.zeros_like(base_output, dtype=torch.float64)
        num_residual = torch.zeros_like(num_sum)
        for s, t, active in zip(states, terms, masks):
            d['potential_unscaled_exp_overflow'] += int(((t > LOG_MAX64) & active).sum())
            shifted = t-m[:, :, None]
            coeff = shifted.exp()
            d['coefficient_exp_underflow'] += int(((coeff == 0) & torch.isfinite(shifted) & active).sum())
            dz, dn = s.dz.double(), s.dn.double()
            for feature in range(self.rank):
                weight = coeff[:, :, feature]
                den_sum, den_residual = _compensated_add(
                    den_sum, den_residual, weight * dz[:, None, feature])
                num_sum, num_residual = _compensated_add(
                    num_sum, num_residual,
                    weight[:, :, None] * dn[:, None, feature, :])
        den_sum, den_residual = _compensated_add(den_sum, den_residual, base_scale)
        num_sum, num_residual = _compensated_add(
            num_sum, num_residual, base_output.double() * base_scale[:, :, None])
        den = den_sum + den_residual
        num = num_sum + num_residual
        positive = den > 0
        log_ratio = torch.where(positive, m + den.clamp_min(torch.finfo(torch.float64).tiny).log(), torch.full_like(den, float('-inf')))
        # Use actual subnormal log rather than clamping positive subnormal den.
        if positive.any():
            log_ratio[positive] = m[positive] + den[positive].log()
        unsafe = ~positive | ~torch.isfinite(den) | (log_ratio < math.log(min_denominator_ratio))
        d['unsafe_denominator_queries'] = int(unsafe.sum())
        fallback = unsafe | (base_scale == 0)
        safe_den = torch.where(fallback, torch.ones_like(den), den)
        corrected = num / safe_den[:, :, None]
        nonfinite = ~torch.isfinite(corrected).all(-1)
        d['nonfinite_output_queries'] = int(nonfinite.sum())
        fallback |= nonfinite
        corrected_cast = corrected.to(base_output.dtype)
        cast_bad = ~torch.isfinite(corrected_cast).all(-1)
        d['output_cast_nonfinite_queries'] = int(cast_bad.sum())
        fallback |= cast_bad
        output = torch.where(fallback[:, :, None], base_output, corrected_cast)
        d['fallback_queries'] = int(fallback.sum())
        return ReadResult(output, fallback, log_ratio, d)
