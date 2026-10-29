"""Development-only fixed-scale value rounding under two frozen attention states.

Weights are supplied attention probabilities [heads, queries, tokens]. Keys and
queries are never optimized here. Values are [tokens, heads, channels]; caller
chooses fitting/evaluation queries and an immutable prefix. This is not a deployed
KV encoder or an implementation of OptR. This optimizer receives the entire
original V tensor (including old tokens): full-prefix error objectives are
offline/oracle diagnostics if an encoder has already discarded old BF16 V.
No actual model experiment was run with this candidate; see
src/experiments/TRANSITION_EQUIVALENCE_GATE_KO.md. Codes occupy uint8; packed bytes are a
hypothetical estimate. Scales persist as FP16 and all objective arithmetic is FP32.
"""
from dataclasses import dataclass
import math

import torch


MIN_SCALE = 2.0**-24


def _values(values):
    if (values.ndim != 3 or any(n <= 0 for n in values.shape)
            or not values.is_floating_point() or not bool(torch.isfinite(values).all())):
        raise ValueError('expected finite nonempty [tokens,heads,channels] values')


def _weights(a0, a1, values):
    _values(values)
    if (a0.ndim != 3 or a0.shape != a1.shape or a0.shape[0] != values.shape[1]
            or a0.shape[-1] != values.shape[0] or a0.shape[1] <= 0
            or a0.device != values.device or a1.device != values.device
            or not a0.is_floating_point() or not a1.is_floating_point()
            or not bool(torch.isfinite(a0).all()) or not bool(torch.isfinite(a1).all())):
        raise ValueError('expected same-device finite paired [heads,queries,tokens] weights')


@dataclass
class ValueCode:
    codes: torch.Tensor
    scale: torch.Tensor
    shape: tuple
    bits: int
    group: int

    def restore(self):
        qmax = (1 << (self.bits-1))-1
        restored = (self.codes.reshape(-1, self.group, *self.shape[1:]).float()-qmax)*self.scale.float()
        return restored.reshape(-1, *self.shape[1:])[:self.shape[0]]

    def bytes(self):
        code_bytes = self.codes.numel()*self.codes.element_size()
        metadata = self.scale.numel()*self.scale.element_size()
        return dict(resident=code_bytes+metadata,
                    packed_estimate=math.ceil(self.codes.numel()*self.bits/8)+metadata,
                    code_container_bytes=code_bytes, scale_bytes=metadata)


def make_rtn(values, bits=4, group=16):
    """Symmetric RTN using the same persisted FP16 scale used by all candidates."""
    _values(values)
    if type(bits) is not int or bits not in (2,4,8): raise ValueError('bits must be 2,4,8')
    if type(group) is not int or group <= 0: raise ValueError('group must be positive integer')
    shape = tuple(values.shape)
    pad = (-shape[0]) % group
    data = values.float()
    if pad: data = torch.cat((data, data.new_zeros(pad, *shape[1:])),0)
    data = data.reshape(-1,group,*shape[1:])
    qmax = (1 << (bits-1))-1
    scale = (data.abs().amax(1,keepdim=True)/qmax).clamp_min(MIN_SCALE).to(torch.float16)
    if not bool(torch.isfinite(scale).all()): raise ValueError('FP16 scale overflow')
    codes = (data/scale.float()).round().clamp(-qmax,qmax).add(qmax).to(torch.uint8)
    return ValueCode(codes.reshape(-1,*shape[1:]),scale,shape,bits,group)


def _errors(a0,a1,restored,values):
    error = (restored.float()-values.float()).permute(1,0,2)
    e0 = a0.float()@error
    e1 = a1.float()@error
    return e0,e1


def _summary(e0,e1,transition_weight):
    mean = (e0.square().sum()+e1.square().sum())*.5
    transition = (e1-e0).square().sum()
    objective = mean+transition_weight*transition
    result = dict(mean_error_sq=float(mean),transition_error_sq=float(transition),objective=float(objective),
                  mean_output_error_rms=float((mean/e0.numel()).sqrt()),
                  transition_error_rms=float((transition/e0.numel()).sqrt()))
    if not all(math.isfinite(v) for v in result.values()): raise ValueError('nonfinite output objective')
    return result


def output_error_metrics(a0,a1,restored,values,transition_weight=0.):
    _weights(a0,a1,values)
    if restored.shape != values.shape or restored.device != values.device: raise ValueError('restored value mismatch')
    if not isinstance(transition_weight,(int,float)) or not math.isfinite(transition_weight) or transition_weight < 0:
        raise ValueError('transition_weight must be finite nonnegative')
    return _summary(*_errors(a0,a1,restored,values),transition_weight)


def optimize_rounding(a0,a1,values,*,suffix_tokens,bits=4,group=16,transition_weight=0.,steps=32,initial=None):
    """Equal-budget projected-gradient candidate search, not an exact code optimum.

    Only the suffix may change. Each scalar chooses floor/ceil of value divided
    by its immutable persisted scale, clipped to symmetric code bounds. The same
    RTN initialization and 32 hard candidates are available to either objective.
    Continuous suffix values use a conservative exact small-Gram spectral step;
    the best *hard* training objective seen, including initial RTN, is returned.
    """
    _weights(a0,a1,values)
    if type(suffix_tokens) is not int or not 0 < suffix_tokens <= values.shape[0]:
        raise ValueError('invalid mutable suffix size')
    if type(steps) is not int or steps < 0: raise ValueError('steps must be nonnegative integer')
    if not isinstance(transition_weight,(int,float)) or not math.isfinite(transition_weight) or transition_weight < 0:
        raise ValueError('transition_weight must be finite nonnegative')
    state = make_rtn(values,bits,group) if initial is None else initial
    if (state.shape != tuple(values.shape) or state.bits != bits or state.group != group
            or state.codes.device != values.device): raise ValueError('initial code mismatch')
    n = values.shape[0]; start = n-suffix_tokens; qmax = (1 << (bits-1))-1
    a0 = a0.float(); a1 = a1.float(); x = values.float()
    restored = state.restore()
    e0_base,e1_base = _errors(a0,a1,restored,x)
    initial_metrics = _summary(e0_base,e1_base,transition_weight)
    # Scale expanded only for mutable tokens; prefix codes remain bit-identical.
    token_groups = torch.arange(start,n,device=x.device)//group
    scale = state.scale[:,0].index_select(0,token_groups).float().permute(1,0,2)
    original = x[start:].permute(1,0,2)
    normalized = original/scale
    low_code = normalized.floor().clamp(-qmax,qmax)
    high_code = normalized.ceil().clamp(-qmax,qmax)
    low,high = low_code*scale,high_code*scale
    base = restored[start:].permute(1,0,2)
    z = base.clone()
    p0,p1 = a0[:,:,start:],a1[:,:,start:]
    pd = p1-p0
    # Hessian = 2 B^T B, but queries are much fewer than mutable tokens.
    b = torch.cat((p0*math.sqrt(.5),p1*math.sqrt(.5),pd*math.sqrt(transition_weight)),1)
    gram = b@b.transpose(-1,-2)
    spectral = torch.linalg.eigvalsh(gram).amax(-1).clamp_min(0.)
    step_size = (2*spectral*1.001).clamp_min(1e-12).reciprocal()[:,None,None]
    best_loss = initial_metrics['objective']; best = state.codes.clone()
    history = [best_loss]; accepted = 0
    for index in range(steps):
        delta = z-base
        e0,e1 = e0_base+p0@delta,e1_base+p1@delta
        grad = p0.transpose(-1,-2)@e0+p1.transpose(-1,-2)@e1
        if transition_weight:
            grad = grad+2*transition_weight*(pd.transpose(-1,-2)@(e1-e0))
        z = torch.maximum(low,torch.minimum(high,z-step_size*grad))
        hard_code = torch.where((z-low) <= (high-z),low_code,high_code)
        hard = hard_code*scale
        hd = hard-base
        metrics = _summary(e0_base+p0@hd,e1_base+p1@hd,transition_weight)
        candidate_loss = metrics['objective']
        history.append(candidate_loss)
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best[start:n] = hard_code.permute(1,0,2).add(qmax).to(torch.uint8)
            accepted += 1
    answer = ValueCode(best,state.scale,state.shape,state.bits,state.group)
    final_metrics = output_error_metrics(a0,a1,answer.restore(),values,transition_weight)
    # Actual reconstructed objective may have harmless FP32 accumulation drift
    # relative to the cached-error search. Fall back to RTN if it is worse.
    if final_metrics['objective'] > initial_metrics['objective']:
        answer = ValueCode(state.codes.clone(),state.scale,state.shape,state.bits,state.group)
        final_metrics = initial_metrics
    if not torch.equal(answer.codes[:start],state.codes[:start]): raise RuntimeError('immutable prefix changed')
    return answer,dict(steps=steps,candidates_evaluated=steps+1,transition_weight=float(transition_weight),
        suffix_tokens=suffix_tokens,initial=initial_metrics,final=final_metrics,
        training_objective_history=history,accepted_candidate_updates=accepted,
        changed_suffix_codes=int((answer.codes[start:n] != state.codes[start:n]).sum()),
        immutable_prefix_tokens=start,step_rule='per-head inverse conservative spectral Hessian bound')
