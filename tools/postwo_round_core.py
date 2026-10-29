"""Development-only new-chunk V rounding with frozen post-projection attention.

A0/A1 [heads,queries,new_tokens] are the new-token columns of attention over
complete BF16 K caches. Only new V [tokens,heads,channels] errors are modeled.
There is no old-V source buffer, no original old-error offset, and no generator
integration here. Wo [output,heads*channels] mixes head contributions. FP32
objectives are offline diagnostic surrogates, not BF16 FlashAttention outputs.
Reuse fixed FP16 scales and UINT8 containers from transition_round_core. Packed
byte estimates are hypothetical. Future-query fitting/evaluation is caller-owned.
"""
import math

import torch

from transition_round_core import ValueCode, make_rtn, _weights


def _validate(a0, a1, values, wo, transition_weight, head_local):
    _weights(a0, a1, values)
    if (wo.ndim != 2 or wo.shape[0] == 0 or wo.shape[1] != values.shape[1]*values.shape[2]
            or wo.device != values.device or not wo.is_floating_point()
            or not bool(torch.isfinite(wo).all())):
        raise ValueError('expected finite same-device Wo [output,heads*channels]')
    if (type(transition_weight) not in (int,float) or not math.isfinite(transition_weight)
            or transition_weight < 0):
        raise ValueError('transition_weight must be finite nonnegative')
    if type(head_local) is not bool:
        raise ValueError('head_local must be bool')


def _project(errors, wo, head_local):
    """errors [H,Q,D], returns [Q,O] or [H,Q,O] without cross-head terms."""
    h, q, d = errors.shape
    if head_local:
        blocks = wo.reshape(wo.shape[0],h,d).permute(1,2,0)
        return errors@blocks
    return errors.permute(1,0,2).reshape(q,h*d)@wo.transpose(0,1)


def _objective(e0, e1, wo, transition_weight, head_local):
    p0 = _project(e0,wo,head_local)
    p1 = _project(e1,wo,head_local)
    mean = .5*(p0.square().sum()+p1.square().sum())
    jump = (p1-p0).square().sum()
    return mean+transition_weight*jump, mean, jump


def _summary(e0,e1,wo,transition_weight,head_local):
    loss, mean, jump = _objective(e0,e1,wo,transition_weight,head_local)
    # Head-local errors sum heads; normalize by queries * physical output width,
    # so local/full use the same denominator and differ only by cross terms.
    count=e0.shape[1]*wo.shape[0]
    result=dict(objective=float(loss),mean_error_sq=float(mean),transition_error_sq=float(jump),
                mean_output_error_rms=float((mean/count).sqrt()),
                transition_error_rms=float((jump/count).sqrt()))
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('nonfinite post-projection objective')
    return result


def output_error_metrics(a0,a1,restored,values,wo,transition_weight=0.,head_local=False):
    _validate(a0,a1,values,wo,transition_weight,head_local)
    if restored.shape!=values.shape or restored.device!=values.device:
        raise ValueError('restored values shape/device mismatch')
    error=(restored.float()-values.float()).permute(1,0,2)
    return _summary(a0.float()@error,a1.float()@error,wo.float(),transition_weight,head_local)


def _gram_max(a):
    return torch.linalg.eigvalsh(a@a.transpose(-1,-2)).amax().clamp_min(0.)


def _pullback(projected,wo,heads,dim,head_local):
    if head_local:
        blocks=wo.reshape(wo.shape[0],heads,dim).permute(1,0,2)
        return projected@blocks
    return (projected@wo).reshape(projected.shape[0],heads,dim).permute(1,0,2)


def optimize_rounding(a0,a1,values,wo,*,transition_weight=0.,head_local=False,
                      steps=32,bits=4,group=16,initial=None,wo_norm_sq=None):
    """Fixed-budget projected-gradient hard-code candidates, not global optimum.

    The RTN candidate and each iteration's floor/ceil projection are evaluated.
    Initial scales/padding/container shape remain bit-identical. A safe global
    Hessian norm bound couples all heads; exact small attention Gram spectra and
    supplied/exact Wo spectral norm avoid a huge Frobenius relaxation. A supplied
    norm is a caller assertion and must be an upper bound on ||Wo||_2^2.
    """
    _validate(a0,a1,values,wo,transition_weight,head_local)
    if type(steps) is not int or steps<0:
        raise ValueError('steps must be nonnegative integer')
    state=make_rtn(values,bits,group) if initial is None else initial
    if (state.shape!=tuple(values.shape) or state.bits!=bits or state.group!=group
            or state.codes.device!=values.device):
        raise ValueError('initial code mismatch')
    a0,a1,wo,x=a0.float(),a1.float(),wo.float(),values.float()
    pd=a1-a0
    if wo_norm_sq is None:
        norm_sq=torch.linalg.svdvals(wo).amax().square()
    else:
        norm_sq=torch.as_tensor(wo_norm_sq,device=x.device,dtype=torch.float32)
        if norm_sq.ndim!=0 or not bool(torch.isfinite(norm_sq)) or float(norm_sq)<0:
            raise ValueError('Wo squared norm must be finite nonnegative scalar')
    lipschitz=norm_sq*(_gram_max(a0)+_gram_max(a1)+2*transition_weight*_gram_max(pd))
    step=1./(lipschitz*1.001).clamp_min(1e-12)
    n,h,d=x.shape;qmax=(1<<(bits-1))-1
    groups=torch.arange(n,device=x.device)//group
    scales=state.scale[:,0].index_select(0,groups).permute(1,0,2).float()
    original=x.permute(1,0,2)
    normalized=original/scales
    low_code=normalized.floor().clamp(-qmax,qmax)
    high_code=normalized.ceil().clamp(-qmax,qmax)
    low,high=low_code*scales,high_code*scales
    initial_values=state.restore().permute(1,0,2)
    z=initial_values.clone()
    initial_errors=(a0@(initial_values-original),a1@(initial_values-original))
    initial_metrics=_summary(*initial_errors,wo,transition_weight,head_local)
    best=state.codes.clone();best_loss=initial_metrics['objective']
    history=[best_loss];accepted=0
    for _ in range(steps):
        error=z-original;e0=a0@error;e1=a1@error
        p0,p1=_project(e0,wo,head_local),_project(e1,wo,head_local)
        # dL/de0 = Wo^T Wo[e0+2lambda(e0-e1)], analogously for e1.
        de0=_pullback(p0+2*transition_weight*(p0-p1),wo,h,d,head_local)
        de1=_pullback(p1+2*transition_weight*(p1-p0),wo,h,d,head_local)
        gradient=a0.transpose(-1,-2)@de0+a1.transpose(-1,-2)@de1
        z=torch.maximum(low,torch.minimum(high,z-step*gradient))
        hard_code=torch.where(z-low<=high-z,low_code,high_code)
        hard_error=hard_code*scales-original
        metrics=_summary(a0@hard_error,a1@hard_error,wo,transition_weight,head_local)
        candidate=metrics['objective'];history.append(candidate)
        if candidate<best_loss:
            best_loss=candidate;accepted+=1
            best[:n]=hard_code.permute(1,0,2).add(qmax).to(torch.uint8)
    result=ValueCode(best,state.scale,state.shape,state.bits,state.group)
    final=output_error_metrics(a0,a1,result.restore(),values,wo,transition_weight,head_local)
    if final['objective']>initial_metrics['objective']:
        result=ValueCode(state.codes.clone(),state.scale,state.shape,state.bits,state.group)
        final=initial_metrics
    return result,dict(steps=steps,candidates_evaluated=steps+1,transition_weight=float(transition_weight),
        head_local=head_local,initial=initial_metrics,final=final,training_objective_history=history,
        accepted_candidate_updates=accepted,changed_codes=int((result.codes[:n]!=state.codes[:n]).sum()),
        wo_norm_sq=float(norm_sq),hessian_bound=float(lipschitz),step_size=float(step),
        step_rule='global inverse conservative post-projection spectral Hessian bound',
        scope='new-V contribution only, no old-V source error offset')
