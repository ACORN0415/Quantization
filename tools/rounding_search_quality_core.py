"""Isolated diagnostics for candidate-pool fairness and a small exact binary block.

The trace deliberately reproduces postwo_round_core.optimize_rounding's FP32
operation order while exposing every hard candidate to a callback. The frozen
optimizer is not modified. The enumerator is exact only for its supplied finite
binary quadratic subproblem; it is not a global bound for all cache codes.
"""
import torch

from transition_round_core import ValueCode, make_rtn
from postwo_round_core import (
    _validate, _summary, _gram_max, _project, _pullback, output_error_metrics,
)


def trace_rounding(a0,a1,values,wo,*,callback=None,transition_weight=0.,head_local=False,
                   steps=32,bits=4,group=16,initial=None,wo_norm_sq=None):
    """Replay frozen optimizer; callback(iteration, fresh_candidate) sees 0..steps.

    Iteration 0 is RTN, and 1..steps are the hard candidates in original order.
    Candidate codes are fresh clones so callers may retain them. The scale tensor
    is immutable and shared; callbacks must not modify it or input tensors.
    Selection must use calibration queries only; the callback does not decide
    which queries or metrics the caller observes.
    """
    _validate(a0,a1,values,wo,transition_weight,head_local)
    if callback is not None and not callable(callback):
        raise ValueError('callback must be callable or None')
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
    if callback is not None:
        callback(0,ValueCode(state.codes.clone(),state.scale,state.shape,state.bits,state.group))
    for iteration in range(steps):
        error=z-original;e0=a0@error;e1=a1@error
        p0,p1=_project(e0,wo,head_local),_project(e1,wo,head_local)
        de0=_pullback(p0+2*transition_weight*(p0-p1),wo,h,d,head_local)
        de1=_pullback(p1+2*transition_weight*(p1-p0),wo,h,d,head_local)
        gradient=a0.transpose(-1,-2)@de0+a1.transpose(-1,-2)@de1
        z=torch.maximum(low,torch.minimum(high,z-step*gradient))
        hard_code=torch.where(z-low<=high-z,low_code,high_code)
        hard_error=hard_code*scales-original
        metrics=_summary(a0@hard_error,a1@hard_error,wo,transition_weight,head_local)
        candidate=metrics['objective'];history.append(candidate)
        if callback is not None:
            candidate_codes=state.codes.clone()
            candidate_codes[:n]=hard_code.permute(1,0,2).add(qmax).to(torch.uint8)
            callback(iteration+1,ValueCode(candidate_codes,state.scale,state.shape,state.bits,state.group))
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
        scope='new-V contribution only, no old-V source error offset',
        callback_candidates=steps+1 if callback is not None else 0)


def enumerate_binary_quadratic(linear_d,quad_d,const_d,linear_j,quad_j,const_j,*,batch_size=1024):
    """Enumerate all K-bit points for D=c+l.x+x^T Q x and analogous J.

    All coefficient tensors must be finite floating tensors on the same device
    and with the same dtype. K is limited to 20 to keep explicit enumeration
    bounded. Row index i encodes x_j=(i>>j)&1 (lowest bit is coordinate 0).
    Ties use the first index, matching torch.argmin. Constants may be scalars or
    same-dtype/device scalar tensors. Return arrays and minimizers for D and D+J.
    """
    vectors=(linear_d,linear_j)
    matrices=(quad_d,quad_j)
    if any(not isinstance(v,torch.Tensor) for v in vectors+matrices):
        raise ValueError('quadratic coefficients must be tensors')
    if linear_d.ndim!=1 or not 1<=linear_d.numel()<=20:
        raise ValueError('expected 1..20 binary coordinates')
    k=linear_d.numel();device=linear_d.device;dtype=linear_d.dtype
    if dtype not in (torch.float32,torch.float64):
        raise ValueError('use float32 or float64 coefficients')
    for v in vectors+matrices:
        if v.device!=device or v.dtype!=dtype or not bool(torch.isfinite(v).all()):
            raise ValueError('finite coefficients must share device and dtype')
    if linear_j.shape!=(k,) or any(q.shape!=(k,k) for q in matrices):
        raise ValueError('quadratic coefficient shapes differ')
    if type(batch_size) is not int or batch_size<=0:
        raise ValueError('batch_size must be positive integer')
    constants=[]
    for value in (const_d,const_j):
        c=torch.as_tensor(value,device=device,dtype=dtype)
        if c.ndim!=0 or not bool(torch.isfinite(c)):
            raise ValueError('expected finite scalar constants')
        constants.append(c)
    count=1<<k;columns=torch.arange(k,device=device,dtype=torch.int64)
    d_values=torch.empty(count,device=device,dtype=dtype)
    j_values=torch.empty_like(d_values)
    for lo in range(0,count,batch_size):
        hi=min(count,lo+batch_size)
        indices=torch.arange(lo,hi,device=device,dtype=torch.int64)
        bits=((indices[:,None]>>columns[None])&1).to(dtype)
        d_values[lo:hi]=constants[0]+bits@linear_d+((bits@quad_d)*bits).sum(1)
        j_values[lo:hi]=constants[1]+bits@linear_j+((bits@quad_j)*bits).sum(1)
    if not bool(torch.isfinite(d_values).all() and torch.isfinite(j_values).all()):
        raise ValueError('nonfinite enumerated objective')
    best_mean_index=int(d_values.argmin())
    best_transition_index=int((d_values+j_values).argmin())
    def bits_at(index):
        return ((torch.tensor(index,device=device,dtype=torch.int64)>>columns)&1).to(torch.uint8)
    return dict(d_values=d_values,j_values=j_values,count=count,coordinates=k,
        best_mean_index=best_mean_index,best_transition_index=best_transition_index,
        best_mean_bits=bits_at(best_mean_index),best_transition_bits=bits_at(best_transition_index))
