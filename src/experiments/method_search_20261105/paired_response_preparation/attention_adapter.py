"""Pure CPU FP32 attention/LSE and post-Wo adapters; no legacy helper imports.

Dense visible-window arithmetic matches old.attn: q-block 32, bmm, softmax,
bmm. Inputs are already selected queries and original post-RoPE coordinates.
No mask, query selection, model/capture loading, GPU or experiment guard lives here.
"""
from dataclasses import dataclass
import math
import torch


def _tensor(x,name,ndim):
    if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float32 or x.ndim!=ndim:
        raise ValueError(f'{name}: CPU FP32 tensor with ndim={ndim} required')
    if not torch.isfinite(x).all():
        raise ValueError(f'{name}: nonfinite input')


@dataclass
class AttentionResult:
    original_output: torch.Tensor   # [Q,H,Dv], old.attn order
    paired_output: torch.Tensor     # [H,Q,Dv], paired.read order
    logz: torch.Tensor              # [H,Q], FP32, exactly the same logits
    past_tokens: int
    current_tokens: int


@torch.no_grad()
def paired_to_original(output):
    _tensor(output,'paired_output',3)
    if min(output.shape)<1:
        raise ValueError('nonempty paired [H,Q,Dv] required')
    return output.permute(1,0,2).contiguous()


@torch.no_grad()
def attention_window(query,past_key,past_value,current_key,current_value):
    """All supplied past AND current tokens are visible; no implicit causal mask.

    Q [Q,H,D], K [N,H,D], V [N,H,Dv]. Every input must already be FP32;
    any BF16-to-FP32 conversion is explicit at the caller, as for old.attn.
    """
    for name,x in (('query',query),('past_key',past_key),('past_value',past_value),('current_key',current_key),('current_value',current_value)):
        _tensor(x,name,3)
    nq,heads,dim=query.shape
    if min(nq,heads,dim)<1 or current_key.shape[0]<1:
        raise ValueError('nonempty queries, heads, dimensions and current chunk required')
    if past_key.shape[1:]!=(heads,dim) or current_key.shape[1:]!=(heads,dim):
        raise ValueError('key coordinate shape mismatch')
    dv=current_value.shape[-1]
    if dv<1 or past_value.shape!=(past_key.shape[0],heads,dv) or current_value.shape!=(current_key.shape[0],heads,dv):
        raise ValueError('value token/head/value-dimension mismatch')
    key=torch.cat((past_key,current_key),0)
    value=torch.cat((past_value,current_value),0)
    kt,vt=key.permute(1,2,0).contiguous(),value.permute(1,0,2).contiguous()
    out=torch.empty_like(query) if dv==dim else torch.empty((nq,heads,dv),dtype=torch.float32)
    logz=torch.empty((heads,nq),dtype=torch.float32)
    for begin in range(0,nq,32):
        logits=torch.bmm(query[begin:begin+32].permute(1,0,2),kt)/math.sqrt(dim)
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite FP32 attention logits')
        out[begin:begin+32]=torch.bmm(logits.softmax(-1),vt).permute(1,0,2)
        logz[:,begin:begin+32]=torch.logsumexp(logits,dim=-1)
    if not torch.isfinite(out).all() or not torch.isfinite(logz).all():
        raise ValueError('nonfinite FP32 attention result')
    return AttentionResult(out,out.permute(1,0,2).contiguous(),logz,
                           past_key.shape[0],current_key.shape[0])


@torch.no_grad()
def post_wo_difference(candidate,reference,weight,bias):
    """FP32 delta before W_o; bias cancels from delta but not reference energy.

    Candidate/reference [Q,H,Dv], weight [Dout,H*Dv], bias [Dout]. SSE and
    reference energy are FP64 reductions; zero reference energy returns NSE=None.
    This reference is caller-specified (K-only or full BF16), never relabeled.
    """
    _tensor(candidate,'candidate',3);_tensor(reference,'reference',3)
    _tensor(weight,'weight',2);_tensor(bias,'bias',1)
    if candidate.shape!=reference.shape or min(reference.shape)<1:
        raise ValueError('candidate/reference shape mismatch')
    nq,heads,dv=reference.shape
    if weight.shape[0]<1 or weight.shape[1]!=heads*dv or bias.shape!=(weight.shape[0],):
        raise ValueError('W_o/bias shape mismatch')
    delta=(candidate-reference).reshape(nq,-1) @ weight.t()
    reference_y=reference.reshape(nq,-1) @ weight.t()+bias
    if not torch.isfinite(delta).all() or not torch.isfinite(reference_y).all():
        raise ValueError('nonfinite FP32 projected output')
    sse=float(delta.double().square().sum())
    energy=float(reference_y.double().square().sum())
    return {'delta':delta,'sse':sse,'reference_energy':energy,'nse':sse/energy if energy>0 else None}
