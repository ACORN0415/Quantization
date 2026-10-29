"""Independent development codecs: six representations, real factor bit packing.
No model/query fitting. FP32 decomposition, FP16 or packed RTN factors.
Sources and fixed choices: src/experiments/CODEC_ORDER_DEVELOPMENT.md.
"""
import math
import torch
from change_mixed_core import pack_bits,unpack_bits
from tensor_adaptive_core import select
METHODS=('tarp','tsvd','pivot_qr','range','tucker','tt')

def factor_shapes(method,shape,r):
 h,t,s,d=shape;rd=min(r,d)
 if method in ('tarp','tsvd'):return [(h,t,s,r),(h,t,r,d)]
 if method in ('pivot_qr','range'):return [(h,t*s,r),(h,r,d)]
 if method=='tucker':return [(h,s,r),(h,t,t),(h,d,rd),(h,r,t,rd)]
 if method=='tt':return [(h,s,r),(h,r,t,rd),(h,rd,d)]
 raise ValueError(method)

def expected_bytes(method,shape,r,bits=None):
 total=0
 for shape_f in factor_shapes(method,shape,r):
  n=math.prod(shape_f)
  total+=2*n if bits is None else math.ceil(n*bits/8)+2*math.ceil(n/16)
 if method in ('tarp','pivot_qr'):total+=4*shape[0]*r
 return total

def choose_rank(method,shape,budget,bits=None):
 cap=min(shape[2],shape[1]*shape[3]) if method in ('tucker','tt') else min(shape[2],shape[3])
 valid=[r for r in range(1,cap+1) if expected_bytes(method,shape,r,bits)<=budget]
 if not valid:raise ValueError('budget smaller than rank1')
 return max(valid)

def pack_factor(x,bits):
 flat=x.float().contiguous().flatten();n=flat.numel();pad=(-n)%16
 padded=torch.cat([flat,flat.new_zeros(pad)]) if pad else flat
 maximum=padded.reshape(-1,16).abs().amax(1).clamp_min(2**-24).half()
 if not bool(torch.isfinite(maximum).all()):raise ValueError('factor scale overflow')
 qmax=(1<<(bits-1))-1
 scale=maximum.float().repeat_interleave(16)[:n]/qmax
 codes=(flat/scale).round().clamp(-qmax,qmax).add(qmax).to(torch.uint8)
 return dict(stream=pack_bits(codes,bits),maximum=maximum,shape=tuple(x.shape),bits=bits)

def unpack_factor(s):
 n=math.prod(s['shape']);qmax=(1<<(s['bits']-1))-1
 code=unpack_bits(s['stream'],s['bits'],n).float()-qmax
 return (code*s['maximum'].float().repeat_interleave(16)[:n]/qmax).reshape(s['shape'])

def store(raw,bits=None):
 return dict(method=raw['method'],shape=raw['shape'],rank=raw['rank'],bits=bits,
   factors=[f.half().contiguous() if bits is None else pack_factor(f,bits) for f in raw['factors']],
   indices=raw['indices'].int().contiguous() if raw.get('indices') is not None else None)

def payload_bytes(state):
 def visit(x):
  if isinstance(x,torch.Tensor):return x.untyped_storage().nbytes()
  if isinstance(x,dict):return sum(visit(v) for v in x.values())
  if isinstance(x,(list,tuple)):return sum(visit(v) for v in x)
  return 0
 return visit(state)

def cpu_state(state):
 def visit(x):
  if isinstance(x,torch.Tensor):return x.cpu()
  if isinstance(x,dict):return {k:visit(v) for k,v in x.items()}
  if isinstance(x,list):return [visit(v) for v in x]
  return x
 return visit(state)

def reconstruct(method,shape,factors):
 h,t,s,d=shape
 if method in ('tarp','tsvd'):
  a,b=factors
  x=torch.fft.irfft(torch.fft.rfft(a,dim=1)@torch.fft.rfft(b,dim=1),n=t,dim=1)
 elif method in ('pivot_qr','range'):
  x=(factors[0]@factors[1]).reshape(h,t,s,d)
 elif method=='tucker':
  a,b,c,g=factors
  y=torch.einsum('hruk,hdk->hrud',g,c)
  y=torch.einsum('hrud,htu->htrd',y,b)
  x=torch.einsum('htrd,hsr->htsd',y,a)
 elif method=='tt':
  a,b,c=factors
  y=torch.einsum('hrtk,hkd->hrtd',b,c)
  x=torch.einsum('hsr,hrtd->htsd',a,y)
 else:raise ValueError(method)
 return x.permute(1,2,0,3).reshape(1,t*s,h,d)

def decode(state):
 factors=[f.float() if state['bits'] is None else unpack_factor(f) for f in state['factors']]
 return reconstruct(state['method'],state['shape'],factors)

def decompose(x,method,r,seed=9101):
 h,t,s,d=x.shape;ids=None
 cap=min(s,t*d) if method in ('tucker','tt') else min(s,d)
 if not 1<=r<=cap:raise ValueError('rank outside supported envelope')
 rd=min(r,d)
 if method in ('tarp','tsvd'):
  a=torch.fft.rfft(x,dim=1);u,sing,vh=torch.linalg.svd(a,full_matrices=False)
  if method=='tsvd':
   f=[torch.fft.irfft(u[...,:r]*sing[...,:r].unsqueeze(-2),n=t,dim=1),torch.fft.irfft(vh[...,:r,:],n=t,dim=1)]
  else:
   ids,_=select(vh[...,:r,:].mH,r,seed)
   c=x.gather(-1,ids[:,None,None,:].expand(-1,t,s,-1)).contiguous()
   z=torch.linalg.pinv(torch.fft.rfft(c,dim=1),rtol=1e-5)@a
   f=[c,torch.fft.irfft(z,n=t,dim=1)]
 elif method in ('pivot_qr','range'):
  a=x.reshape(h,t*s,d)
  if method=='range':
   gen=torch.Generator(device=x.device).manual_seed(seed)
   omega=torch.randn(h,d,r,device=x.device,generator=gen)
   q=torch.linalg.qr(a@omega,mode='reduced').Q
   # One stabilized power iteration, no oversampling or SVD refinement.
   z=torch.linalg.qr(a.mT@q,mode='reduced').Q
   q=torch.linalg.qr(a@z,mode='reduced').Q
  else:
   residual=a.clone();qs=[];chosen=[];head=torch.arange(h,device=x.device)
   for _ in range(r):
    score=residual.square().sum(1)
    if chosen:score.scatter_(1,torch.stack(chosen,1),-1.)
    j=score.argmax(1);vec=residual[head,:,j]
    if qs:
     basis=torch.stack(qs,-1)
     for _ in range(2):vec=vec-(basis@(basis.mT@vec.unsqueeze(-1))).squeeze(-1)
    norm=vec.norm(dim=-1,keepdim=True)
    if float(norm.min())<1e-10:raise RuntimeError('pivoted QR rank deficiency')
    qnew=vec/norm;qs.append(qnew);chosen.append(j)
    residual-=qnew.unsqueeze(-1)*(qnew.unsqueeze(-2)@residual)
   q=torch.stack(qs,-1);ids=torch.stack(chosen,1)
  f=[q,q.mT@a]
 elif method=='tucker':
  # HOSVD, not iterative HOOI: separate mode bases followed by core projection.
  us=torch.linalg.svd(x.permute(0,2,1,3).reshape(h,s,t*d),full_matrices=False).U[...,:r]
  ut=torch.linalg.svd(x.reshape(h,t,s*d),full_matrices=False).U[...,:t]
  ud=torch.linalg.svd(x.permute(0,3,1,2).reshape(h,d,t*s),full_matrices=False).U[...,:rd]
  g=torch.einsum('htsd,hsr->htrd',x,us)
  g=torch.einsum('htrd,htu->hrud',g,ut)
  g=torch.einsum('hrud,hdk->hruk',g,ud)
  f=[us,ut,ud,g]
 elif method=='tt':
  u,sing,vh=torch.linalg.svd(x.permute(0,2,1,3).reshape(h,s,t*d),full_matrices=False)
  first=u[...,:r];rest=(sing[...,:r].unsqueeze(-1)*vh[...,:r,:]).reshape(h,r*t,d)
  u2,s2,v2=torch.linalg.svd(rest,full_matrices=False)
  f=[first,u2[...,:rd].reshape(h,r,t,rd),s2[...,:rd].unsqueeze(-1)*v2[...,:rd,:]]
 else:raise ValueError(method)
 return dict(method=method,shape=tuple(x.shape),rank=r,factors=[z.contiguous() for z in f],indices=ids)
