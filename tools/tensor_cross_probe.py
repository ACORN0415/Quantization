"""Development V-only common-index Fourier column approximation, NOT T-ARP."""
import csv, json, math, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import torch
import postwo_round_probe as old
from postwo_round_core import output_error_metrics
from change_mixed_core import encode_uniform
DATA=Path.home()/'sfq_data/tensor_cross'; OUT=ROOT/'src/experiments/tensor_cross_development'

def nbytes(state):
 return sum(v.untyped_storage().nbytes() for v in state.values() if isinstance(v,torch.Tensor))

def decode(s):
 c,z=s['columns'].float(),s['coeff'].float()
 if s['fourier']:
  y=torch.fft.irfft(torch.fft.rfft(c,dim=1)@torch.fft.rfft(z,dim=1),n=c.shape[1],dim=1)
 else:y=c@z
 return y.permute(1,2,0,3).reshape(1,-1,y.shape[0],y.shape[-1])

def encode(x,rank,fourier):
 # x [H,T,S,D]; real time-domain columns and coefficients form the stored state.
 a=torch.fft.rfft(x,dim=1) if fourier else x
 _,sing,vh=torch.linalg.svd(a,full_matrices=False)
 score=vh[...,:rank,:].abs().square().sum(-2)
 if fourier:
  # Full Fourier spectrum gives weights 1,2 for T=3 (DC plus conjugate pair).
  weights=torch.ones(a.shape[1],device=x.device);weights[1:]=2
  if x.shape[1]%2==0:weights[-1]=1
  indices=(score*weights[None,:,None]).sum(1).argsort(descending=True,stable=True)[...,:rank]
  selected=indices[:,None,None,:].expand(-1,x.shape[1],x.shape[2],-1)
 else:
  indices=score.argsort(descending=True,stable=True)[...,:rank]
  selected=indices[:,:,None,:].expand(-1,-1,x.shape[2],-1)
 columns=torch.gather(x,-1,selected).contiguous().half()
 # Fit to the actual stored columns, not ideal FP32 columns.
 c=torch.fft.rfft(columns.float(),dim=1) if fourier else columns.float()
 coeff=torch.linalg.pinv(c,rtol=1e-5)@a
 if fourier:coeff=torch.fft.irfft(coeff,n=x.shape[1],dim=1)
 state=dict(columns=columns,coeff=coeff.half().contiguous(),indices=indices.int().contiguous(),fourier=fourier)
 weights=torch.ones(a.shape[1],device=x.device)
 if fourier:
  weights[1:]=2
  if x.shape[1]%2==0:weights[-1]=1
 tail=(sing[...,rank:].square().sum(-1)*weights).sum()
 total=(sing.square().sum(-1)*weights).sum()
 return state,float((tail/total).sqrt())

def rank_for(budget,h,t,s,d,fourier):
 per_rank=2*h*t*(s+d)+4*h*(1 if fourier else t)
 return min(d,budget//per_rank)

def gate():
 torch.manual_seed(8021)
 x=torch.randn(2,3,17,5,device='cuda')
 for f in (False,True):
  state,bound=encode(x,5,f)
  target=x.permute(1,2,0,3).reshape(1,51,2,5)
  error=float((decode(state)-target).norm()/target.norm())
  assert error<.002 and bound==0, (f,error,bound)
  expected=2*2*3*5*(17+5)+4*2*5*(1 if f else 3)
  assert nbytes(state)==expected,(nbytes(state),expected)
 return 'full-rank real reconstruction <0.002, exact storage accounting'

def status(s):
 print(time.strftime('%F %T'),s,flush=True)
 old.save(DATA/'status.json',dict(detail=s,pid=os.getpid(),updated_utc=time.strftime('%FT%TZ',time.gmtime())))

def main():
 old.free_gpu()
 if OUT.exists():raise RuntimeError('refuse overwrite')
 DATA.mkdir(parents=True,exist_ok=True)
 torch.backends.cuda.matmul.allow_tf32=False
 sources={p:old.sha(ROOT/p) for p in ('tools/tensor_cross_probe.py','src/experiments/TENSOR_CROSS_DEVELOPMENT.md','tools/change_mixed_core.py','tools/postwo_round_core.py','tools/postwo_round_probe.py')}
 gates=[gate()]; rows=[]; inputs=[]; artifacts=[]; start=time.monotonic()
 previous=json.loads((ROOT/'src/experiments/postwo_rounding_development/decision.json').read_text())
 for prompt in (0,4):
  captured=next(c for c in previous['captures'] if c['prompt']==prompt)
  for a in captured['artifacts']:
   path=Path.home()/'sfq_data/postwo_rounding'/a['file']
   assert old.sha(path)==a['sha256'];inputs.append(a)
  for layer in (0,14,29):
   status(f'p{prompt} layer{layer}: beginning')
   base=Path.home()/f'sfq_data/postwo_rounding/p{prompt:02d}_s101'
   p=torch.load(base/f'l{layer:02d}_c6.pt',map_location='cuda',weights_only=True)
   q=torch.load(base/f'l{layer:02d}_c7.pt',map_location='cuda',weights_only=True)['query']
   v=p['new_value']; assert v.shape==(1,4680,12,128)
   x=v[0].reshape(3,1560,12,128).permute(2,0,1,3).float().contiguous()
   a0,a1=old.probabilities(q,p['key'])
   for bits in (2,3,4):
    torch.cuda.synchronize();t=time.perf_counter();baseline=encode_uniform(v,bits=bits,group=16);torch.cuda.synchronize();base_s=time.perf_counter()-t
    budget=baseline.bytes()['resident']
    for arm in ('rtn','independent','common_fourier'):
     if arm=='rtn':
      state=None;rank=0;bound=None;enc=base_s;size=budget
     else:
      f=arm=='common_fourier';rank=rank_for(budget,12,3,1560,128,f)
      torch.cuda.synchronize();t=time.perf_counter();state,bound=encode(x,rank,f);torch.cuda.synchronize();enc=time.perf_counter()-t;size=nbytes(state)
      assert size<=budget
     torch.cuda.synchronize();t=time.perf_counter();restored=baseline.restore() if state is None else decode(state);torch.cuda.synchronize();dec=time.perf_counter()-t
     m=output_error_metrics(a0,a1,restored[0],v[0].float(),p['wo'].float())
     rel=float((restored-v.float()).norm()/v.float().norm())
     row=dict(prompt=prompt,layer=layer,bits_budget=bits,arm=arm,rank=rank,payload_bytes=size,budget_bytes=budget,v_rel=rel,svd_rank_rel=bound,D=m['mean_error_sq']/(a0.shape[1]*1536),encode_seconds=enc,decode_seconds=dec)
     assert all(math.isfinite(row[k]) for k in ('v_rel','D','encode_seconds','decode_seconds'))
     if state is not None:
      path=DATA/f'p{prompt:02d}_l{layer:02d}_b{bits}_{arm}.pt'
      torch.save({k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in state.items()},path)
      # Independent CPU restore of the serialized payload.
      loaded=torch.load(path,weights_only=True,map_location='cpu')
      cpu=decode(loaded); delta=float((cpu-restored.cpu()).norm()/restored.cpu().norm())
      assert delta<2e-5,delta
      artifacts.append(dict(file=path.name,sha256=old.sha(path),cpu_gpu_rel=delta))
     rows.append(row)
     status(f'{len(rows)}/54 p{prompt} layer{layer} bits{bits} {arm}: D={row["D"]:.7g}, Vrel={rel:.4g}, bytes={size}')
     del restored,state
    del baseline
   del p,q,v,x,a0,a1
   torch.cuda.empty_cache()
 assert len(rows)==54
 assert sources=={p:old.sha(ROOT/p) for p in sources}
 OUT.mkdir(parents=True)
 with (OUT/'rows.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 summary=[]
 for bits in (2,3,4):
  for prompt in (0,4):
   means={a:sum(r['D'] for r in rows if r['prompt']==prompt and r['bits_budget']==bits and r['arm']==a)/3 for a in ('rtn','independent','common_fourier')}
   summary.append(dict(bits=bits,prompt=prompt,**means,common_over_rtn=means['common_fourier']/means['rtn'],common_over_independent=means['common_fourier']/means['independent']))
 decision=any(all(s['common_over_rtn']<=.9 for s in summary if s['bits']==b) for b in (2,3,4))
 old.save(OUT/'evidence.json',dict(source_sha256=sources,input_artifacts=inputs,payload_artifacts=artifacts,gates=gates,summary=summary,screen_pass=decision,seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),rows_sha256=old.sha(OUT/'rows.csv')))
 status(f'COMPLETE screening pass={decision}; see evidence.json')

if __name__=='__main__':main()
