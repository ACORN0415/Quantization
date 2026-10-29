"""Frozen-input 2-bit-budget adaptive column selection development measurement."""
import csv,json,time,statistics,math
from pathlib import Path
import torch
import tensor_cross_probe as base
from tensor_adaptive_core import encode
from postwo_round_core import output_error_metrics
ROOT=base.ROOT;OUT=ROOT/'src/experiments/tensor_adaptive_development';DATA=Path.home()/'sfq_data/tensor_adaptive'
ARMS=['rtn','static','svd','greedy']+[f'arp{s}' for s in range(9101,9106)]

def timed(fn):
 fn();torch.cuda.synchronize();durations=[]
 for _ in range(3):
  t=time.perf_counter();result=fn();torch.cuda.synchronize();durations.append(time.perf_counter()-t)
 return result,statistics.median(durations)

def status(message):
 print(time.strftime('%F %T'),message,flush=True)
 base.old.save(DATA/'status.json',dict(detail=message,updated_utc=time.strftime('%FT%TZ',time.gmtime())))

def main():
 base.old.free_gpu()
 if OUT.exists():raise RuntimeError('refuse overwrite')
 DATA.mkdir(parents=True,exist_ok=True)
 torch.backends.cuda.matmul.allow_tf32=False
 paths=['tools/tensor_adaptive_core.py','tools/tensor_adaptive_probe.py','tools/test_tensor_adaptive.py','src/experiments/TENSOR_ADAPTIVE_DEVELOPMENT.md','tools/tensor_cross_probe.py','tools/change_mixed_core.py','tools/postwo_round_core.py','tools/postwo_round_probe.py']
 hashes={p:base.old.sha(ROOT/p) for p in paths}
 ev=json.loads((base.OUT/'evidence.json').read_text())
 for p,sha in ev['source_sha256'].items():assert base.old.sha(ROOT/p)==sha
 for a in ev['input_artifacts']:assert base.old.sha(Path.home()/'sfq_data/postwo_rounding'/a['file'])==a['sha256']
 previous=list(csv.DictReader((base.OUT/'rows.csv').open()))+list(csv.DictReader((ROOT/'src/experiments/tensor_cross_repair/rows.csv').open()))
 for folder in (base.OUT,ROOT/'src/experiments/tensor_cross_repair'):
  assert base.old.sha(folder/'rows.csv')==json.loads((folder/'evidence.json').read_text())['rows_sha256']
 rows=[];artifacts=[];start=time.monotonic();max_repro=0.
 for prompt in (0,4):
  for layer in (0,14,29):
   folder=Path.home()/f'sfq_data/postwo_rounding/p{prompt:02d}_s101'
   p=torch.load(folder/f'l{layer:02d}_c6.pt',map_location='cuda',weights_only=True)
   q=torch.load(folder/f'l{layer:02d}_c7.pt',map_location='cuda',weights_only=True)['query']
   v=p['new_value'];x=v[0].reshape(3,1560,12,128).permute(2,0,1,3).float().contiguous()
   a0,a1=base.old.probabilities(q,p['key'])
   budget=base.encode_uniform(v,bits=2,group=16).bytes()['resident'];rank=base.rank_for(budget,12,3,1560,128,True)
   assert rank==22
   for arm in ARMS:
    def make():
     if arm=='rtn':return base.encode_uniform(v,bits=2,group=16),{}
     return encode(x,rank,seed=int(arm[3:]) if arm.startswith('arp') else 9101,greedy=arm=='greedy',kind=arm if arm in ('static','svd') else 'adaptive')
    (state,diagnostics),enc=timed(make)
    y,dec=timed(state.restore if arm=='rtn' else lambda:base.decode(state))
    size=state.bytes()['resident'] if arm=='rtn' else base.nbytes(state)
    assert size<=budget
    metrics=output_error_metrics(a0,a1,y[0],v[0].float(),p['wo'].float())
    D=metrics['mean_error_sq']/(a0.shape[1]*1536)
    if arm in ('rtn','static','svd'):
     oldarm={'rtn':'rtn','static':'common_fourier','svd':'svd_fp16'}[arm]
     old=next(r for r in previous if int(r['prompt'])==prompt and int(r['layer'])==layer and r['bits_budget']=='2' and r['arm']==oldarm)
     delta=abs(float(old['D'])-D);max_repro=max(max_repro,delta);assert delta<1e-7
    cpu_rel=0.
    if arm!='rtn':
     path=DATA/f'p{prompt}_l{layer}_{arm}.pt'
     torch.save({k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in state.items()},path)
     cpu=base.decode(torch.load(path,weights_only=True,map_location='cpu'))
     cpu_rel=float((cpu-y.cpu()).norm()/y.cpu().norm());assert cpu_rel<2e-5
     artifacts.append(dict(file=path.name,sha256=base.old.sha(path)))
    row=dict(prompt=prompt,layer=layer,arm=arm,D=D,v_rel=float((y-v.float()).norm()/v.float().norm()),rank=0 if arm=='rtn' else rank,payload_bytes=size,budget_bytes=budget,encode_seconds=enc,decode_seconds=dec,cpu_gpu_rel=cpu_rel,min_frequency_pivot_norm_sq=diagnostics.get('min_frequency_pivot_norm_sq',''),max_selected_residual=diagnostics.get('max_selected_residual',''))
    assert all(math.isfinite(row[k]) for k in ('D','v_rel','encode_seconds','decode_seconds'))
    rows.append(row);status(f'{len(rows)}/54 p{prompt} l{layer} {arm} D={D:.8g}')
    del state,y
   del p,q,v,x,a0,a1
   torch.cuda.empty_cache()
 assert len(rows)==54 and hashes=={p:base.old.sha(ROOT/p) for p in hashes}
 OUT.mkdir(parents=True)
 with (OUT/'rows.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 summary=[]
 for prompt in (0,4):
  m={a:statistics.mean(r['D'] for r in rows if r['prompt']==prompt and r['arm']==a) for a in ARMS}
  samples=[m[a] for a in ARMS if a.startswith('arp')];mean=statistics.mean(samples)
  summary.append(dict(prompt=prompt,means=m,arp_mean=mean,arp_min=min(samples),arp_max=max(samples),arp_over_rtn=mean/m['rtn'],arp_over_svd=mean/m['svd']))
 base.old.save(OUT/'evidence.json',dict(source_sha256=hashes,input_artifacts=ev['input_artifacts'],payload_artifacts=artifacts,summary=summary,rtn_gate=all(s['arp_over_rtn']<=.9 for s in summary),svd_gate=all(s['arp_over_svd']<=.95 for s in summary),previous_max_abs_D_delta=max_repro,seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),rows_sha256=base.old.sha(OUT/'rows.csv')))
 status('COMPLETE: all 54 rows and payload CPU checks passed')
if __name__=='__main__':main()
