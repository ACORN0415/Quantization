"""Six codecs, quantization order, fixed-rank and fixed-byte comparisons."""
import csv,json,time,math,os,statistics
from pathlib import Path
import torch
import codec_order_core as codec
import tensor_cross_probe as old
from postwo_round_core import output_error_metrics
ROOT=old.ROOT;OUT=ROOT/'src/experiments/codec_order_development';DATA=Path.home()/'sfq_data/codec_order'
SOURCES=['tools/codec_order_core.py','tools/codec_order_probe.py','tools/test_codec_order.py','src/experiments/CODEC_ORDER_DEVELOPMENT.md','tools/tensor_adaptive_core.py','tools/change_mixed_core.py','tools/postwo_round_core.py','tools/postwo_round_probe.py']

def status(message,state='running'):
 print(time.strftime('%F %T'),message,flush=True)
 old.old.save(DATA/'status.json',dict(state=state,detail=message,pid=os.getpid(),updated_utc=time.strftime('%FT%TZ',time.gmtime())))

def clocked(fn):
 torch.cuda.synchronize();begin=time.perf_counter();value=fn();torch.cuda.synchronize()
 return value,time.perf_counter()-begin

def decode_timed(fn):
 fn();torch.cuda.synchronize();ts=[]
 for _ in range(3):
  y,t=clocked(fn);ts.append(t)
 return y,statistics.median(ts)

def main():
 old.old.free_gpu()
 if OUT.exists():raise RuntimeError('refuse overwrite')
 DATA.mkdir(parents=True,exist_ok=True)
 torch.backends.cuda.matmul.allow_tf32=False
 hashes={p:old.old.sha(ROOT/p) for p in SOURCES}
 previous=json.loads((ROOT/'src/experiments/tensor_cross_development/evidence.json').read_text())
 inputs=previous['input_artifacts']
 for a in inputs:assert old.old.sha(Path.home()/'sfq_data/postwo_rounding'/a['file'])==a['sha256']
 prior=list(csv.DictReader((ROOT/'src/experiments/tensor_cross_development/rows.csv').open()))
 assert old.old.sha(ROOT/'src/experiments/tensor_cross_development/rows.csv')==previous['rows_sha256']
 rows=[];artifacts=[];started=time.monotonic();max_plain_delta=0.
 for prompt in (0,4):
  for layer in (0,14,29):
   status(f'p{prompt} l{layer}: load frozen inputs')
   folder=Path.home()/f'sfq_data/postwo_rounding/p{prompt:02d}_s101'
   p=torch.load(folder/f'l{layer:02d}_c6.pt',map_location='cuda',weights_only=True)
   q=torch.load(folder/f'l{layer:02d}_c7.pt',map_location='cuda',weights_only=True)['query']
   v=p['new_value'].float();x=v[0].reshape(3,1560,12,128).permute(2,0,1,3).contiguous()
   shape=tuple(x.shape);a0,a1=old.old.probabilities(q,p['key']);den=a0.shape[1]*1536
   raw_cache={}
   for bits in (2,3,4):
    plain,quant_s=clocked(lambda:old.encode_uniform(v,bits=bits,group=16))
    qv,plain_decode_s=decode_timed(plain.restore)
    qx=qv[0].reshape(3,1560,12,128).permute(2,0,1,3).contiguous()
    budget=plain.bytes()['resident']
    baseline=output_error_metrics(a0,a1,qv[0],v[0],p['wo'].float())['mean_error_sq']/den
    previous_d=float(next(r['D'] for r in prior if int(r['prompt'])==prompt and int(r['layer'])==layer and int(r['bits_budget'])==bits and r['arm']=='rtn'))
    max_plain_delta=max(max_plain_delta,abs(previous_d-baseline));assert abs(previous_d-baseline)<1e-8
    def add(method,order,state,rank,source_s=0.,decomp_s=0.,store_s=0.,decode_fn=None):
     if state is None:
      y=qv;decode_s=plain_decode_s;payload=budget;cpu_rel=0.;filename='';filehash=''
     else:
      y,decode_s=decode_timed(lambda:codec.decode(state));payload=codec.payload_bytes(state)
      assert payload==codec.expected_bytes(method,shape,rank,None if order in ('method_only','quant_then_method') else bits)
      assert payload<=budget
      filename=f'p{prompt}_l{layer}_b{bits}_{method}_{order}.pt';path=DATA/filename
      torch.save(codec.cpu_state(state),path)
      loaded=torch.load(path,map_location='cpu',weights_only=True)
      cy=codec.decode(loaded);cpu_rel=float((cy-y.cpu()).norm()/y.cpu().norm().clamp_min(1e-12));assert cpu_rel<3e-5,(filename,cpu_rel)
      filehash=old.old.sha(path);artifacts.append(dict(file=filename,sha256=filehash,bytes=path.stat().st_size))
     final=output_error_metrics(a0,a1,y[0],v[0],p['wo'].float())['mean_error_sq']/den
     extra=output_error_metrics(a0,a1,y[0],qv[0],p['wo'].float())['mean_error_sq']/den
     row=dict(prompt=prompt,layer=layer,bits=bits,method=method,order=order,rank=rank,payload_bytes=payload,budget_bytes=budget,payload_ratio=payload/budget,D=final,plain_D=baseline,D_ratio=final/baseline,added_output_D=extra,v_rel=float((y-v).norm()/v.norm()),v_change_from_plain=float((y-qv).norm()/qv.norm()),source_quant_decode_seconds=source_s,decompose_seconds=decomp_s,store_seconds=store_s,encode_seconds=source_s+decomp_s+store_s,decode_seconds=decode_s,cpu_gpu_rel=cpu_rel,state_file=filename,state_sha256=filehash)
     for key in ('D','D_ratio','added_output_D','v_rel','v_change_from_plain','encode_seconds','decode_seconds'):assert math.isfinite(row[key]),row
     rows.append(row)
     with (DATA/'rows.partial.csv').open('w') as f:
      w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
     status(f'{len(rows)}/450 p{prompt} l{layer} b{bits} {method}/{order} ratio={row["D_ratio"]:.4g}, bytes={payload}')
    add('plain','plain',None,0,quant_s)
    for method in codec.METHODS:
     rlow=codec.choose_rank(method,shape,budget)
     rhigh=codec.choose_rank(method,shape,budget,bits)
     for order in ('method_only','quant_then_method','method_then_quant_rank','method_then_quant_budget'):
      rank=rhigh if order=='method_then_quant_budget' else rlow
      if order=='quant_then_method':
       raw,decomp_s=clocked(lambda:codec.decompose(qx,method,rank))
      else:
       key=(method,rank)
       if key not in raw_cache:raw_cache[key]=clocked(lambda:codec.decompose(x,method,rank))
       raw,decomp_s=raw_cache[key]
      storage_bits=None if order in ('method_only','quant_then_method') else bits
      state,store_s=clocked(lambda:codec.store(raw,storage_bits))
      source_s=quant_s+plain_decode_s if order=='quant_then_method' else 0.
      add(method,order,state,rank,source_s,decomp_s,store_s)
      del state,raw
    del plain,qv,qx
   del p,q,v,x,a0,a1,raw_cache
   torch.cuda.empty_cache()
 assert len(rows)==450 and hashes=={p:old.old.sha(ROOT/p) for p in SOURCES}
 OUT.mkdir(parents=True)
 with (OUT/'rows.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 old.old.save(OUT/'evidence.json',dict(source_sha256=hashes,input_artifacts=inputs,payload_artifacts=artifacts,rows_sha256=old.old.sha(OUT/'rows.csv'),max_plain_D_delta=max_plain_delta,seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),rows=len(rows),selection_seed=9101))
 status('COMPLETE all 450 rows, 432 CPU payload checks','complete')
if __name__=='__main__':main()
