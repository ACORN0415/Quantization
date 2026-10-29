"""Change-ranked, inverse, random and magnitude mixed precision; actual packing."""
import csv,hashlib,json,math,os,statistics,sys,time
from dataclasses import fields
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src'),str(ROOT/'src/scripts')]
import torch
import postwo_round_probe as old
from postwo_round_core import output_error_metrics
from change_mixed_core import encode_uniform,encode_mixed,pack_bits,unpack_bits
DATA=Path.home()/'sfq_data/change_mixed';MOTION=Path.home()/'sfq_data/motion_codec'
POSTWO=Path.home()/'sfq_data/postwo_rounding';OUT=ROOT/'src/experiments/change_mixed_development'
LAYERS=(0,14,29);SEEDS=(9101,9102,9103,9104,9105)
ARMS=('uniform2','uniform3','uniform4','change_high','change_low','magnitude_high')+tuple(f'random_{s}' for s in SEEDS)
FILES=('tools/change_mixed_probe.py','tools/change_mixed_core.py','tools/test_change_mixed_core.py',
       'tools/postwo_round_core.py','tools/postwo_round_probe.py','tools/transition_round_core.py',
       'src/experiments/CHANGE_MIXED_DEVELOPMENT.md')


def sha(path):return old.sha(path)


def save(path,obj):old.save(path,obj)


def fingerprint():
 h=hashlib.sha256()
 for rel in FILES:h.update(rel.encode());h.update((ROOT/rel).read_bytes())
 return h.hexdigest()


def status(detail,state='running'):
 print(time.strftime('%Y-%m-%d %H:%M:%S'),state,detail,flush=True)
 save(DATA/'status.json',dict(state=state,detail=detail,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


def timed(fn):
 value=fn();torch.cuda.synchronize();times=[]
 for _ in range(3):
  torch.cuda.synchronize();begin=time.perf_counter();value=fn();torch.cuda.synchronize();times.append(time.perf_counter()-begin)
 return value,statistics.median(times)


def rel(x,y):return float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-12))


def unique_bytes(states):
 seen=set();total=0
 for state in states:
  for field in fields(state):
   value=getattr(state,field.name)
   if isinstance(value,torch.Tensor):
    storage=value.untyped_storage();key=(str(value.device),storage.data_ptr(),storage.nbytes())
    if key not in seen:total+=storage.nbytes();seen.add(key)
 return total


def masks(v,grid):
 frame_count,height,width=grid;spatial=height*width
 if frame_count!=3 or spatial%2:raise RuntimeError('need3frames and even spatial tokens')
 x=v.reshape(1,frame_count,spatial,*v.shape[2:]).float()
 change=(x[:,1:]-x[:,:-1]).square().mean((0,1,3,4)).sqrt()
 magnitude=x.square().mean((0,1,3,4)).sqrt();global_rms=x.square().mean().sqrt()
 score=change/magnitude.clamp_min(max(float(.1*global_rms),1e-12))
 high=torch.zeros(spatial,device=v.device,dtype=torch.bool)
 high[torch.argsort(score,descending=True,stable=True)[:spatial//2]]=True
 mag=torch.zeros_like(high);mag[torch.argsort(magnitude,descending=True,stable=True)[:spatial//2]]=True
 result=dict(change_high=high.repeat(3),change_low=(~high).repeat(3),magnitude_high=mag.repeat(3))
 for seed in SEEDS:
  chosen=torch.randperm(spatial,generator=torch.Generator(device='cpu').manual_seed(seed))[:spatial//2]
  random=torch.zeros(spatial,dtype=torch.bool);random[chosen]=True
  result[f'random_{seed}']=random.to(v.device).repeat(3)
 return result,dict(score=score,magnitude=magnitude,change=change,global_rms=global_rms)


def main():
 from wan.modules.causal_model import causal_rope_apply
 old.free_gpu();source=fingerprint();started=time.monotonic()
 if OUT.exists():raise RuntimeError('refuse overwrite results')
 # Actual CUDA packing gate, independent of research captures.
 for bits in (1,2,3,4):
  x=(torch.arange(137,device='cuda')%(1<<bits)).to(torch.uint8)
  if not torch.equal(unpack_bits(pack_bits(x,bits),bits,len(x)),x):raise RuntimeError('CUDA pack roundtrip failed')
 rows=[];costs=[];gates=[];artifacts=[];mask_records=[]
 previous=json.loads((ROOT/'src/experiments/postwo_rounding_development/decision.json').read_text())
 if previous['source_sha256']!=old.fingerprint():raise RuntimeError('legacy attention source changed')
 for prompt in (0,4):
  status(f'p{prompt}: validating shared BF16 inputs')
  motion_manifest=json.loads((MOTION/f'p{prompt:02d}_s101/manifest.json').read_text())
  post_manifest=next(c for c in previous['captures'] if c['prompt']==prompt)
  wanted=[(MOTION,a) for a in motion_manifest['artifacts'] if a['chunk']==6]+[(POSTWO,a) for a in post_manifest['artifacts']]
  for root,item in wanted:
   if sha(root/item['file'])!=item['sha256']:raise RuntimeError('input SHA mismatch')
   artifacts.append(dict(root='motion_codec' if root==MOTION else 'postwo_rounding',**item))
  raw={};post={};weights={}
  for layer in LAYERS:
   raw[layer]=torch.load(MOTION/f'p{prompt:02d}_s101/c06_l{layer:02d}.pt',map_location='cuda',weights_only=True)
   post[layer]=torch.load(POSTWO/f'p{prompt:02d}_s101/l{layer:02d}_c6.pt',map_location='cuda',weights_only=True)
   q=torch.load(POSTWO/f'p{prompt:02d}_s101/l{layer:02d}_c7.pt',map_location='cuda',weights_only=True)['query']
   if not torch.equal(raw[layer]['v'],post[layer]['new_value']):raise RuntimeError('crosscapture V mismatch')
   grid=torch.tensor([raw[layer]['grid']],device='cuda')
   roped=causal_rope_apply(raw[layer]['k'],grid,raw[layer]['freqs'],start_frame=raw[layer]['start_frame'])
   if not torch.equal(roped,post[layer]['key'][:,-4680:]):raise RuntimeError('crosscapture K/RoPE mismatch')
   weights[layer]=old.probabilities(q,post[layer]['key'])
  torch.cuda.synchronize();begin=time.perf_counter();mask_set,score=masks(raw[0]['v'],raw[0]['grid']);torch.cuda.synchronize()
  mask_seconds=time.perf_counter()-begin
  packed={arm:pack_bits(mask,1) for arm,mask in mask_set.items()}
  if not torch.equal(mask_set['change_low'],~mask_set['change_high']):raise RuntimeError('inverse not complement')
  if any(int(mask.sum())!=2340 for mask in mask_set.values()):raise RuntimeError('high token budget mismatch')
  score_path=DATA/f'p{prompt:02d}_scores.pt';score_path.parent.mkdir(parents=True,exist_ok=True)
  torch.save({k:v.cpu() for k,v in score.items()},score_path)
  for arm,mask in mask_set.items():
   path=DATA/f'p{prompt:02d}_{arm}_mask.pt';torch.save(dict(mask=mask.cpu(),packed_mask=packed[arm].cpu()),path)
   mask_records.append(dict(prompt=prompt,arm=arm,file=path.name,sha256=sha(path),high_tokens=int(mask.sum()),mask_bytes=packed[arm].numel(),
       score_file=score_path.name,score_sha256=sha(score_path),mask_construction_seconds=mask_seconds))
  expected_max={}
  for arm in ARMS:
   states=[];serial={};vstates=[];encode_total=0.;decode_total=0.;arm_rows=[]
   for layer in LAYERS:
    r=raw[layer];p=post[layer]
    def encode_pair():
     if arm.startswith('uniform'):return {kind:encode_uniform(r[kind],bits=int(arm[-1]),group=16) for kind in ('k','v')}
     return {kind:encode_mixed(r[kind],mask_set[arm],low_bits=2,high_bits=4,group=16,packed_mask=packed[arm]) for kind in ('k','v')}
    pair,encode_s=timed(encode_pair);restored,decode_s=timed(lambda:{kind:state.restore() for kind,state in pair.items()})
    for kind,state in pair.items():
     if arm=='uniform2':expected_max[layer,kind]=state.group_max.clone()
     elif not torch.equal(state.group_max,expected_max[layer,kind]):raise RuntimeError('group maximum depends on allocation')
    # Same-bit arbitrary partition must not change original-group reconstruction.
    if arm=='change_high':
     for kind in ('k','v'):
      same=encode_mixed(r[kind],mask_set[arm],low_bits=4,high_bits=4,group=16,packed_mask=packed[arm]).restore()
      uniform=encode_uniform(r[kind],bits=4,group=16).restore()
      if not torch.equal(same,uniform):raise RuntimeError('same-bit group identity failed')
     gates.append(dict(prompt=prompt,layer=layer,v_bitexact=True,k_rope_bitexact=True,samebit_kv_identity=True))
    m=output_error_metrics(*weights[layer],restored['v'][0],r['v'][0].float(),p['wo'].float(),transition_weight=1.,head_local=False)
    count=weights[layer][0].shape[1]*p['wo'].shape[0]
    grid=torch.tensor([r['grid']],device='cuda')
    kpost=causal_rope_apply(restored['k'].to(r['k'].dtype),grid,r['freqs'],start_frame=r['start_frame'])
    arm_rows.append(dict(prompt=prompt,layer=layer,arm=arm,eval_d=m['mean_error_sq']/count,eval_j=m['transition_error_sq']/count,
        v_rel=rel(restored['v'],r['v']),k_pre_rel=rel(restored['k'],r['k']),k_post_rel=rel(kpost,p['key'][:,-4680:]),
        v_layer_bytes=pair['v'].bytes()['resident'],kv_layer_bytes=unique_bytes(list(pair.values())),
        mask_bytes=0 if arm.startswith('uniform') else packed[arm].numel(),
        encode_kv_seconds_median3=encode_s,decode_kv_seconds_median3=decode_s))
    encode_total+=encode_s;decode_total+=decode_s;states.extend(pair.values());vstates.append(pair['v'])
    if arm.startswith('uniform'):mask_cpu=None
    else:
     if layer==LAYERS[0]:mask_cpu=packed[arm].cpu()
    serial[str(layer)]={}
    for kind,state in pair.items():
     serial[str(layer)][kind]={field.name:(mask_cpu if field.name=='packed_mask' else getattr(state,field.name).cpu()
       if isinstance(getattr(state,field.name),torch.Tensor) else getattr(state,field.name)) for field in fields(state)}
   artifact=DATA/f'p{prompt:02d}_{arm}_packed.pt';torch.save(serial,artifact)
   for row in arm_rows:row.update(state_file=artifact.name,state_sha256=sha(artifact))
   rows.extend(arm_rows)
   costs.append(dict(prompt=prompt,arm=arm,v_actual_packed_bytes=unique_bytes(vstates),kv_actual_packed_bytes=unique_bytes(states),
     mask_bytes=0 if arm.startswith('uniform') else packed[arm].numel(),encode_kv_seconds_median3_sum=encode_total,
     decode_kv_seconds_median3_sum=decode_total,mask_construction_seconds=0. if arm.startswith('uniform') else mask_seconds,
     state_file=artifact.name,state_sha256=sha(artifact)))
   status(f'p{prompt} {arm}: {len(rows)}/66 rows')
   del states,vstates,pair,restored,serial
  del raw,post,weights,mask_set,packed,score,expected_max
  torch.cuda.empty_cache()
 if len(rows)!=66 or len(costs)!=22 or source!=fingerprint():raise RuntimeError('incomplete or source changed')
 if any(not math.isfinite(float(row[k])) for row in rows for k in ('eval_d','eval_j','v_rel','k_pre_rel','k_post_rel')):raise RuntimeError('nonfinite metric')
 decisions=[]
 for prompt in (0,4):
  means={arm:statistics.mean(r['eval_d'] for r in rows if r['prompt']==prompt and r['arm']==arm) for arm in ARMS}
  random=[means[f'random_{s}'] for s in SEEDS];random_mean=statistics.mean(random);high=means['change_high']
  random_gain=1-high/random_mean;inverse_gain=1-high/means['change_low'];uniform_ratio=high/means['uniform3']
  b={c['arm']:c['v_actual_packed_bytes'] for c in costs if c['prompt']==prompt}
  mixed_bytes=[b[arm] for arm in ARMS if not arm.startswith('uniform')]
  if len(set(mixed_bytes))!=1:raise RuntimeError('mixed budgets differ')
  overhead=b['change_high']/b['uniform3']-1
  decisions.append(dict(prompt=prompt,means=means,random_mean=random_mean,random_min=min(random),random_max=max(random),
     high_vs_random_reduction=random_gain,high_vs_inverse_reduction=inverse_gain,high_over_uniform3=uniform_ratio,
     v_payload_overhead=overhead,passes=random_gain>=.05 and inverse_gain>=.05 and uniform_ratio<=1.02 and overhead<=.01))
 OUT.mkdir(parents=True)
 for name,records in (('rows',rows),('costs',costs)):
  with (OUT/f'{name}.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 save(OUT/'decision.json',dict(source_sha256=source,advance=all(d['passes'] for d in decisions),decisions=decisions,
      input_artifacts=artifacts,masks=mask_records,gates=gates,cuda_pack_bits=[1,2,3,4],
      seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
      scope='V-only incremental teacher-query diagnostic; K reconstruction secondary; true packed payload, not full cache peak/video'))
 status('COMPLETE '+('GO' if all(d['passes'] for d in decisions) else 'NO-GO'),'pass');print(json.dumps(decisions,indent=2),flush=True)

if __name__=='__main__':
 torch.set_grad_enabled(False);torch.set_num_threads(4)
 try:main()
 except Exception as exc:status(str(exc),'failed');raise
