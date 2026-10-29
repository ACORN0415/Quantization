"""One real target chunk: fixed/independent/BanLast/round-robin V precision."""
import csv,json,os,sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/scripts')]
import torch
import p3_tf_fr as pilot
import wan.modules.causal_model as model
import postwo_round_probe as util
from change_mixed_core import encode_uniform
from history_precision_core import schedules,POLICIES
DATA=Path.home()/'sfq_data/history_precision';OUT=ROOT/'src/experiments/history_precision_development'
LAYERS=(0,14,29);SEEDS=(9101,9102,9103)
SOURCES=['tools/history_precision_probe.py','tools/history_precision_core.py','tools/test_history_precision.py','src/experiments/HISTORY_PRECISION_DEVELOPMENT.md','tools/change_mixed_core.py']

def status(msg):
 print(time.strftime('%F %T'),msg,flush=True)
 util.save(DATA/'status.json',dict(detail=msg,pid=os.getpid(),updated_utc=time.strftime('%FT%TZ',time.gmtime())))

def main():
 torch.set_grad_enabled(False);torch.set_num_threads(4);util.free_gpu()
 if OUT.exists():raise RuntimeError('refuse overwrite')
 DATA.mkdir(parents=True,exist_ok=True)
 source={p:util.sha(ROOT/p) for p in SOURCES};modelhash=pilot.source_hash()
 status('building model; method 2 real one-chunk pilot')
 started=time.monotonic();pipe=pilot.build(101)
 prompts=(ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines()
 conds=pilot.encode(pipe,[prompts[0],prompts[4]])
 noise=pilot.chunk_noise(pipe,126,101,pilot.gpu)
 assert len(pipe.denoising_step_list)==4
 modules=[m for m in pipe.generator.modules() if isinstance(m,model.CausalWanSelfAttention)]
 assert len(modules)==30
 rows=[];gates=[];artifacts=[];reference_hashes=[];memory_rows=[]
 for prompt,cond in zip((0,4),conds):
  refpath=Path.home()/f'sfq_data/policy_study/reference/p{prompt:02d}_s101/bf16.pt'
  reference_hashes.append(dict(prompt=prompt,file=f'policy_study/reference/p{prompt:02d}_s101/bf16.pt',sha256=util.sha(refpath)))
  expected=torch.load(refpath,map_location='cpu',weights_only=True)['latent']
  cache=pilot.bf16_cache(pipe,pilot.gpu);xc=pilot.cross_cache(pipe,pilot.gpu)
  for chunk in range(7):
   y=pilot.run_chunk(pipe,cache,xc,cond,noise,chunk,101)
   assert torch.equal(y.cpu(),expected[:,3*chunk:3*(chunk+1)]),(prompt,chunk,'prefix mismatch')
   status(f'p{prompt} BF16 prefix {chunk+1}/7 exact')
  backup=[{k:v.cpu().clone() if isinstance(v,torch.Tensor) else v for k,v in c.items()} for c in cache]
  targets={l:cache[l]['v'][:,-4680:].clone() for l in LAYERS}
  packed={l:encode_uniform(targets[l],bits=2,group=16) for l in LAYERS}
  low={l:packed[l].restore().to(torch.bfloat16) for l in LAYERS}
  memory_rows.append(dict(prompt=prompt,low_code_scale_bytes=sum(s.bytes()['resident'] for s in packed.values()),full_bf16_reference_bytes=sum(v.numel()*v.element_size() for v in targets.values()),mask_bits_per_call=4680,logical_bf16_values_read_per_call=1170*12*128*3))
  def reset():
   for dest,src in zip(cache,backup):
    for k,value in src.items():
     if isinstance(value,torch.Tensor):dest[k].copy_(value)
     else:dest[k]=value
  reset();reference=pilot.run_chunk(pipe,cache,xc,cond,noise,7,101)
  assert torch.equal(reference.cpu(),expected[:,21:24]),'target BF16 mismatch'
  original=model.attention;active={};counts={l:0 for l in LAYERS};handles=[];reading=[];masks=None
  def before(layer):
   def hook(module,args):
    active.clear()
    if layer in LAYERS:
     active.update(layer=layer,call=counts[layer]);counts[layer]+=1
   return hook
  for l,m in enumerate(modules):handles.append(m.register_forward_pre_hook(before(l)))
  def attention(q,k,v,*args,**kwargs):
   if active and active['call']<4:
    l=active['layer'];call=active['call']
    assert v.shape[1]==32760
    part=v[:,-9360:-4680]
    assert torch.equal(part,targets[l]),'past chunk content or indexing changed'
    mask=masks[call]
    changed=torch.where(mask[None,:,None,None],targets[l],low[l])
    vv=v.clone();vv[:,-9360:-4680]=changed
    assert torch.equal(part,targets[l]),'original cache mutated'
    reading.append(dict(layer=l,call=call,high_tokens=int(mask.sum())))
    return original(q,k,vv,*args,**kwargs)
   return original(q,k,v,*args,**kwargs)
  model.attention=attention
  def run(mask):
   nonlocal masks,counts,reading
   masks=mask.to(pilot.gpu);counts={l:0 for l in LAYERS};reading=[];reset()
   torch.cuda.synchronize();begin=time.monotonic()
   out=pilot.run_chunk(pipe,cache,xc,cond,noise,7,101)
   torch.cuda.synchronize();elapsed=time.monotonic()-begin
   assert counts=={l:5 for l in LAYERS} and len(reading)==12
   assert all(r['high_tokens']==int(mask[r['call']].sum()) for r in reading)
   assert bool(torch.isfinite(out).all())
   return out,elapsed
  try:
   high,_=run(torch.ones(4,4680,dtype=torch.bool))
   assert torch.equal(high,reference),'all-high hook gate failed'
   gates.append(dict(prompt=prompt,prefix_chunks_bitexact=7,target_bitexact=True,all_high_hook_bitexact=True))
   status(f'p{prompt} all-high actual hook gate PASS')
   experiments=[('low_only',0,torch.zeros(4,4680,dtype=torch.bool))]
   for seed in SEEDS:
    choices=schedules(seed=seed)
    experiments.extend((name,seed,choices[name]) for name in POLICIES)
   for policy,seed,mask in experiments:
    result,seconds=run(mask)
    error=result.float()-reference.float();mse=float(error.square().mean());energy=float(reference.float().square().mean())
    path=DATA/f'p{prompt}_{policy}_s{seed}.pt'
    torch.save(dict(latent=result.cpu(),mask=mask,reference=reference.cpu()),path)
    artifacts.append(dict(file=path.name,sha256=util.sha(path)))
    row=dict(prompt=prompt,policy=policy,selection_seed=seed,latent_mse=mse,latent_nmse=mse/energy,latent_rel=math.sqrt(mse/energy),sampling_seconds=seconds,high_tokens_per_call=int(mask[0].sum()),target_chunk=7)
    rows.append(row)
    with (DATA/'rows.partial.csv').open('w') as f:
     w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    status(f'{len(rows)}/26 p{prompt} {policy} s{seed}: latent_rel={row["latent_rel"]:.7g}, {seconds:.2f}s')
  finally:
   model.attention=original
   for h in handles:h.remove()
  del cache,xc,backup,targets,low,packed,expected,reference,result,high,y,experiments
  torch.cuda.empty_cache()
 assert len(rows)==26 and source=={p:util.sha(ROOT/p) for p in SOURCES} and modelhash==pilot.source_hash()
 OUT.mkdir(parents=True)
 with (OUT/'rows.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 util.save(OUT/'evidence.json',dict(source_sha256=source,model_source_sha256=modelhash,reference_files=reference_hashes,gates=gates,artifacts=artifacts,memory=memory_rows,rows_sha256=util.sha(OUT/'rows.csv'),seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
 status('COMPLETE method 2: 24 scheduled chunks + 2 low baselines, all gates passed')
if __name__=='__main__':main()
