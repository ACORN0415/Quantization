"""Causal current-Q selection and future-Q diagnostic; incremental new-V error only."""
import argparse,csv,hashlib,json,math,os,resource,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src'),str(ROOT/'src/scripts')]
import torch
from transition_round_core import make_rtn
from postwo_round_core import optimize_rounding,output_error_metrics
DATA=Path.home()/'sfq_data/postwo_rounding'
OUT=ROOT/'src/experiments/postwo_rounding_development'
LAYERS=(0,14,29)
FILES=('tools/postwo_round_probe.py','tools/postwo_round_core.py','tools/test_postwo_round_core.py',
       'tools/transition_round_core.py','tools/postwo_transition_gate.py','src/experiments/POSTWO_ROUNDING_DEVELOPMENT.md')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()


def fingerprint():
 h=hashlib.sha256()
 for rel in FILES:h.update(rel.encode());h.update((ROOT/rel).read_bytes())
 return h.hexdigest()


def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def status(detail,state='running'):
 print(time.strftime('%Y-%m-%d %H:%M:%S'),state,detail,flush=True)
 save(DATA/'status.json',dict(state=state,detail=detail,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


def free_gpu():
 other=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 if any(s.strip().isdigit() and int(s.strip())!=os.getpid() for s in other.splitlines()):raise RuntimeError('GPU occupied')


def capture(prompt):
 import mechanism_worker as common
 import wan.modules.causal_model as model
 free_gpu();source=fingerprint();pilot=common.pilot;model_source=pilot.source_hash()
 folder=DATA/f'p{prompt:02d}_s101'
 if folder.exists():raise RuntimeError('refuse overwrite capture directory')
 folder.mkdir(parents=True);started=time.monotonic();status(f'CAPTURE p{prompt}: loading model')
 pipe=pilot.build(101);text=(ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines()[prompt]
 cond=pilot.encode(pipe,[text])[0];cache=pilot.bf16_cache(pipe,pilot.gpu);xc=pilot.cross_cache(pipe,pilot.gpu)
 noise=pilot.chunk_noise(pipe,126,101,pilot.gpu)
 expected=torch.load(Path.home()/f'sfq_data/policy_study/reference/p{prompt:02d}_s101/bf16.pt',map_location='cpu',weights_only=True)['latent']
 modules=[m for m in pipe.generator.modules() if isinstance(m,model.CausalWanSelfAttention)]
 if len(modules)!=30:raise RuntimeError('unexpected attention layer count')
 counts={layer:0 for layer in LAYERS};active={};handles=[];artifacts=[]
 def hook(layer):
  def before(module,args):
   call=counts[layer];counts[layer]+=1;active.update(layer=layer,call=call)
  return before
 for layer in LAYERS:handles.append(modules[layer].register_forward_pre_hook(hook(layer)))
 original=model.attention
 # Non-selected modules must clear active context too.
 def clear(module,args):active.clear()
 for layer,module in enumerate(modules):
  if layer not in LAYERS:handles.append(module.register_forward_pre_hook(clear))
 def attention(q,k,v,*args,**kwargs):
  if active:
   layer=active['layer'];chunk,call=divmod(active['call'],5)
   if (chunk,call) in ((6,4),(7,0)):
    indices=torch.linspace(0,q.shape[1]-1,32,device=q.device).long()
    payload=dict(query=q[:,indices].cpu(),query_indices=indices.cpu(),chunk=chunk,call=call)
    if (chunk,call)==(6,4):
     if k.shape[1]!=32760:raise RuntimeError('full7chunk cache required')
     payload.update(key=k.cpu(),new_value=v[:,-4680:].cpu(),wo=modules[layer].o.weight.cpu())
    path=folder/f'l{layer:02d}_c{chunk}.pt';torch.save(payload,path)
    artifacts.append(dict(file=str(path.relative_to(DATA)),bytes=path.stat().st_size,sha256=sha(path),layer=layer,chunk=chunk,call=call))
  return original(q,k,v,*args,**kwargs)
 model.attention=attention
 try:
  for chunk in range(8):
   result=pilot.run_chunk(pipe,cache,xc,cond,noise,chunk,101)
   if not torch.equal(result.cpu(),expected[:,chunk*3:(chunk+1)*3]):raise RuntimeError('BF16 capture identity failed')
   status(f'CAPTURE p{prompt}: {chunk+1}/8 chunks bitexact')
 finally:
  model.attention=original
  for handle in handles:handle.remove()
 if len(artifacts)!=6 or any(n!=40 for n in counts.values()):raise RuntimeError('incomplete capture')
 if source!=fingerprint() or model_source!=pilot.source_hash():raise RuntimeError('source changed during capture')
 save(folder/'manifest.json',dict(prompt=prompt,seed=101,source_sha256=source,model_source_sha256=model_source,
  teacher_bitexact_chunks=8,artifacts=artifacts,seconds=time.monotonic()-started,
  peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
  cpu_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def probabilities(query,key):
 q=query[0].float().permute(1,0,2);k=key[0].float().permute(1,2,0)
 logits=(q@k)/math.sqrt(q.shape[-1])
 before=logits.softmax(-1)
 after=logits.clone();after[:,:,:4680]=-torch.inf;after=after.softmax(-1)
 return before[:,:,-4680:].contiguous(),after[:,:,-4680:].contiguous()


def analyze():
 free_gpu();source=fingerprint();started=time.monotonic();rows=[];manifests=[];preparations=[]
 if OUT.exists():raise RuntimeError('refuse overwrite results')
 for prompt in (0,4):
  folder=DATA/f'p{prompt:02d}_s101';manifest=json.loads((folder/'manifest.json').read_text())
  if manifest['source_sha256']!=source:raise RuntimeError('capture/source mismatch')
  for item in manifest['artifacts']:
   if sha(DATA/item['file'])!=item['sha256']:raise RuntimeError('capture SHA mismatch')
  manifests.append(manifest)
  for layer in LAYERS:
   status(f'OPTIMIZE p{prompt} layer{layer}')
   data=torch.load(folder/f'l{layer:02d}_c6.pt',map_location='cuda',weights_only=True)
   later=torch.load(folder/f'l{layer:02d}_c7.pt',map_location='cuda',weights_only=True)
   torch.cuda.synchronize();begin=time.monotonic()
   train=probabilities(data['query'],data['key']);evaluation=probabilities(later['query'],data['key'])
   torch.cuda.synchronize();probability_seconds=time.monotonic()-begin
   wo=data['wo'].float();v=data['new_value'][0].float()
   torch.cuda.synchronize();begin=time.monotonic();norm_sq=float(torch.linalg.svdvals(wo).max().square());torch.cuda.synchronize()
   norm_seconds=time.monotonic()-begin
   preparations.append(dict(prompt=prompt,layer=layer,probability_seconds=probability_seconds,wo_spectral_seconds=norm_seconds,wo_norm_sq=norm_sq))
   initial=make_rtn(v,bits=4,group=16)
   if not math.isfinite(norm_sq):raise RuntimeError('nonfinite Wo norm')
   for arm,lam,local in (('rtn',0.,False),('full_mean',0.,False),('full_transition',1.,False),('headlocal_transition',1.,True)):
    torch.cuda.synchronize();begin=time.monotonic()
    if arm=='rtn':state=initial;meta=dict(candidates=1)
    else:state,meta=optimize_rounding(*train,v,wo,transition_weight=lam,head_local=local,steps=32,initial=initial,wo_norm_sq=norm_sq)
    torch.cuda.synchronize();seconds=time.monotonic()-begin
    restored=state.restore()
    metrics={}
    for split,weights in (('train',train),('eval',evaluation)):
     # All arms are evaluated under the same full post-Wo criterion.
     m=output_error_metrics(*weights,restored,v,wo,transition_weight=1.,head_local=False)
     metrics.update({split+'_'+name:(value/(weights[0].shape[1]*wo.shape[0]) if name in ('objective','mean_error_sq','transition_error_sq') else value) for name,value in m.items()})
     local_metrics=output_error_metrics(*weights,restored,v,wo,transition_weight=1.,head_local=True)
     metrics.update({split+'_headlocal_'+name:local_metrics[name]/(weights[0].shape[1]*wo.shape[0]) for name in ('mean_error_sq','transition_error_sq')})
    if not torch.equal(state.scale,initial.scale):raise RuntimeError('scale changed')
    if state.bytes()!=initial.bytes():raise RuntimeError('payload changed')
    path=DATA/f'p{prompt:02d}_l{layer:02d}_{arm}_codes.pt'
    torch.save(dict(codes=state.codes.cpu(),scale=state.scale.cpu(),shape=state.shape,bits=state.bits,group=state.group),path)
    meta_path=DATA/f'p{prompt:02d}_l{layer:02d}_{arm}_search.json'
    save(meta_path,meta)
    rows.append(dict(prompt=prompt,layer=layer,arm=arm,**metrics,search_seconds=seconds,
      metric_denominator=train[0].shape[1]*wo.shape[0],search_file=meta_path.name,search_sha256=sha(meta_path),
      changed_codes=meta.get('changed_codes',0),
      resident_bytes=state.bytes()['resident'],packed_estimate_bytes=state.bytes()['packed_estimate'],
      code_file=path.name,code_sha256=sha(path)))
   del data,later,train,evaluation,wo,v,state,restored,initial
   torch.cuda.empty_cache()
 if len(rows)!=24 or fingerprint()!=source:raise RuntimeError('incomplete or changed source')
 decisions=[]
 for prompt in (0,4):
  means={arm:{metric:statistics.mean(float(r['eval_'+metric]) for r in rows if r['prompt']==prompt and r['arm']==arm)
              for metric in ('mean_error_sq','transition_error_sq')} for arm in ('rtn','full_mean','full_transition','headlocal_transition')}
  base=means['full_mean'];candidate=means['full_transition'];valid=base['transition_error_sq']>1e-12*max(base['mean_error_sq'],1e-30)
  reduction=1-candidate['transition_error_sq']/base['transition_error_sq'] if valid else None
  damage=candidate['mean_error_sq']/base['mean_error_sq'] if base['mean_error_sq'] else None
  decisions.append(dict(prompt=prompt,means=means,baseline_jump_valid=valid,jump_reduction=reduction,mean_damage_ratio=damage,
                        passes=valid and reduction>=.1 and damage is not None and damage<=1.02))
 if any(not math.isfinite(float(r[k])) for r in rows for k in r if k.startswith(('train_','eval_'))):raise RuntimeError('nonfinite metric')
 OUT.mkdir(parents=True)
 with (OUT/'rows.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 save(OUT/'decision.json',dict(source_sha256=source,advance=all(d['passes'] for d in decisions),decisions=decisions,captures=manifests,
      preparation=preparations,seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
      peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope='new-V-only incremental FP32 post-Wo attention; teacher future-Q and counterfactual deletion, not FR'))
 status('COMPLETE '+('GO' if all(d['passes'] for d in decisions) else 'NO-GO'),'pass')
 print(json.dumps(decisions,ensure_ascii=False,indent=2),flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['capture','analyze']);p.add_argument('--prompt',type=int,choices=[0,4]);a=p.parse_args()
 torch.set_grad_enabled(False);torch.set_num_threads(4)
 try:
  if a.mode=='capture':
   if a.prompt is None:p.error('--prompt required')
   capture(a.prompt)
  else:analyze()
 except Exception as exc:status(str(exc),'failed');raise

if __name__=='__main__':main()
