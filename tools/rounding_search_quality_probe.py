"""Common training-only code pool and exact chosen-16-code diagnostic."""
import csv,hashlib,json,math,os,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src'),str(ROOT/'src/scripts')]
import torch
import postwo_round_probe as old
from postwo_round_core import output_error_metrics
from transition_round_core import make_rtn,ValueCode
from rounding_search_quality_core import trace_rounding,enumerate_binary_quadratic
DATA=Path.home()/'sfq_data/rounding_search_quality'
INPUT=Path.home()/'sfq_data/postwo_rounding'
OUT=ROOT/'src/experiments/rounding_search_quality_development'
FILES=('tools/rounding_search_quality_probe.py','tools/rounding_search_quality_core.py','tools/test_rounding_search_quality_core.py',
       'tools/postwo_round_core.py','tools/postwo_round_probe.py','tools/transition_round_core.py',
       'src/experiments/ROUNDING_SEARCH_QUALITY_DEVELOPMENT.md')
ARMS=('full_mean','full_transition','headlocal_transition')


def sha(path):return old.sha(path)


def save(path,value):old.save(path,value)


def fingerprint():
 h=hashlib.sha256()
 for rel in FILES:h.update(rel.encode());h.update((ROOT/rel).read_bytes())
 return h.hexdigest()


def status(detail,state='running'):
 print(time.strftime('%Y-%m-%d %H:%M:%S'),state,detail,flush=True)
 save(DATA/'status.json',dict(state=state,detail=detail,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


def code_hash(state):return hashlib.sha256(state.codes.cpu().contiguous().numpy().tobytes()).hexdigest()


def metric(weights,state,v,wo):
 m=output_error_metrics(*weights,state.restore(),v,wo,transition_weight=1.,head_local=False)
 count=weights[0].shape[1]*wo.shape[0]
 return dict(d=m['mean_error_sq']/count,j=m['transition_error_sq']/count,objective=m['objective']/count)


def copy_state(state):return ValueCode(state.codes.clone(),state.scale,state.shape,state.bits,state.group)


def exact_problem(train,v,wo,initial):
 """FP64 model of the stored FP32 coefficients; all nonselected codes at RTN."""
 n,h,d=v.shape;a0,a1=(a.double() for a in train);w=wo.double()
 baseline=initial.restore().double();x=v.double();error=(baseline-x).permute(1,0,2)
 e0,e1=a0@error,a1@error
 p0=e0.permute(1,0,2).reshape(a0.shape[1],-1)@w.T
 p1=e1.permute(1,0,2).reshape(a1.shape[1],-1)@w.T
 g0=(p0@w).reshape(a0.shape[1],h,d).permute(1,0,2)
 g1=(p1@w).reshape(a1.shape[1],h,d).permute(1,0,2)
 gradient=(a0.transpose(-1,-2)@g0+a1.transpose(-1,-2)@g1).permute(1,0,2)
 scales=initial.scale[:,0].index_select(0,torch.arange(n,device=v.device)//initial.group).double()
 qmax=(1<<(initial.bits-1))-1;current=initial.codes[:n].double()-qmax
 low=(x/scales).floor().clamp(-qmax,qmax);high=(x/scales).ceil().clamp(-qmax,qmax)
 if not ((current==low)|(current==high)).all():raise RuntimeError('RTN not in FP64 neighbor set')
 alternative=torch.where(current==low,high,low)
 flip=(alternative-current)*scales
 # Diagonal Hessian of .5(||P0||²+||P1||²) for each value scalar.
 attn_square=(a0.square().sum(1)+a1.square().sum(1)).T[:,:,None]
 column_square=w.square().sum(0).reshape(h,d)
 gain=-(gradient*flip+.5*attn_square*column_square[None]*flip.square())
 gain=torch.where(low!=high,gain,torch.full_like(gain,-torch.inf))
 flat=torch.argsort(gain.reshape(-1),descending=True,stable=True)[:16]
 if not torch.isfinite(gain.reshape(-1)[flat]).all():raise RuntimeError('fewer than16 valid flips')
 token=flat//(h*d);head=(flat//d)%h;channel=flat%d
 coords=torch.stack((token,head,channel),1)
 del gradient,g0,g1,gain,attn_square,column_square
 effects0=[];effects1=[]
 for t,hh,cc in coords.tolist():
  vector=w[:,hh*d+cc];delta=flip[t,hh,cc]
  effects0.append((a0[hh,:,t,None]*vector[None,:]*delta).reshape(-1))
  effects1.append((a1[hh,:,t,None]*vector[None,:]*delta).reshape(-1))
 b0,b1=torch.stack(effects0),torch.stack(effects1);diff=b1-b0;base_diff=(p1-p0).reshape(-1)
 linear_d=b0@p0.reshape(-1)+b1@p1.reshape(-1);quad_d=.5*(b0@b0.T+b1@b1.T)
 const_d=.5*(p0.square().sum()+p1.square().sum())
 linear_j=2*(diff@base_diff);quad_j=diff@diff.T;const_j=base_diff.square().sum()
 coeff=dict(linear_d=linear_d,quad_d=quad_d,const_d=const_d,linear_j=linear_j,quad_j=quad_j,const_j=const_j)
 result=enumerate_binary_quadratic(**coeff,batch_size=1024)
 alternative_codes=alternative[token,head,channel].add(qmax).to(torch.uint8)
 return coeff,result,coords,flat,alternative_codes


def main():
 source=fingerprint();started=time.monotonic();old.free_gpu()
 if OUT.exists():raise RuntimeError('refuse overwrite results')
 previous=json.loads((ROOT/'src/experiments/postwo_rounding_development/decision.json').read_text())
 previous_rows=list(csv.DictReader((ROOT/'src/experiments/postwo_rounding_development/rows.csv').open()))
 if previous['source_sha256']!=old.fingerprint():raise RuntimeError('legacy numeric source mismatch')
 input_artifacts=[]
 for manifest in previous['captures']:
  for artifact in manifest['artifacts']:
   if sha(INPUT/artifact['file'])!=artifact['sha256']:raise RuntimeError('capture SHA mismatch')
   input_artifacts.append(artifact)
 for row in previous_rows:
  for field,hash_field in (('code_file','code_sha256'),('search_file','search_sha256')):
   if sha(INPUT/row[field])!=row[hash_field]:raise RuntimeError('legacy artifact SHA mismatch')
 pool=[];selections=[];cells=[];parities=[]
 norms={(p['prompt'],p['layer']):p['wo_norm_sq'] for p in previous['preparation']}
 for prompt in (0,4):
  for layer in (0,14,29):
   status(f'p{prompt} layer{layer}: replaying shared97 candidate pool')
   begin=time.monotonic();folder=INPUT/f'p{prompt:02d}_s101'
   data=torch.load(folder/f'l{layer:02d}_c6.pt',map_location='cuda',weights_only=True)
   later=torch.load(folder/f'l{layer:02d}_c7.pt',map_location='cuda',weights_only=True)
   train=old.probabilities(data['query'],data['key'])
   v=data['new_value'][0].float();wo=data['wo'].float();initial=make_rtn(v,bits=4,group=16)
   candidates=[];best={};states={}
   def visit(arm,iteration,state):
    m=metric(train,state,v,wo);idx=len(candidates)
    row=dict(prompt=prompt,layer=layer,slot=idx,generator=arm,iteration=iteration,
             train_d=m['d'],train_j=m['j'],train_objective=m['objective'],code_hash=code_hash(state))
    candidates.append(row)
    for selector,key in (('mean','d'),('transition','objective')):
     if selector not in best or m[key]<best[selector]['score']:
      best[selector]=dict(score=m[key],slot=idx,metrics=m);states[selector]=copy_state(state)
   visit('rtn',0,initial)
   for arm,lam,local in (('full_mean',0.,False),('full_transition',1.,False),('headlocal_transition',1.,True)):
    def callback(iteration,state,arm=arm):
     if iteration:visit(arm,iteration,state)
    result,meta=trace_rounding(*train,v,wo,callback=callback,transition_weight=lam,head_local=local,steps=32,
                               initial=initial,wo_norm_sq=norms[prompt,layer])
    legacy_row=next(r for r in previous_rows if int(r['prompt'])==prompt and int(r['layer'])==layer and r['arm']==arm)
    legacy=torch.load(INPUT/legacy_row['code_file'],map_location='cuda',weights_only=True)
    legacy_meta=json.loads((INPUT/legacy_row['search_file']).read_text())
    if not torch.equal(result.codes,legacy['codes']) or meta['training_objective_history']!=legacy_meta['training_objective_history']:
     raise RuntimeError(f'legacy trace not bitexact {prompt}/{layer}/{arm}')
    parities.append(dict(prompt=prompt,layer=layer,arm=arm,bitexact_codes=True,bitexact_history=True))
   if len(candidates)!=97:raise RuntimeError('candidate pool incomplete')
   pool.extend(candidates);pool_seconds=time.monotonic()-begin
   # Evaluation queries are first introduced only after selection.
   evaluation=old.probabilities(later['query'],data['key'])
   def record(space,selector,state,chosen,extra):
    tr=metric(train,state,v,wo);ev=metric(evaluation,state,v,wo)
    if not torch.equal(state.scale,initial.scale) or state.bytes()!=initial.bytes():raise RuntimeError('storage changed')
    path=DATA/f'p{prompt:02d}_l{layer:02d}_{space}_{selector}.pt'
    torch.save(dict(codes=state.codes.cpu(),scale=state.scale.cpu(),shape=state.shape,bits=state.bits,group=state.group),path)
    selections.append(dict(prompt=prompt,layer=layer,space=space,selector=selector,chosen_index=int(chosen),
      train_d=tr['d'],train_j=tr['j'],train_objective=tr['objective'],eval_d=ev['d'],eval_j=ev['j'],
      resident_bytes=state.bytes()['resident'],packed_estimate_bytes=state.bytes()['packed_estimate'],
      code_hash=code_hash(state),code_file=path.name,code_sha256=sha(path),**extra))
   for selector in ('mean','transition'):
    record('common_pool',selector,states[selector],best[selector]['slot'],dict(quadratic_train_d=None,quadratic_train_j=None))
   status(f'p{prompt} layer{layer}: exact65,536 chosen16code patterns')
   torch.cuda.synchronize();exact_started=time.monotonic()
   coeff,enum,coords,flat,alternative=exact_problem(train,v,wo,initial)
   torch.cuda.synchronize();exact_seconds=time.monotonic()-exact_started
   count=train[0].shape[1]*wo.shape[0]
   artifact=DATA/f'p{prompt:02d}_l{layer:02d}_exact.pt'
   torch.save(dict(**{k:v.cpu() for k,v in coeff.items()},d_values=enum['d_values'].cpu(),j_values=enum['j_values'].cpu(),
      coords=coords.cpu(),flat_indices=flat.cpu(),alternative_codes=alternative.cpu(),metric_denominator=count),artifact)
   for selector,key in (('mean','best_mean_index'),('transition','best_transition_index')):
    index=int(enum[key]);bits=torch.tensor([(index>>j)&1 for j in range(16)],device='cuda',dtype=torch.bool)
    state=copy_state(initial);chosen_coords=coords[bits]
    if chosen_coords.numel():state.codes[chosen_coords[:,0],chosen_coords[:,1],chosen_coords[:,2]]=alternative[bits]
    record('exact16',selector,state,index,dict(quadratic_train_d=float(enum['d_values'][index])/count,
       quadratic_train_j=float(enum['j_values'][index])/count))
   cells.append(dict(prompt=prompt,layer=layer,pool_slots=97,unique_pool_codes=len({r['code_hash'] for r in candidates}),
      common_pool_seconds=pool_seconds,exact_seconds=exact_seconds,exact_patterns=65536,exact_file=artifact.name,exact_sha256=sha(artifact),
      selected_coordinates=coords.cpu().tolist(),baseline_quadratic_d=float(enum['d_values'][0])/count,
      baseline_quadratic_j=float(enum['j_values'][0])/count))
   del data,later,train,evaluation,v,wo,initial,result,legacy,states,coeff,enum
   torch.cuda.empty_cache()
 if len(pool)!=582 or len(selections)!=24 or len(parities)!=18 or fingerprint()!=source:raise RuntimeError('incomplete or changed source')
 decisions=[]
 for space in ('common_pool','exact16'):
  for prompt in (0,4):
   means={selector:{key:statistics.mean(float(r['eval_'+key]) for r in selections if r['space']==space and r['prompt']==prompt and r['selector']==selector)
                    for key in ('d','j')} for selector in ('mean','transition')}
   b=means['mean'];c=means['transition'];valid=b['j']>1e-12*max(b['d'],1e-30)
   reduction=1-c['j']/b['j'] if valid else None;damage=c['d']/b['d'] if b['d'] else None
   decisions.append(dict(space=space,prompt=prompt,means=means,baseline_jump_valid=valid,jump_reduction=reduction,mean_damage_ratio=damage,
      passes=valid and reduction>=.1 and damage is not None and damage<=1.02))
 advance=any(all(d['passes'] for d in decisions if d['space']==space) for space in ('common_pool','exact16'))
 OUT.mkdir(parents=True)
 for name,rows in (('pool_candidates',pool),('selections',selections)):
  with (OUT/f'{name}.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save(OUT/'decision.json',dict(source_sha256=source,advance=advance,decisions=decisions,cells=cells,trace_parity=parities,
    captures=input_artifacts,legacy_decision_sha256=sha(ROOT/'src/experiments/postwo_rounding_development/decision.json'),
    seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    scope='reused development data; shared pool and chosen D-selected16bit block; no global optimum/video claim'))
 status('COMPLETE '+('GO' if advance else 'NO-GO'),'pass');print(json.dumps(decisions,indent=2),flush=True)

if __name__=='__main__':
 torch.set_grad_enabled(False);torch.set_num_threads(4)
 try:main()
 except Exception as exc:status(str(exc),'failed');raise
