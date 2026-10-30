"""Fixed-rank extension; reuse frozen trajectory implementation without editing it."""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import torch
import tt_video_run as base
from tt_rank_codec import FixedRankVCodec
ROOT=base.ROOT
DATA=Path.home()/'sfq_data/tt_rank_sweep'
OLD=Path.home()/'sfq_data/tt_video'
RANKS=(64,128,192)
CASES=[(p,b,r) for b in (2,3,4) for r in RANKS for p in (0,4)]
FILES=tuple(base.FILES)+('tools/tt_rank_run.py','tools/tt_rank_codec.py','tools/tt_rank_quality.py','tools/test_tt_rank_codec.py','tools/test_tt_rank_quality.py','tools/report_tt_video.py','src/experiments/TT_RANK_SWEEP.md')

def key(p,b,r):return f'p{p}_b{b}_r{r}'
def signature():
 sources={p:base.util.sha(ROOT/p) for p in FILES}
 return hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),sources

def status(message,**extra):
 extra.pop('total',None);extra.pop('completed',None)
 base.status(message,completed=sum((DATA/key(*c)/'sampling.json').exists() for c in CASES),total=18,**extra)

original_status=base.status

def configure():
 # base.trajectory resolves these module-level helpers at runtime.
 base.DATA=DATA
 base.status=lambda message,**extra: original_status(message,**dict(extra,completed=sum((DATA/key(*c)/'sampling.json').exists() for c in CASES),total=18))
 base.key=lambda p,b,a: key(p,b,CURRENT_RANK)
 base.keys=lambda: [(p,b,r) for p,b,r in CASES]

CURRENT_RANK=64

def run(pipe,cond,p,b,r,n,source,reference,folder=None,identity=False):
 global CURRENT_RANK
 CURRENT_RANK=r
 base.VCodec=(lambda arm,bits: OriginalCodec('identity',bits)) if identity else (lambda arm,bits: FixedRankVCodec(bits,r))
 result=base.trajectory(pipe,cond,p,b,'identity' if identity else 'tt_quant',n,source,reference,folder,base.pilot.source_hash())
 if folder:
  result.update(rank=r,bond_ranks=[r,min(r,128)])
  base.util.save(folder/'sampling.json',result)
 return result
OriginalCodec=base.VCodec

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['all','gate','evaluate'],default='all');args=ap.parse_args()
 os.chdir(ROOT/'src');torch.set_grad_enabled(False);torch.set_num_threads(4);base.util.free_gpu();DATA.mkdir(parents=True,exist_ok=True)
 configure();sig,sources=signature()
 # Validate the complete previous run before using its numerical baselines.
 old=json.loads((OLD/'manifest.json').read_text())
 for p,h in old['source_files'].items():
  if base.util.sha(ROOT/p)!=h:raise RuntimeError('frozen baseline source changed: '+p)
 artifacts={}
 for p in (0,4):
  for b,a in [(0,'bf16')]+[(b,a) for b in (2,3,4) for a in ('quant','tt_quant')]:
   d=OLD/f'p{p}_b{b}_{a}';s=base.check_done(d,old['source_sha256'])
   if s is None:raise RuntimeError('missing baseline')
   for name in ('latent.pt','sampling.json','quality.json'):
    artifacts[f'{d.name}/{name}']=base.util.sha(d/name)
 current=dict(source_sha256=sig,source_files=sources,model_source_sha256=base.pilot.source_hash(),baseline_manifest_sha256=base.util.sha(OLD/'manifest.json'),baseline_artifact_sha256=artifacts,checkpoint_sha256=base.util.sha(ROOT/'src/checkpoints/self_forcing_dmd.pt'),text_encoder_sha256=base.util.sha(ROOT/'src/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth'),prompt_sha256=base.util.sha(ROOT/'src/experiments/policy_study_prompts.txt'),cases=[key(*c) for c in CASES])
 for new_key,old_key in [('checkpoint_sha256','checkpoint_sha256'),('text_encoder_sha256','text_encoder_sha256'),('prompt_sha256','prompt_file_sha256'),('model_source_sha256','model_source_sha256')]:
  if current[new_key]!=old[old_key]:raise RuntimeError('baseline input changed: '+new_key)
 mp=DATA/'manifest.json'
 if mp.exists():
  if json.loads(mp.read_text())!=current:raise RuntimeError('manifest changed; refuse mixed resume')
 else:base.util.save(mp,current)
 if args.mode!='evaluate' and (not (DATA/'gates.json').exists() or any(base.check_done(DATA/key(*c),sig) is None for c in CASES)):
  status('loading model',phase='loading')
  pipe=base.pilot.build(101);texts=(ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines();conds=dict(zip((0,4),base.pilot.encode(pipe,[texts[p] for p in (0,4)])))
  refs={p:torch.load(OLD/f'p{p}_b0_bf16/latent.pt',map_location='cpu',weights_only=True)['latent'] for p in (0,4)}
  if not (DATA/'gates.json').exists():
   records=[run(pipe,conds[0],0,0,64,8,sig,refs[0],identity=True)]
   for b in (2,3,4):
    ref=torch.load(OLD/f'p0_b{b}_tt_quant/latent.pt',map_location='cpu',weights_only=True)['latent']
    rec=run(pipe,conds[0],0,b,299,8,sig,ref)
    if any(row['latent_nmse']!=0 for row in rec['chunks']):raise RuntimeError('rank299 old trajectory mismatch')
    records.append(rec)
    for r in (64,192):records.append(run(pipe,conds[0],0,b,r,8,sig,refs[0]))
   if signature()[0]!=sig:raise RuntimeError('source changed in gates')
   base.util.save(DATA/'gates.json',dict(source_sha256=sig,pass_all=True,records=records))
   status('GATES PASS',phase='gates_pass')
  else:
   g=json.loads((DATA/'gates.json').read_text())
   if g['source_sha256']!=sig or g['pass_all'] is not True or len(g['records'])!=10:raise RuntimeError('invalid gates')
  if args.mode!='gate':
   for p,b,r in CASES:
    folder=DATA/key(p,b,r)
    if base.check_done(folder,sig) is not None:status('REUSE '+folder.name);continue
    run(pipe,conds[p],p,b,r,42,sig,refs[p],folder)
    if signature()[0]!=sig:raise RuntimeError('measurement sources changed')
    status('COMPLETE sampling '+folder.name,phase='sampling')
  del pipe,conds,refs;gc.collect();torch.cuda.empty_cache()
 if args.mode in ('all','evaluate'):
  if any(base.check_done(DATA/key(*c),sig) is None for c in CASES):raise RuntimeError('incomplete generation')
  from tt_rank_quality import evaluate_all,report
  evaluate_all(sig,base.status)
  if signature()[0]!=sig:raise RuntimeError('source changed during evaluation')
  report()
  status('COMPLETE 18/18 generation, quality and report',phase='complete')

if __name__=='__main__':
 try:main()
 except Exception as exc:
  if DATA.exists():status(f'FAILED: {type(exc).__name__}: {exc}',phase='failed')
  raise
