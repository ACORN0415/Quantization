"""Resumable 26-video TT/quantization development queue; cwd is set to src."""
import argparse,gc,json,math,os,sys,time,hashlib,subprocess
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/scripts')]
import torch
import p3_tf_fr as pilot
import postwo_round_probe as util
from tt_video_codec import VCodec,cache_bytes,history_error
DATA=Path.home()/'sfq_data/tt_video';OUT=ROOT/'src/experiments/tt_video_development'
ARMS=('tt','quant','quant_tt','tt_quant');PROMPTS=(0,4);BITS=(2,3,4)
FILES=('tools/tt_video_run.py','tools/tt_video_codec.py','tools/tt_video_quality.py',
 'tools/codec_order_core.py','tools/change_mixed_core.py','src/kvq/store.py',
 'src/kvq/lockstep.py','tools/test_tt_video_codec.py','tools/test_tt_video_quality.py',
 'tools/evaluate_policy_supplement.py','tools/mechanism_worker.py',
 'src/experiments/TT_VIDEO_DEVELOPMENT.md')

def status(message,**extra):
 stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
 print(stamp,message,flush=True)
 util.save(DATA/'status.json',dict(detail=message,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**extra))

def signatures():
 sources={p:util.sha(ROOT/p) for p in FILES}
 signature=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
 return signature,sources

def keys():
 return [(p,0,'bf16') for p in PROMPTS]+[(p,b,a) for b in BITS for p in PROMPTS for a in ARMS]

def key(p,b,a):return f'p{p}_b{b}_{a}'

def check_done(folder,signature):
 path=folder/'sampling.json'
 if not path.exists():return None
 result=json.loads(path.read_text())
 if result['source_sha256']!=signature or result['latent_sha256']!=util.sha(folder/'latent.pt'):
  raise RuntimeError(f'completed case source/latent mismatch: {folder.name}')
 if len(result['chunks'])!=42:raise RuntimeError('incomplete sampling record')
 return result

def trajectory(pipe,cond,prompt,bits,arm,nchunks,signature,reference,folder=None,model_hash=None):
 codec=None if arm=='bf16' else VCodec(arm,bits)
 noise=pilot.chunk_noise(pipe,126,101,pilot.gpu)
 cache=pilot.bf16_cache(pipe,pilot.gpu) if codec is None else pilot.compressed_cache(pipe,codec,pilot.gpu)
 xc=pilot.cross_cache(pipe,pilot.gpu)
 values=[];rows=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 started=time.perf_counter()
 for c in range(nchunks):
  torch.cuda.synchronize();begin=time.perf_counter()
  y=pilot.run_chunk(pipe,cache,xc,cond,noise,c,101)
  torch.cuda.synchronize();seconds=time.perf_counter()-begin
  if not bool(torch.isfinite(y).all()):raise RuntimeError(f'nonfinite {prompt,bits,arm,c}')
  cpu=y.cpu();target=reference[:,c*3:(c+1)*3]
  if arm in ('identity','bf16') or c==0:
   if not torch.equal(cpu,target):raise RuntimeError(f'identity failure {prompt,bits,arm,c}')
  error=(cpu.double()-target.double());nmse=float(error.square().sum()/target.double().square().sum().clamp_min(1e-30))
  metrics=codec.drain() if codec else dict(encode_seconds=0.,restore_seconds=0.,encode_events=0,restore_events=0)
  row=dict(chunk=c,seconds=seconds,latent_nmse=nmse,latent_rel=math.sqrt(nmse),history_v_rel=history_error(cache),cache=cache_bytes(cache),gpu_allocated_bytes_after=torch.cuda.memory_allocated(),**metrics)
  rows.append(row);values.append(cpu)
  status(f'{key(prompt,bits,arm)} chunk {c+1}/{nchunks}, {seconds:.2f}s, latent_rel={row["latent_rel"]:.5g}',phase='sampling' if folder else 'gates',case=key(prompt,bits,arm),chunk=c+1,chunks=nchunks,completed=sum((DATA/key(*k)/'sampling.json').exists() for k in keys()),total=26)
 elapsed=time.perf_counter()-started
 latent=torch.cat(values,1)
 ref=reference[:,:nchunks*3]
 def nmse_part(lo):
  x=latent[:,lo:].double();r=ref[:,lo:].double()
  return float((x-r).square().sum()/r.square().sum().clamp_min(1e-30))
 result=dict(key=key(prompt,bits,arm),prompt=prompt,bits=bits,arm=arm,seed=101,frames=nchunks*3,
  source_sha256=signature,model_source_sha256=model_hash,chunks=rows,sampling_seconds=elapsed,
  sum_chunk_seconds=sum(r['seconds'] for r in rows),generation_seconds=sum(r['seconds'] for r in rows),
  generation_latent_frames_per_second=nchunks*3/sum(r['seconds'] for r in rows),
  generation_pixel_frames_per_second=(nchunks*12-3)/sum(r['seconds'] for r in rows),
  latent_frames_per_second=nchunks*3/elapsed,
  encode_seconds=sum(r['encode_seconds'] for r in rows),restore_seconds=sum(r['restore_seconds'] for r in rows),
  encode_calls=codec.encode_calls if codec else 0,ranks=sorted(codec.ranks) if codec else [],
  latent_nmse_whole=nmse_part(0),latent_nmse_late=nmse_part(63) if nchunks==42 else None,
  peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
 if codec and codec.encode_calls!=30*(nchunks-1):raise RuntimeError('once-per-finalized-chunk encoding count failed')
 if folder:
  folder.mkdir(parents=True,exist_ok=True);tmp=folder/'latent.pt.tmp'
  torch.save(dict(latent=latent),tmp);tmp.replace(folder/'latent.pt')
  result['latent_sha256']=util.sha(folder/'latent.pt');util.save(folder/'sampling.json',result)
 del cache,xc,codec,noise,y,cpu,latent,values
 gc.collect();torch.cuda.empty_cache()
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['all','gate','sample','evaluate'],default='all');args=ap.parse_args()
 os.chdir(ROOT/'src');torch.set_grad_enabled(False);torch.set_num_threads(4)
 util.free_gpu();DATA.mkdir(parents=True,exist_ok=True)
 signature,sources=signatures();model_hash=pilot.source_hash()
 manifest_path=DATA/'manifest.json'
 artifacts={'checkpoint_sha256':util.sha(ROOT/'src/checkpoints/self_forcing_dmd.pt'),
  'prompt_file_sha256':util.sha(ROOT/'src/experiments/policy_study_prompts.txt'),
  'reference_sha256':{str(p):util.sha(Path.home()/f'sfq_data/policy_study/reference/p{p:02d}_s101/bf16.pt') for p in PROMPTS},
  'text_encoder_sha256':util.sha(ROOT/'src/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth')}

 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text())
  if manifest['source_sha256']!=signature or manifest['model_source_sha256']!=model_hash:raise RuntimeError('run source changed; use a new protocol/data version')
  if any(manifest.get(k)!=v for k,v in artifacts.items()):raise RuntimeError('checkpoint/prompt/reference changed; refuse mixed resume')
 else:
  manifest=dict(source_sha256=signature,source_files=sources,model_source_sha256=model_hash,
   **artifacts,cases=[key(*k) for k in keys()],created_utc=datetime.now(timezone.utc).isoformat())
  util.save(manifest_path,manifest)
 if args.mode!='evaluate':
  need_gate=not (DATA/'gates.json').exists()
  need_sample=any(check_done(DATA/key(*k),signature) is None for k in keys())
  if need_gate or (need_sample and args.mode!='gate'):
   status('loading Self-Forcing/Wan model',phase='loading',completed=0,total=26)
   load_start=time.perf_counter();pipe=pilot.build(101)
   texts=(ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines()
   conds=dict(zip(PROMPTS,pilot.encode(pipe,[texts[p] for p in PROMPTS])))
   model_loading_seconds=time.perf_counter()-load_start
   expected={p:torch.load(Path.home()/f'sfq_data/policy_study/reference/p{p:02d}_s101/bf16.pt',map_location='cpu',weights_only=True)['latent'] for p in PROMPTS}
   if need_gate:
    records=[]
    for p in PROMPTS:records.append(trajectory(pipe,conds[p],p,0,'identity',8,signature,expected[p],model_hash=model_hash))
    for a in ARMS:records.append(trajectory(pipe,conds[0],0,2,a,8,signature,expected[0],model_hash=model_hash))
    if signature!=signatures()[0]:raise RuntimeError('sources changed during gates')
    util.save(DATA/'gates.json',dict(source_sha256=signature,pass_all=True,records=records,model_loading_seconds=model_loading_seconds))
    predictions={a:42*sum(r['seconds'] for r in next(g for g in records if g['arm']==a)['chunks'][-4:])/4 for a in ARMS}
    util.save(DATA/'estimate.json',dict(per_case_seconds_from_full_window=predictions,generation_seconds_24_cases=6*sum(predictions.values()),quality_time_not_included=True))
    status('GATES PASS; launching complete 26-video queue',phase='gates_pass',estimate=predictions)
   else:
    gate=json.loads((DATA/'gates.json').read_text())
    expected_gates={(p,0,'identity') for p in PROMPTS}|{(0,2,a) for a in ARMS}
    actual_gates={(r['prompt'],r['bits'],r['arm']) for r in gate['records']}
    if gate['source_sha256']!=signature or gate.get('pass_all') is not True or actual_gates!=expected_gates or any(len(r['chunks'])!=8 for r in gate['records']):raise RuntimeError('gate source/completeness mismatch')
   if args.mode!='gate':
    for p,b,a in keys():
     folder=DATA/key(p,b,a)
     if check_done(folder,signature) is not None:status(f'REUSE {folder.name}');continue
     reference=expected[p] if a=='bf16' else torch.load(DATA/key(p,0,'bf16')/'latent.pt',map_location='cpu',weights_only=True)['latent']
     trajectory(pipe,conds[p],p,b,a,42,signature,reference,folder,model_hash)
     if signature!=signatures()[0] or model_hash!=pilot.source_hash():raise RuntimeError('sources changed during sampling')
     status(f'COMPLETE sampling {folder.name}',phase='sampling',completed=sum((DATA/key(*k)/'sampling.json').exists() for k in keys()),total=26)
   del pipe,conds,expected
   gc.collect();torch.cuda.empty_cache()
 if args.mode in ('all','evaluate'):
  for item in keys():
   if check_done(DATA/key(*item),signature) is None:raise RuntimeError('cannot evaluate incomplete generation queue')
  from tt_video_quality import evaluate_all
  evaluate_all(signature,status)
  if signature!=signatures()[0]:raise RuntimeError('sources changed during quality')
  subprocess.run([sys.executable,str(ROOT/'tools/report_tt_video.py')],check=True)
  status('COMPLETE generation, quality, audited report: 26/26',phase='complete',completed=26,total=26)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  if DATA.exists():status(f'FAILED: {type(exc).__name__}: {exc}',phase='failed')
  raise
