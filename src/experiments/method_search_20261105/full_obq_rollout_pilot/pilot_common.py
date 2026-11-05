"""Pilot contracts. Importing this module does not load a model or use CUDA."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import hashlib,importlib.util,json,math,resource,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
import torch
torch.set_num_threads(4);torch.set_grad_enabled(False)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
CAL=Path('/home/acorn0415/sfq_data/kv_alllayer_capture/p00_s101')
ARMS=('BF16','R_bf16','R_fp32','O_full_gpu')
ORDERS=(ARMS,('BF16','O_full_gpu','R_fp32','R_bf16'))
PREVIEW_API=HERE.parent/'obq_gpu_encoder_stage2_layoutfix/experiment.py'

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def save(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def record(path):return {'path':str(path),'sha256':sha(path),'bytes':Path(path).stat().st_size}
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def tsha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def nbytes(packet):return sum(t.numel()*t.element_size() for t in packet.values() if isinstance(t,torch.Tensor))
def prompts():return json.loads((HERE/'PROMPTS.json').read_text())['inputs']
def guard():
 r=json.loads((HERE/'RUN.json').read_text())
 if time.monotonic()-r['started_monotonic']>3600:raise TimeoutError('60-minute complete pilot budget')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>40*1024**3:raise MemoryError('40GiB RSS budget')
 if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>22*1024**3:raise MemoryError('22GiB observed GPU allocated peak budget')
 if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())>4*1024**3:raise OSError('4GiB new disk budget')
def count(kind):
 guard();p=HERE/'COUNTERS.json';c=json.loads(p.read_text());c[kind]+=1
 if c[kind]>8:raise RuntimeError(f'{kind} exceeds eight')
 save(p,c)
def validate_review_status(record):
 # Administrative authorization and actual Claude review are different facts.
 for key in ('root_execution_authorized','root_code_review_accepted','independent_codex_code_review_accepted',
             'root_stage2_accepted','cost_review_accepted','historical_fable_design_reviewed'):
  assert record.get(key) is True,key
 assert record.get('review_route')=='root_accepted_deferred_review'
 assert record.get('review_status')=='deferred_not_completed'
 assert record.get('claude_final_source_reviewed') is False
 assert record.get('claude_final_result_reviewed') is False
 assert record.get('final_reviewer_model')=='Opus 5.5'
 assert record.get('claude_not_before')=='2026-11-05T21:50:00+09:00'
 assert record.get('administrative_amendment_adopted') is True
def binding():
 b=json.loads((HERE/'STAGE2_BINDING.json').read_text())
 assert b.get('status')=='ready', 'Final stage2 source/result binding is pending; run forbidden'
 validate_review_status(b)
 for key in ('api_source','extension_binary','quality_result','stage2_result'):assert b[key] in b['sha256'],key
 for path,h in b['sha256'].items():guard();assert sha(path)==h,path
 q=json.loads(Path(b['quality_result']).read_text());assert q['status']=='completed' and q['quality_gate'] is True
 assert len(q['endpoints'])==4
 assert all(row['native_replay_byte_equal'] and all(row['gains'][a]>=.2 for a in ('R_bf16','R_fp32')) for row in q['endpoints'])
 assert json.loads(Path(b['stage2_result']).read_text())['status']=='completed'
 return b
def verify_lock():
 guard();lock=json.loads((HERE/'LOCK.json').read_text())
 validate_review_status(lock)
 for source in required_sources():assert str(source) in lock['sha256'],f'Unbound required source: {source}'
 for path,h in lock['sha256'].items():guard();assert sha(path)==h,path
 binding();return lock
def required_sources():
 files=[HERE/name for name in ('pilot_common.py','packed_adapter.py','worker.py','quality.py','supervisor.py',
  'selftest.py','SELFTEST_V3.json','PROMPTS.json','PROMPT_HISTORY_CHECK.json','REGISTRATION_DRAFT_KO.txt','STAGE2_BINDING.json',
  'ADMINISTRATIVE_AMENDMENT_KO.txt')]
 files+=[ROOT/name for name in ('src/scripts/p3_tf_fr.py','src/scripts/anchor_identity_gate.py',
  'src/experiments/P3_TF_FR_PILOT.md','src/configs/default_config.yaml','src/configs/a1.yaml',
  'tools/kv_channel_rollout.py','tools/kv_channel_full_attention.py','tools/mechanism_worker.py',
  'tools/evaluate_policy_supplement.py','src/experiments/kv_channel_budget_rollout/quality.json')]
 for folder in ('kvq','pipeline','wan','utils'):files+=sorted((ROOT/'src'/folder).rglob('*.py'))
 files += [CAL/'manifest.json',HERE.parent/'causal_obq/capture/manifest.json',HERE.parent/'causal_obq/packets/manifest.json']
 files.append(ROOT/'src/checkpoints/self_forcing_dmd.pt')
 weights=ROOT/'src/wan_models/Wan2.1-T2V-1.3B'
 files += [weights/name for name in ('config.json','diffusion_pytorch_model.safetensors','models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth')]
 files += sorted((weights/'google/umt5-xxl').glob('*'))
 prior=json.loads((ROOT/'src/experiments/kv_channel_budget_rollout/quality.json').read_text())
 files += [Path(info['path']) for info in prior['weights'].values()]
 return files
def load_api(*,execution=False,kernel=False):
 b=binding() if execution else None
 path=Path(b['api_source']) if b else PREVIEW_API
 spec=importlib.util.spec_from_file_location('bound_obq_stage2',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 if kernel:
  assert execution and b is not None
  binary=Path(b['extension_binary']);assert str(binary) in b['sha256']
  spec=importlib.util.spec_from_file_location(binary.name.split('.')[0],binary)
  ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext);module.EXT=ext
 return module
def verified(path,manifest):
 guard();row=next(a for a in manifest['artifacts'] if Path(a['path']).name==Path(path).name)
 assert sha(path)==row['sha256'] and Path(path).stat().st_size==row['bytes']
 return torch.load(path,map_location='cpu',weights_only=True)
def cuda_settings():
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.set_float32_matmul_precision('highest')
def peak():
 r={'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 if torch.cuda.is_initialized():r.update(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
 return r
def latent_metric(x,reference,start=63,end=None):
 a=x[:,start:end].double();b=reference[:,start:end].double();sse=float((a-b).square().sum());energy=float(b.square().sum())
 assert math.isfinite(sse) and energy>0 and math.isfinite(energy)
 return {'sse':sse,'reference_energy':energy,'nse':sse/energy,'start_latent_frame':start,'end_latent_frame_exclusive':end}
def decide(rows):
 """Two inputs and both RTNs; fidelity proxy plus separate image safeguards."""
 assert len(rows)==8
 lookup={(r['input_id'],r['arm']):r for r in rows};details=[];signal=True;bad=False
 for base in ('R_bf16','R_fp32'):
  dm=[];dd=[]
  for item in prompts():
   f=lookup[(item['id'],'O_full_gpu')];b=lookup[(item['id'],base)]
   den=b['late_latent']['nse'];gain=1-f['late_latent']['nse']/den if den>0 else None
   m=f['windows']['late']['musiq']-b['windows']['late']['musiq']
   d=f['windows']['late']['dino_reference']-b['windows']['late']['dino_reference']
   assert math.isfinite(m) and math.isfinite(d)
   signal &= gain is not None and gain>=.1 and d>=-.005 and m>=-1.
   bad |= (gain is not None and gain<=-.1) or d<-.02 or m<-2.
   dm.append(m);dd.append(d);details.append({'input_id':item['id'],'baseline':base,'latent_NSE_reduction':gain,'DINO_delta':d,'MUSIQ_delta':m})
  signal &= sum(dm)/2>=-.5 and sum(dd)/2>=0
 return {'decision':'signal' if signal else ('stop_quality' if bad else 'undecided'),'comparisons':details,
  'latent_interpretation':'BF16 trajectory fidelity proxy, not video quality or a ground-truth target'}
