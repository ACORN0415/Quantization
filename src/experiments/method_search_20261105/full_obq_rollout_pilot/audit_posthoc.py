"""Independent read-only audit, including partially completed pilots.

Does not import a worker, run CUDA, regenerate noise, quantize, decode or score
images again. Default output is stdout; --out uses exclusive creation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import argparse,hashlib,json,math,statistics,time
from pathlib import Path
import torch
torch.set_num_threads(4)
HERE=Path(__file__).resolve().parent;ARMS=('BF16','R_bf16','R_fp32','O_full_gpu')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(path.read_text()) if path.exists() else None
def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);args=ap.parse_args()
 if args.out:assert not args.out.exists(),'Refuse overwrite audit output'
 start=time.monotonic();errors=[];checks=[];protected={};missing=[]
 def check(condition,label):
  checks.append({'label':label,'pass':bool(condition)})
  if not condition:errors.append(label)
 def protect(path,digest=None):
  path=Path(path)
  if not path.exists():missing.append(str(path));check(False,'missing referenced artifact '+str(path));return False
  actual=sha(path);protected[str(path)]=actual
  check(digest is None or actual==digest,'sha '+str(path));return digest is None or actual==digest
 run=read(HERE/'RUN.json');lock=read(HERE/'LOCK.json');quality=read(HERE/'QUALITY.json');counts=read(HERE/'COUNTERS.json')
 for name in ('RUN.json','LOCK.json','QUALITY.json','COUNTERS.json','PROMPTS.json','FACTORS.json','INTEGRATION.json'):
  if (HERE/name).exists():protect(HERE/name)
 if lock:
  for path,digest in lock['sha256'].items():protect(path,digest)
 inputs=read(HERE/'PROMPTS.json')['inputs'];completed=[];latent_scores={};sampling={};features={}
 for item in inputs:
  reference=None;noise=None
  for arm in ARMS:
   folder=HERE/item['id']/arm;meta=read(folder/'sampling.json')
   if meta is None:missing.append(str(folder/'sampling.json'));continue
   protect(folder/'sampling.json');sampling[(item['id'],arm)]=meta
   check(meta['model_forwards']<=210,f'{item["id"]}/{arm}: forward cap')
   if meta['status']!='completed':continue
   completed.append((item['id'],arm));check(meta['model_forwards']==210,'completed 210 forwards')
   check(len(meta['chunks'])==42,'completed 42 chunks')
   check(len(meta['initial_noise_chunk_sha256'])==42,'42 stored initial-noise hashes')
   if arm=='BF16':noise=meta['initial_noise_chunk_sha256']
   elif noise is not None:check(meta['initial_noise_chunk_sha256']==noise,'same 42 initial-noise hashes '+item['id']+'/'+arm)
   expected_counts=[0]*30 if arm=='BF16' else [41]*30
   check(meta['layer_encode_pair_calls']==expected_counts,'layer encode counts '+arm)
   if not protect(meta['events']['path'],meta['events']['sha256']):continue
   events=[json.loads(x) for x in Path(meta['events']['path']).read_text().splitlines() if x.strip()]
   expected=[] if arm=='BF16' else [(c,l) for c in range(1,42) for l in range(30)]
   check([(x['reader_chunk'],x['layer']) for x in events]==expected,'chronological actual encoder events '+item['id']+'/'+arm)
   check(all(x['source_chunk']==x['reader_chunk']-1 and x['actual_encode_pair_count']==x['reader_chunk'] and x['K_bytes']==4515841 and x['V_bytes']==4500480 for x in events),'event source/count/payload '+arm)
   if arm!='BF16':
    check(all(x['past_KV_payload_bytes']==min(x['chunk'],6)*30*9016321 for x in meta['chunks']),'reported cache payload arithmetic '+arm)
   info=meta['latent']
   if not protect(info['path'],info['sha256']):continue
   check(Path(info['path']).stat().st_size==info['bytes'],'latent size '+arm)
   latent=torch.load(info['path'],map_location='cpu',weights_only=True)['latent']
   check(tuple(latent.shape)==(1,126,16,60,104),'latent shape '+arm);check(torch.isfinite(latent).all(),'finite latent '+arm)
   if arm=='BF16':reference=latent
   if reference is not None:
    check(same(latent[:,:3],reference[:,:3]),'actual first chunk bytes '+item['id']+'/'+arm)
    for name,a,b in (('late_latent',63,None),('whole_latent',0,None)):
     x=latent[:,a:b].double();ref=reference[:,a:b].double();sse=float((x-ref).square().sum());energy=float(ref.square().sum());nse=sse/energy
     check(close(sse,meta[name]['sse']) and close(energy,meta[name]['reference_energy']) and close(nse,meta[name]['nse']),'recomputed '+name+' '+item['id']+'/'+arm)
     if name=='late_latent':latent_scores[(item['id'],arm)]=nse
    for name,a,b in (('chunks_1_to_7',3,24),('chunks_8_to_20',24,63),('chunks_21_to_41',63,126)):
     x=latent[:,a:b].double();ref=reference[:,a:b].double();value=float((x-ref).square().sum()/ref.square().sum())
     check(close(value,meta['descriptive_latent_windows'][name]['nse']),'descriptive window '+name+' '+arm)
   del latent
  del reference
 score_rows={}
 if quality:
  for row in quality['rows']:
   key=(row['input_id'],row['arm']);score_rows[key]=row
   info=row['features']
   if not protect(info['path'],info['sha256']):continue
   feature=torch.load(info['path'],map_location='cpu',weights_only=True)['features']
   features[key]=feature;check(feature.shape[0]==501,'501 DINO features')
   check(len(row['musiq_per_frame'])==501 and all(math.isfinite(x) for x in row['musiq_per_frame']),'501 finite MUSIQ scores')
   for name,start_index in (('whole',0),('late',249)):
    check(close(statistics.mean(row['musiq_per_frame'][start_index:]),row['windows'][name]['musiq']),'saved MUSIQ mean '+str(key)+name)
   if key in latent_scores:check(close(latent_scores[key],row['late_latent']['nse']),'quality latent score matches artifact '+str(key))
   if 'preview' in row:protect(row['preview']['path'],row['preview']['sha256'])
  for key,feature in features.items():
   ref=features.get((key[0],'BF16'))
   if ref is None:continue
   a=torch.nn.functional.normalize(feature.double(),dim=1);b=torch.nn.functional.normalize(ref.double(),dim=1)
   sim=(a*b).sum(1).clamp(-1,1);row=score_rows[key]
   check(all(close(a,b) for a,b in zip(sim.tolist(),row['dino_reference_per_frame'])),'DINO per-frame from saved features '+str(key))
   for name,start_index in (('whole',0),('late',249)):
    check(close(float(sim[start_index:].mean()),row['windows'][name]['dino_reference']),'DINO window mean '+str(key)+name)
 decision=None
 if quality and quality['status']=='completed' and len(score_rows)==8 and len(latent_scores)==8:
  signal=True;bad=False;comparisons=[]
  for base in ('R_bf16','R_fp32'):
   dino=[];musiq=[]
   for item in inputs:
    f=score_rows[(item['id'],'O_full_gpu')];b=score_rows[(item['id'],base)]
    den=latent_scores[(item['id'],base)];gain=1-latent_scores[(item['id'],'O_full_gpu')]/den if den>0 else None
    dd=f['windows']['late']['dino_reference']-b['windows']['late']['dino_reference'];dm=f['windows']['late']['musiq']-b['windows']['late']['musiq']
    signal &= gain is not None and gain>=.1 and dd>=-.005 and dm>=-1
    bad |= (gain is not None and gain<=-.1) or dd<-.02 or dm<-2
    dino.append(dd);musiq.append(dm);comparisons.append({'input_id':item['id'],'baseline':base,'gain':gain,'DINO_delta':dd,'MUSIQ_delta':dm})
   signal &= sum(dino)/2>=0 and sum(musiq)/2>=-.5
  decision={'decision':'signal' if signal else ('stop_quality' if bad else 'undecided'),'comparisons':comparisons}
  check(decision['decision']==quality['decision']['decision'],'independent combined gate')
 if counts:
  check(counts['generations']<=8 and counts['vae_decodes']<=8,'global attempt caps')
  check(counts['generations']>=len(completed),'completed generations do not exceed attempted count')
 if run and run.get('status')=='completed':
  check(len(completed)==8 and quality is not None and quality['status']=='completed','complete run has all eight generations and quality')
  check(counts=={'generations':8,'vae_decodes':8},'completed run exact counters')
  check(run['elapsed_seconds']<=3600,'completed run wall cap')
 for path,digest in protected.items():check(sha(path)==digest,'read-only preserved '+path)
 output={'status':'invalid_evidence' if errors else ('not_started' if run is None else 'completed_audit' if run.get('status')=='completed' else 'partial_audit'),
  'run_status':None if run is None else run['status'],'completed_generations':len(completed),'protected_files':len(protected),
  'checks':checks,'errors':errors,'missing_or_not_yet_produced':missing,'independent_decision':decision,
  'scope':'Actual saved latent first-chunk bytes and NSE; stored noise-hash equality; recorded encode chronology/payload; stored DINO features and MUSIQ means/gates. No noise regeneration, new encoder/attention/decoder/model/GPU or raw-image metric rerun; packet-byte evidence at rollout is logged, not a saved full cache.',
  'GPU_calls':0,'elapsed_seconds':time.monotonic()-start,'auditor_sha256':sha(__file__)}
 text=json.dumps(output,indent=2)+'\n'
 if args.out:
  with args.out.open('x') as f:f.write(text)
 else:print(text,end='')
if __name__=='__main__':main()
