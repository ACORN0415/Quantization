"""Read-only artifact/decision audit. CPU tensor inspection only; no quantization/attention replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,math,statistics
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;ROOT=P.parents[3];SRC=ROOT/'src'
torch.set_num_threads(4);torch.set_grad_enabled(False)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def tsha(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def read(p):return json.loads(p.read_text())
lock=read(P/'LOCK.json');run=read(P/'RUN.json');driver=read(P/'DRIVER_STATUS.json');res=read(P/'NATIVE_RESULT.json')
ms={x:read(P/x/'manifest.json') for x in ('reference','capture','packets')}
assert driver['status']==res['status']=='completed'
assert sha(P/'NATIVE_RESULT.json')=='1b3e932f350978a96e891356282e3818a8e1546c9030f038f3bee0ec3cd2b2b9'==driver['native_result_sha256']
assert sha(P/'packets/manifest.json')==res['packet_manifest_sha256']
for name in ('sha256','review_sha256'):
 for path,h in lock[name].items():assert sha(path)==h,path
h=hashlib.sha256()
for folder in ('kvq','pipeline','wan','utils'):
 for f in sorted((SRC/folder).rglob('*.py')):h.update(f.read_bytes())
for f in ('scripts/p3_tf_fr.py','scripts/anchor_identity_gate.py','experiments/P3_TF_FR_PILOT.md'):h.update((SRC/f).read_bytes())
assert h.hexdigest()==ms['reference']['source_sha256']==ms['capture']['source_sha256']
assert lock['registered_utc']<driver['experiment_started_utc']<=run['started_utc']<ms['reference']['finished_utc']<ms['capture']['started_utc']<ms['capture']['finished_utc']<ms['packets']['started_utc']<ms['packets']['finished_utc']<res['started_utc']<res['finished_utc']
assert driver['elapsed_experiment_seconds']<1800 and driver['waiting_seconds']<7200
assert all(s['status']=='completed' and s['returncode']==0 for s in driver['stages'])
assert ms['reference']['model_forwards']==ms['capture']['model_forwards']==40
assert ms['capture']['layer_call_counts']=={'18':40,'24':40}
assert all(c['reference_byte_equal'] for c in ms['capture']['chunks']) and len(ms['capture']['chunks'])==8
artcounts={};artifactbytes={};qh={};packetdata={}
for stage,m in ms.items():
 assert m['status']=='completed'
 paths=[a['path'] for a in m['artifacts']];assert len(paths)==len(set(paths))
 artcounts[stage]=len(paths);artifactbytes[stage]=0
 for a in m['artifacts']:
  p=Path(a['path']);assert p.stat().st_size==a['bytes'] and sha(p)==a['sha256'];artifactbytes[stage]+=a['bytes']
  if stage=='capture' and ('write' in p.name or 'first_reader' in p.name or 'read' in p.name):
   t=torch.load(p,map_location='cpu',weights_only=True);qh[p.name]={'sha256':tsha(t['q']),'context':t['context']}
  if stage=='packets':
   t=torch.load(p,map_location='cpu',weights_only=True)
   assert t['packed'].dtype==torch.uint8 and t['scale'].dtype==torch.float16
   packetdata[p.name]={'bytes':sum(v.numel()*v.element_size() for v in t.values() if isinstance(v,torch.Tensor)),
      'code_sha256':tsha(t['packed']),'scale_sha256':tsha(t['scale'])}
assert artcounts=={'reference':1,'capture':31,'packets':72}
a=torch.load(P/'reference/latent.pt',map_location='cpu',weights_only=True)['latent'];b=torch.load(P/'capture/latent.pt',map_location='cpu',weights_only=True)['latent']
assert a.shape==b.shape and a.dtype==b.dtype and a.shape[1]==24
latent_shape=list(a.shape);latent_dtype=str(a.dtype)
assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
for c in range(8):assert torch.equal(a[:,3*c:3*c+3].contiguous().view(torch.uint8),b[:,3*c:3*c+3].contiguous().view(torch.uint8))
events=ms['capture']['events'];assert len(events)==28
for row in ms['packets']['rows']:
 l,c=row['layer'],row['source_chunk'];assert row['first_reader_chunk']==c+1 and row['first_reader_call']==0
 for arm,suffix,kind in [('O_own_refresh','write','refresh_write'),('O_first_reader','first_reader','first_reader_available')]:
  q=qh[f'l{l}_{suffix}{c}.pt'];ht=row['H_times'][arm]
  assert q['sha256']==ht['input_q_sha256']
  assert q['context']['call']==(4 if arm=='O_own_refresh' else 0) and q['context']['chunk']==(c if arm=='O_own_refresh' else c+1)
  ev=next(e for e in events if e['kind']==kind and e['layer']==l and e['chunk']==q['context']['chunk'])
  assert ev['q_sha256']==q['sha256']
  assert math.isclose(statistics.median(ht['seconds']),ht['median']) and len(ht['seconds'])==3
 codes=[];scales=[];sizes=[]
 for arm,z in row['arms'].items():
  p=packetdata[f'l{l}_c{c}_{arm}.pt'];assert z['code_sha256']==p['code_sha256'] and z['scale_sha256']==p['scale_sha256'] and z['payload_bytes']==p['bytes']
  assert len(z['seconds'])==3 and math.isclose(z['median'],statistics.median(z['seconds']))
  scales.append(p['scale_sha256']);sizes.append(p['bytes'])
 assert len(set(scales))==len(set(sizes))==1
 assert row['v_payload_bytes']==packetdata[f'l{l}_c{c}_V.pt']['bytes']
for l in (18,24):assert qh[f'l{l}_first_reader6.pt']['sha256']==qh[f'l{l}_read0.pt']['sha256']!=qh[f'l{l}_read3.pt']['sha256']
capacity=read(P.parent/'obq_capacity_codex/RESULT.json')
for o in ms['packets']['offline']:
 old=next(x for x in capacity['rows'] if x['layer']==o['layer'])
 assert o['raw_H_sha256']==old['hessian_raw_sha256']
 assert sha(o['source']['path'])==o['source']['sha256']==old['calibration']['sha256']
 assert o['factor_seconds']<=o['offline_prepare_including_validation_factor_seconds']
comparison=[]
for e in res['endpoints']:
 assert e['native_replay_byte_equal'];aa=e['arms'];l=e['layer'];call=e['call']
 for arm,z in aa.items():
  payload=sum(packetdata[f'l{l}_c{c}_{arm}.pt']['bytes']+packetdata[f'l{l}_c{c}_V.pt']['bytes'] for c in range(1,7))
  assert z['payload_bytes']==payload==54097926 and z['repeated_outputs_byte_equal']
  assert len(z['native_attention_seconds'])==5 and math.isclose(statistics.median(z['native_attention_seconds']),z['native_attention_median'])
 row={'layer':l,'call':call,'sse':{a:z['sse'] for a,z in aa.items()},'gains':{}}
 for arm in ('O_offline','O_own_refresh','O_first_reader'):
  g={a:1-aa[arm]['sse']/aa[a]['sse'] for a in ('R_bf16','R_fp32')};row['gains'][arm]=g
  recorded=next(x for x in res['decisions'][arm]['gains'] if x['layer']==l and x['call']==call)
  for a,v in g.items():assert math.isclose(v,recorded[a],abs_tol=1e-15)
 row['first_reader_reduction_vs_offline']=1-aa['O_first_reader']['sse']/aa['O_offline']['sse'];comparison.append(row)
assert len(comparison)==4
for arm,d in res['decisions'].items():
 vals=[v for e in comparison for v in e['gains'][arm].values()]
 expected='signal' if min(vals)>=.2 else ('stop_arm' if min(vals)<=-.1 else 'undecided')
 assert expected==d['decision']
cost={}
for arm in ('R_bf16','R_fp32','O_offline','O_own_refresh','O_first_reader'):
 vals=[r['arms'][arm]['median'] for r in ms['packets']['rows']]
 allvals=[x for r in ms['packets']['rows'] for x in r['arms'][arm]['seconds']]
 hvals=[r['H_times'][arm]['median'] for r in ms['packets']['rows']] if arm in ('O_own_refresh','O_first_reader') else []
 gpu=[e['arms'][arm]['native_attention_median'] for e in res['endpoints']]
 cost[arm]={'median_of_12_encode_medians':statistics.median(vals),'encode_median_range':[min(vals),max(vals)],'all_36_encode_range':[min(allvals),max(allvals)],
  'H_median_range':[min(hvals),max(hvals)] if hvals else None,'sum_12_encode_medians':sum(vals),'sum_12_H_medians':sum(hvals),
  'gpu_read_median_range': [min(gpu),max(gpu)]}
peaks={s:{'rss':m['max_rss_bytes'],'cuda_allocated':m.get('peak_cuda_bytes',0)} for s,m in ms.items()}
peaks['native']={'rss':res['max_rss_bytes'],'cuda_allocated':res['peak_cuda_bytes']}
for s,p in peaks.items():assert p['rss']<(40 if s in ('reference','capture') else 8)*1024**3 and p['cuda_allocated']<22*1024**3
result={'passed':True,'scope':'current source/review and all artifact hashes; CPU inspection of saved latents, q, packed tensors; metadata/timing/decision arithmetic; no H/OBQ/attention recomputation or GPU rerun',
 'artifact_counts':artcounts,'artifact_bytes':artifactbytes,'locked_source_files':len(lock['sha256']),'locked_review_files':len(lock['review_sha256']),
 'reference_capture_latent_shape':latent_shape,'reference_capture_latent_dtype':latent_dtype,'latent_byte_equal_chunks_independently_rechecked':8,
 'native_identity_checks_from_execution_records':4,'source_tensor_pre_post_active_comparison':'not recorded; not claimed',
 'encoder_input_q_hash_links_checked':24,'packet_payload_and_hashes_independently_checked':72,'comparisons':comparison,
 'decisions':{a:d['decision'] for a,d in res['decisions'].items()},'cost_summary':cost,'peaks':peaks,
 'wait_seconds':driver['waiting_seconds'],'experiment_seconds':driver['elapsed_experiment_seconds'],
 'stage_wall_seconds':{s['name']:s['finished_utc']-s['started_utc'] for s in driver['stages']},
 'folder_bytes_before_audit_output':sum(p.stat().st_size for p in P.rglob('*') if p.is_file()),'native_result_sha256':sha(P/'NATIVE_RESULT.json')}
print(json.dumps(result,ensure_ascii=False,indent=2))
