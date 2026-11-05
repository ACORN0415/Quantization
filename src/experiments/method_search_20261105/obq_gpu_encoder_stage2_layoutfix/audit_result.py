"""Independent read-only scalar/hash/CPU-output audit; never launches CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
from pathlib import Path
import hashlib,json,math,re,time
import torch
torch.set_num_threads(4);torch.set_grad_enabled(False)
P=Path(__file__).resolve().parent;C=P.parent/'causal_obq'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
checks=[]
def ck(name,ok):assert ok,name;checks.append(name)
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def close(a,b):return math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-12)
def qstats(x):
 x=sorted(x)
 def q(p):
  pos=(len(x)-1)*p;lo=int(pos);f=pos-lo
  return x[lo]*(1-f)+x[min(lo+1,len(x)-1)]*f
 return {'median':q(.5),'q1':q(.25),'q3':q(.75),'iqr':q(.75)-q(.25)}
t0=time.monotonic();r=read(P/'RESULT.json');l=read(P/'LOCK.json');s=read(P/'SUPERVISOR.json');b=read(P/'BUILD.json');quality=read(P/'QUALITY.json')
ck('completed worker/supervisor',r['status']=='completed' and s['status']=='finished' and s['return_code']==0)
for path,h in l['sha256'].items():ck('source '+path,sha(path)==h)
ck('lock hash',r['lock_sha256']==sha(P/'LOCK.json'))
ck('shared deadline',not s['deadline_exceeded'] and s['absolute_deadline_utc']==1793875658.1258163 and s['total_since_original_seconds']<=2700)
ck('reused identical binary',b['compile_calls']==0 and b['status']=='reused_existing_binary' and sha(b['module'])==b['binary_sha256']=='ae6e1a50caeff3b13327ab15b27a3e20695530938d4068ab5e578291697eff75')
ck('no models/generation',r['models_loaded']==r['generations']==0)
ck('resource caps',r['max_RSS_bytes']<8*1024**3 and r['peak_cuda_allocated_bytes']<4*1024**3 and r['disk_bytes']<2*1024**3)
ck('quality hash',r['quality']['sha256']==sha(P/'QUALITY.json'))
ck('four lifetime validations',[(x['layer'],x['chunk'],x['stage']) for x in r['code_lifetime_validation']]==[(18,1,'T_code'),(24,6,'T_code'),(18,1,'T_encoder'),(24,6,'T_encoder')])
for row in r['code_lifetime_validation']:
 ck('all lifetime flags '+str((row['layer'],row['stage'])),len(row['lifetime'])==4 and all(all(x.values()) for x in row['lifetime'].values()))
 ck('owned copy '+str((row['layer'],row['stage'])),row['setup']['OBQ_graph']['owned_output_copy_bytes']==4515841 and row['setup']['OBQ_graph']['factor_copied_each_call'])
ck('quality completed',quality['status']=='completed' and quality['quality_gate'])
ck('four native endpoints',[(x['layer'],x['call']) for x in quality['endpoints']]==[(18,0),(18,3),(24,0),(24,3)])
scales={};bytes_by_file=[]
for item in quality['packets']:
 ck('packet file '+item['path'],sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes'])
 d=torch.load(item['path'],map_location='cpu',weights_only=True)
 n=sum(v.numel()*v.element_size() for v in d.values() if isinstance(v,torch.Tensor));ck('packet bytes '+item['path'],n==4515841)
 key=re.match(r'quality_l(\d+)_c(\d+)_(.*)\.pt',Path(item['path']).name).groups();group=key[:2]
 if group in scales:ck('same scale across arms '+item['path'],bits(d['scale'],scales[group]))
 else:scales[group]=d['scale'].clone()
 ck('V exact record '+item['path'],item['V_GPU_restore_exact']);bytes_by_file.append(n)
ck('all 36 packets/12 groups',len(bytes_by_file)==36 and len(scales)==12)
endpoint_rows=[]
for row in quality['endpoints']:
 outitem=row['output_artifact'];ck('output file',sha(outitem['path'])==outitem['sha256'])
 out=torch.load(outitem['path'],map_location='cpu',weights_only=True);wo=torch.load(C/f"capture/l{row['layer']}_wo.pt",map_location='cpu',weights_only=True)
 original=torch.load(C/f"capture/l{row['layer']}_read{row['call']}.pt",map_location='cpu',weights_only=True)
 ck('native BF16 replay',bits(out['reference'],original['native_av'][0]) and row['native_replay_byte_equal'])
 y=out['reference'].float().reshape(128,-1)@wo['wo'].float().t()+wo['bias'].float();energy=float(y.double().square().sum());ss={}
 for a in ('R_bf16','R_fp32','O_gpu'):
  delta=(out[a].float()-out['reference'].float()).reshape(128,-1)@wo['wo'].float().t();ss[a]=float(delta.double().square().sum())
  ck('recomputed metric '+str((row['layer'],row['call'],a)),close(ss[a],row['arms'][a]['sse']) and close(ss[a]/energy,row['arms'][a]['nse']) and close(energy,row['arms'][a]['reference_energy']))
 gains={a:1-ss['O_gpu']/ss[a] for a in ('R_bf16','R_fp32')}
 ck('recomputed quality gate '+str((row['layer'],row['call'])),all(close(gains[a],row['gains'][a]) and gains[a]>=.2 for a in gains))
 endpoint_rows.append({'layer':row['layer'],'call':row['call'],'sse':ss,'reference_energy':energy,'gains':gains})
summary=[];pairs_count=0
for row in r['timings']:
 ck('input preserved',row['input_preserved'] and row['GPU_preprocessed_CPU_code_reference_gate'])
 ck('three registered pairs',[x['pair'] for x in row['pairs']]==[['RTN_eager','OBQ_graph'],['RTN_graph','OBQ_graph'],['OBQ_fused','OBQ_graph']])
 for pair in row['pairs']:
  a,bname=pair['pair'];samples=pair['samples'];ck('20 rounds',len(samples)==20)
  for n,x in enumerate(samples):ck('round order',x['round']==n and x['order']==([a,bname] if n%2==0 else [bname,a]))
  for kind in ('host_seconds','gpu_seconds'):
   for alg in (a,bname):
    calc=qstats([x['times'][alg][kind] for x in samples]);ck('time summary',all(close(calc[k],pair['summary'][alg][kind][k]) for k in calc))
   calc=qstats([x['times'][bname][kind]-x['times'][a][kind] for x in samples]);ck('paired summary',all(close(calc[k],pair['paired_second_minus_first'][kind][k]) for k in calc))
  pairs_count+=20
 rtne=row['pairs'][0]['summary']['RTN_eager']['host_seconds']['median'];rtng=row['pairs'][1]['summary']['RTN_graph']['host_seconds']['median'];p=row['pairs'][2]
 fused=p['summary']['OBQ_fused']['host_seconds']['median'];graph=p['summary']['OBQ_graph']['host_seconds']['median']
 summary.append({'layer':row['layer'],'chunk':row['chunk'],'stage':row['stage'],'RTN_eager_ms':rtne*1000,'RTN_graph_ms':rtng*1000,'OBQ_graph_ms':graph*1000,'OBQ_fused_ms':fused*1000,
   'Graph_over_fused_ratio':graph/fused,'fused_increment_over_fastest_RTN_ms':(fused-min(rtne,rtng))*1000,'conditional30_increment_over_BF16_chunk_percent':30*(fused-min(rtne,rtng))/row['BF16_reference_chunk_seconds']*100,
   'CPU_original_rotated_unequal_elements':row['CPU_original_rotated_unequal_elements'],'CPU_original_scale_unequal_elements':row['CPU_original_scale_unequal_elements']})
ck('four timed cases',len(r['timings'])==4 and pairs_count==240)
ck('no audit CUDA',not torch.cuda.is_initialized())
result={'passed':True,'checks_passed':len(checks),'locked_source_files':len(l['sha256']),'elapsed_seconds':time.monotonic()-t0,'GPU_reruns':0,'paired_rounds':pairs_count,'algorithm_timing_samples':pairs_count*2,
 'result_sha256':sha(P/'RESULT.json'),'quality_sha256':sha(P/'QUALITY.json'),'endpoints':endpoint_rows,'timing_summary':summary,
 'limitations':['GPU exact/lifetime assertions verified from records without GPU reexecution.','Native post-W_O SSE/NSE independently recalculated from saved CPU outputs.','30-layer estimate is not full-model generation timing.','Compiler binary reused; original compiler child RSS was not measured.']}
with (P/'AUDIT_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
