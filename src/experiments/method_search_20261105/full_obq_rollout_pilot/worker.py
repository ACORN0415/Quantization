"""Separated preparation, integration and generation workers. No implicit launch."""
import argparse,gc,json,math,sys,time,traceback
from pathlib import Path
import pilot_common as p
import torch
from packed_adapter import Quantizer,encode_v,PackedState
sys.path[:0]=[str(p.ROOT/'tools'),str(p.ROOT/'src'),str(p.ROOT/'src/scripts')]

def prepare():
 api=p.load_api(execution=True);manifest=json.loads((p.CAL/'manifest.json').read_text())
 rows=[];factors=[];start=time.monotonic();target=p.HERE/'FACTORS.json'
 assert not target.exists() and not (p.HERE/'FACTORS.pt').exists()
 causal=json.loads((p.HERE.parent/'causal_obq/packets/manifest.json').read_text())
 for layer in range(30):
  p.guard();path=p.CAL/f'l{layer:02d}_c7.pt';t=time.monotonic();data=p.verified(path,manifest);io=time.monotonic()-t
  t=time.monotonic();h,raw=api.c.hessian(data['q'],return_raw=True);u=api.c.factorize(h);compute=time.monotonic()-t
  assert u.shape==(12,128,128) and u.dtype==torch.float32 and torch.isfinite(u).all()
  assert (u.diagonal(dim1=1,dim2=2)>0).all() and torch.count_nonzero(torch.tril(u,-1))==0
  raw_hash=p.tsha(raw)
  if layer in (18,24):assert raw_hash==next(x for x in causal['offline'] if x['layer']==layer)['raw_H_sha256']
  rows.append({'layer':layer,'source':p.record(path),'IO_and_hash_seconds':io,'H_factor_seconds':compute,
   'raw_H_sha256':raw_hash,'H_sha256':p.tsha(h),'U_sha256':p.tsha(u)})
  factors.append(u);del data,h,raw,u;gc.collect();p.guard()
 t=time.monotonic();path=p.HERE/'FACTORS.pt';torch.save({'U':torch.stack(factors),'layers':torch.arange(30)},path);write=time.monotonic()-t
 p.save(target,{'status':'completed','rows':rows,'artifact':p.record(path),'write_seconds':write,
  'total_seconds':time.monotonic()-start,'U_tensor_bytes':23592960,'peak':p.peak()})

def load_factors():
 meta=json.loads((p.HERE/'FACTORS.json').read_text());assert meta['status']=='completed'
 assert p.sha(meta['artifact']['path'])==meta['artifact']['sha256']
 data=torch.load(meta['artifact']['path'],map_location='cpu',weights_only=True)
 assert torch.equal(data['layers'],torch.arange(30)) and data['U'].shape==(30,12,128,128)
 for row in meta['rows']:assert p.tsha(data['U'][row['layer']])==row['U_sha256']
 return data['U'],meta

def integration():
 p.cuda_settings();torch.cuda.init();torch.cuda.reset_peak_memory_stats()
 api=p.load_api(execution=True,kernel=True);u,meta=load_factors();hm=api.c.hadamard().cuda()
 cap=p.HERE.parent/'causal_obq/capture';manifest=json.loads((cap/'manifest.json').read_text());rows=[]
 for layer,chunk in ((18,1),(24,6)):
  p.guard();data=p.verified(cap/f'l{layer}_write{chunk}.pt',manifest)
  k=data['k'].cuda();v=data['v'].cuda();factor=u[layer].cuda();z,sc=api.gpu_frames(k,hm)
  # The stage2 arithmetic/decoder is reused; check the pilot adapter on these fixed cases.
  vp=encode_v(v,api);vc=api.c.encode_v(data['v'])
  assert all(p.same(vp[n].cpu(),vc[n]) for n in ('packed','scale'))
  assert p.same(api.restore_gpu(vp,hm).cpu(),api.c.restore(vc))
  for arm in p.ARMS[1:]:
   context={'chunk':1};events=[];q=Quantizer(arm,layer,factor if arm=='O_full_gpu' else None,hm,api,context,events.append)
   one=q.encode_pair(k,v);before=[{n:t.cpu().clone() for n,t in s.packet.items() if isinstance(t,torch.Tensor)} for s in one]
   alg='O_offline' if arm=='O_full_gpu' else arm
   ref=api.v1.code_packet(z.cpu(),sc.cpu(),u[layer],alg)
   assert all(p.same(one[0].packet[n].cpu(),ref[j]) for j,n in enumerate(('packed','scale','sentinel')))
   context['chunk']=2;two=q.encode_pair((-k*.75).bfloat16(),(-v*.5).bfloat16())
   assert not p.same(one[0].packet['packed'],two[0].packet['packed'])
   for i,state in enumerate(one):
    assert all(p.same(state.packet[n].cpu(),value) for n,value in before[i].items())
    assert all(state.packet[n].data_ptr()!=two[i].packet[n].data_ptr() for n in before[i])
   assert events[0]['layer']==layer and events[0]['source_chunk']==0 and events[1]['source_chunk']==1
   rows.append({'layer':layer,'source_chunk':chunk,'arm':arm,'native_GPU_preprocessed_CPU_reference_exact':True,
    'V_CPU_packet_exact':True,'V_reader_exact':True,'owned_packet_lifetime':True})
   del q,one,two,before,ref
  del data,k,v,factor,z,sc,vp,vc;gc.collect();torch.cuda.empty_cache();p.guard()
 p.save(p.HERE/'INTEGRATION.json',{'status':'completed','rows':rows,'peak':p.peak(),'generations':0,'VAE_decodes':0})

def encoders(arm,factors,hm,api,context,sink):
 return [Quantizer(arm,l,factors[l] if arm=='O_full_gpu' else None,hm,api,context,sink) for l in range(30)]

def generate(index):
 import p3_tf_fr as pilot
 item=p.prompts()[index];assert json.loads((p.HERE/'INTEGRATION.json').read_text())['status']=='completed'
 p.cuda_settings();torch.cuda.init();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 api=p.load_api(execution=True,kernel=True);u_cpu,fmeta=load_factors();hm=api.c.hadamard().cuda()
 source=pilot.source_hash();pipe=pilot.build(item['seed']);p.guard();phase={'model_build':p.peak()}
 assert pipe.num_transformer_blocks==30 and pipe.num_frame_per_block==3 and pipe.frame_seq_length==1560 and pipe.local_attn_size==21
 cond=pilot.encode(pipe,[item['text']])[0];p.guard();phase['conditioning']=p.peak();reference=None;noise_reference=None
 model_prepare_seconds=time.monotonic()-start
 for arm in p.ORDERS[index]:
  p.guard();p.count('generations');folder=p.HERE/item['id']/arm;folder.mkdir(parents=True,exist_ok=False)
  r={'status':'running','input_id':item['id'],'seed':item['seed'],'text_sha256':item['text_sha256'],
   'arm':arm,'model_source_sha256':source,'model_forwards':0,'chunks':[],'phases_before_sampling':phase,
   'shared_model_and_conditioning_seconds':model_prepare_seconds,'context_window':21,'started_utc':time.time()}
  events=(folder/'encode_events.jsonl').open('x');handle=None;context={'chunk':0}
  def sink(row):events.write(json.dumps(row)+'\n')
  try:
   t=time.monotonic();factors=u_cpu.cuda() if arm=='O_full_gpu' else None;torch.cuda.synchronize()
   r['factor_H2D_seconds']=time.monotonic()-t;r['factor_GPU_bytes']=factors.numel()*4 if factors is not None else 0
   noise=pilot.chunk_noise(pipe,126,item['seed'],pilot.gpu)
   t=time.monotonic();noise_cpu=noise.cpu();r['noise_sha256']=p.tsha(noise_cpu)
   r['initial_noise_chunk_sha256']=[p.tsha(noise_cpu[:,3*c:3*(c+1)]) for c in range(42)]
   if arm=='BF16':noise_reference=r['initial_noise_chunk_sha256']
   else:assert r['initial_noise_chunk_sha256']==noise_reference
   r['noise_verification_seconds']=time.monotonic()-t;del noise_cpu
   if arm=='BF16':cache=pilot.bf16_cache(pipe,pilot.gpu);qs=[]
   else:
    cache=pilot.compressed_cache(pipe,None,pilot.gpu);qs=encoders(arm,factors,hm,api,context,sink)
    for layer,cc in enumerate(cache):
     assert cc['layer']==layer and cc['store'].residual_tokens==0 and cc['store'].sink_tokens==0
     cc['store'].quantizer=qs[layer]
   xc=pilot.cross_cache(pipe,pilot.gpu);values=[]
   def forward_count(module,args):
    p.guard();r['model_forwards']+=1
    if r['model_forwards']>210:raise RuntimeError('210 model forwards exceeded')
   handle=pipe.generator.register_forward_pre_hook(forward_count)
   torch.cuda.synchronize();tick=time.perf_counter()
   for chunk in range(42):
    context['chunk']=chunk;y=pilot.run_chunk(pipe,cache,xc,cond,noise,chunk,item['seed']).cpu()
    assert torch.isfinite(y).all();values.append(y)
    row={'chunk':chunk}
    if arm!='BF16':
     target=reference[:,chunk*3:(chunk+1)*3]
     if chunk==0:assert p.same(y,target),'initial no-history chunk differs from BF16'
     row['latent_metric']=p.latent_metric(y,target,start=0)
     expected=list(range(max(0,chunk-6),chunk))
     for layer,cc in enumerate(cache):
      store=cc['store'];assert [s['start']//4680 for s in store.segments]==expected
      assert store.pending_start==chunk*4680 and all(s['raw'] is None and s['state'] is not None for s in store.segments)
      assert qs[layer].pairs==chunk
     row['past_KV_payload_bytes']=sum(p.nbytes(s.packet) for cc in cache for seg in cc['store'].segments for s in seg['state'])
     assert row['past_KV_payload_bytes']==len(expected)*30*(4515841+4500480)
    r['chunks'].append(row)
    if chunk%7==6 or chunk==41:p.save(folder/'sampling.json',r);print(item['id'],arm,chunk+1,'/42',flush=True)
   torch.cuda.synchronize();r['sampling_seconds']=time.perf_counter()-tick
   assert r['model_forwards']==210 and (arm=='BF16' or all(q.pairs==41 for q in qs))
   latent=torch.cat(values,1);assert latent.shape==(1,126,16,60,104)
   if arm=='BF16':reference=latent.clone()
   r['late_latent']=p.latent_metric(latent,reference);r['whole_latent']=p.latent_metric(latent,reference,start=0)
   r['descriptive_latent_windows']={name:p.latent_metric(latent,reference,start=a,end=b) for name,a,b in
    (('chunks_1_to_7',3,24),('chunks_8_to_20',24,63),('chunks_21_to_41',63,126))}
   path=folder/'latent.pt';torch.save({'latent':latent},path);r['latent']=p.record(path)
   r['layer_encode_pair_calls']=[q.pairs for q in qs] if qs else [0]*30
   r['layer_encode_timing']=[q.timing_after_arm_sync() for q in qs]
   assert pilot.source_hash()==source;p.guard();r['status']='completed'
  except BaseException as exc:
   r.update(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc());raise
  finally:
   if handle:handle.remove()
   events.close();r.update(finished_utc=time.time(),arm_total_seconds=time.time()-r['started_utc'],
    cumulative_process_peak=p.peak(),peak_scope='Cumulative from before model build across all arms in this input worker; not independent per-arm peaks',
    events=p.record(folder/'encode_events.jsonl'))
   p.save(folder/'sampling.json',r)
  del cache,xc,noise,values,latent,y,factors,qs;gc.collect();torch.cuda.empty_cache();p.guard()
 assert len({json.loads((p.HERE/item['id']/a/'sampling.json').read_text())['noise_sha256'] for a in p.ARMS})==1

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','integration','generate'));ap.add_argument('--input',type=int,choices=(0,1));args=ap.parse_args()
 p.verify_lock()
 if args.action=='generate':assert args.input is not None;generate(args.input)
 else:{'prepare':prepare,'integration':integration}[args.action]()
 p.verify_lock();p.guard()
if __name__=='__main__':main()
