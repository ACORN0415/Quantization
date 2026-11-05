"""Generate procedural synthetic tensors; never load a source capture tensor."""
import hashlib,importlib.util,json,math,os,resource,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('synthetic_frozen_driver',HERE/'diagnose.py');d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d);d.setup_cpu();m=d.modules();c=m['common']
REG=json.loads((HERE/'FIXTURE_REGISTRATION.json').read_text());START=REG['started_utc']
def guard():
 if time.time()-START>1200:raise TimeoutError('20-minute fixture including preparation')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:raise MemoryError('fixture RSS')
 if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())>2*1024**3:raise OSError('fixture disk')
 if torch.cuda.is_initialized():raise RuntimeError('GPU forbidden')
def save(name,value):
 guard();p=HERE/'inputs'/name;p.parent.mkdir(exist_ok=True);torch.save(value,p);return {'path':str(p),'sha256':d.sha(p),'bytes':p.stat().st_size,'provenance':'procedural synthetic tensor, not research capture'}
def sample(seed,shape,scale=.2,mean=0.):
 g=torch.Generator().manual_seed(seed);return (torch.randn(shape,generator=g)*scale+mean).bfloat16()
@torch.no_grad()
def direct_attention(q,k,v):
 # Independent explicit FP32 visible-window expression matching registered arithmetic.
 kt=k.permute(1,2,0).contiguous();vt=v.permute(1,0,2).contiguous();out=torch.empty_like(q)
 for i in range(0,len(q),32):
  score=torch.bmm(q[i:i+32].permute(1,0,2),kt)/math.sqrt(128)
  out[i:i+32]=torch.bmm(torch.softmax(score,dim=-1),vt).permute(1,0,2)
 return out
layers=[]
for layer in (18,24):
 guard();calq=sample(1000+layer,(1,4680,12,128),.1)
 # Both layers use real production dimensions, independent procedural seeds.
 rawkeys=[];rawvalues=[];writes=[];packets=[];decoded={a:[] for a in ('R_bf16','R_fp32','O_offline')};v4=[]
 hd,hr=c.hessian(calq,return_raw=True);factor=c.factorize(hd);qq=calq[0].permute(1,0,2).double();rawC=qq.transpose(1,2).bmm(qq)/4680
 # A valid supplied fixed R=H128 is sufficient for a runtime fixture; no method selection claim.
 rotation=c.hadamard().repeat(12,1,1);choice=torch.full((12,),16,dtype=torch.uint8)
 fixed=save(f'l{layer}_fixed.pt',{'R':rotation,'choice':choice,'U':factor})
 weight=torch.eye(1536,dtype=torch.bfloat16);bias=torch.linspace(-.02,.02,1536).bfloat16();wo=save(f'l{layer}_wo.pt',{'wo':weight,'bias':bias})
 for chunk in range(1,7):
  guard();key=sample(10000+layer*100+chunk,(1,4680,12,128),.2,.04);value=sample(20000+layer*100+chunk,(1,4680,12,128),.2)
  rawkeys.append(key);rawvalues.append(value)
  writes.append(save(f'l{layer}_write{chunk}.pt',{'q':calq,'k':key,'v':value,'context':{'layer':layer,'chunk':chunk,'call':4,'timestep':0.}}))
  vp=c.encode_v(value);vv=c.restore(vp);v4.append(vv);row={'chunk':chunk,'V':save(f'l{layer}_c{chunk}_V.pt',vp),'V_tensor_payload_bytes':c.nbytes(vp),'K':{}}
  for arm in decoded:
   kp=c.encode_k(key,arm,hd,factor);record=save(f'l{layer}_c{chunk}_{arm}.pt',kp);decoded[arm].append(c.restore(kp))
   row['K'][arm]={'artifact':record,'payload_bytes':c.nbytes(kp),'code_sha256':d.tsha(c.unpack(kp['packed'],kp['shape'])),'scale_sha256':d.tsha(kp['scale'])}
  packets.append(row);print('fixture input',layer,chunk,flush=True)
 current=sample(30000+layer,(1,4680,12,128),.2,.04);currentv=sample(40000+layer,(1,4680,12,128),.2)
 pastk=torch.cat(rawkeys,1);pastv=torch.cat(rawvalues,1);pastv4=torch.cat(v4,1);p0k=torch.cat((pastk,current),1)
 p0=save(f'l{layer}_p0.pt',{'q':calq,'k':p0k,'v':torch.cat((pastv,currentv),1),'wo':weight,'bias':bias})
 reads=[];anchors={};fullk=torch.cat((pastk,current),1)[0].float();fullv=torch.cat((pastv,currentv),1)[0].float();vwindow=torch.cat((pastv4,currentv),1)[0].float()
 # Independently reconstruct fixed rotation from its stated arithmetic, not driver helper.
 z=torch.einsum('...hd,hde->...he',pastk.float(),rotation).bfloat16()[0].reshape(18,1560,12,128)
 z=torch.cat((z,z.new_zeros(18,8,12,128)),1).reshape(18,98,16,12,128);ss=(z.abs().amax(2,keepdim=True)/7).clamp_min(1e-8).half();code=(z.float()/ss.float()).round().clamp(-8,7).to(torch.int8)
 fixedK=(code.bfloat16()*ss.bfloat16()).reshape(18,1568,12,128)[:,:1560].reshape_as(pastk)
 fixedK=torch.einsum('...hd,hde->...he',fixedK.float(),rotation.transpose(-1,-2)).bfloat16()
 for call in (0,3):
  guard();qfull=sample(50000+layer*10+call,(1,4680,12,128),.1);q=qfull[0,d.QIDS].float();native=direct_attention(q,fullk,fullv).bfloat16()[None]
  reads.append(save(f'l{layer}_read{call}.pt',{'q':qfull,'k_current':current,'v_current':currentv,'native_av':native,'context':{'layer':layer,'chunk':7,'call':call},'past_chunk_ids':[1,2,3,4,5,6]}))
  reference=direct_attention(q,fullk,vwindow);anchors[str(call)]={}
  for arm,keys in [('R_bf16',torch.cat(decoded['R_bf16'],1)),('R_fp32',torch.cat(decoded['R_fp32'],1)),('FULL',torch.cat(decoded['O_offline'],1)),('FIXED17',fixedK)]:
   out=direct_attention(q,torch.cat((keys,current),1)[0].float(),vwindow);delta=(out-reference).reshape(128,-1)@weight.float().t();anchors[str(call)][arm]=float(delta.double().square().sum())
 binding={'layer':layer,'p0_calibration':p0,'Wo':wo,'writes':writes,'reads':reads,'packets':packets,'offline_rotated_raw_H_sha256':d.tsha(hr),'offline_rotated_damped_H_sha256':d.tsha(hd),'fixed_state_artifact':fixed,'fixed_matrix_sha256':d.tsha(rotation),'fixed_U_sha256':d.tsha(factor),'fixed_raw_C_sha256':d.tsha(rawC),'anchors':anchors}
 layers.append(binding);print('fixture layer bindings ready',layer,flush=True)
 del decoded,rawkeys,rawvalues,pastk,pastv,pastv4,p0k,fullk,fullv,vwindow,z,ss,code,fixedK
 import gc;gc.collect()
inputs={'status':'synthetic_only','Q128_indices':d.QIDS.tolist(),'layers':layers}
def verify_paths(x):
 if isinstance(x,dict):
  if 'path' in x:
   p=Path(x['path']).resolve();assert p.is_relative_to(HERE/'inputs') and p.suffix=='.pt'
  for v in x.values():verify_paths(v)
 elif isinstance(x,list):
  for v in x:verify_paths(v)
verify_paths(inputs);d.save_json(HERE/'INPUT_BINDINGS_DRAFT.json',inputs)
evidence={'synthetic_only':True,'test_authority':'root authorized isolated runtime fixture; no real-data execution authority','actual_Opus_final_source_review_complete':False,'purpose':'unchanged driver guard exercise; true guard booleans in fixture LOCK are test fixtures only','original_stage0_LOCK_RUN_touched':False}
d.save_json(HERE/'SYNTHETIC_EVIDENCE.json',evidence)
sources={str(p):d.sha(p) for p in [HERE/'diagnose.py',HERE/'bit_baseline.py',HERE/'mean_baseline.py',HERE/'OMEGA_BINDINGS_V2.json',HERE/'build_inputs.py',HERE/'sitecustomize.py',HERE/'FIXTURE_REGISTRATION.json',*d.MODULE_PATHS.values()]}
d.save_json(HERE/'SOURCES_V3.json',{'synthetic_only':True,'sources':sources})
lock={'synthetic_only':True,'test_authority':True,'NOT_REAL_REVIEW_OR_REAL_EXPERIMENT_AUTHORIZATION':True,'root_execution_authorized':True,'actual_opus_review_complete':True,'validity_policy_accepted':True,'review_evidence_sha256':{str(HERE/'SYNTHETIC_EVIDENCE.json'):d.sha(HERE/'SYNTHETIC_EVIDENCE.json')},'dense_extra_bytes':d.DENSE_EXTRA,'primary_baselines':list(d.PRIMARY_BASELINES),'pending':[],'cap_extra_bytes':d.CAP_EXTRA,'arms':list(d.ARMS),'total_seconds':d.LIMIT_SECONDS,'rss_bytes':d.LIMIT_RSS,'disk_bytes':d.LIMIT_DISK,'sources':sources,'input_bindings_sha256':d.sha(HERE/'INPUT_BINDINGS_DRAFT.json'),'outer_stricter_fixture_limit_seconds':1200}
d.save_json(HERE/'LOCK.json',lock);guard();d.save_json(HERE/'BUILD_RESULT.json',{'status':'completed','scope':'procedural synthetic inputs only','elapsed_since_fixture_start':time.time()-START,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'generated_disk_bytes':sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),'real_capture_reads':0});print('synthetic input build completed',flush=True)
