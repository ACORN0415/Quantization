"""CPU-only adapter contracts, state transitions and fixed decision tests. No capture/model load."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,copy,json,sys,time
from pathlib import Path
from unittest.mock import patch
import pilot_common as p
import torch
from packed_adapter import encode_v,PackedState
from worker import encoders
sys.path.insert(0,str(p.ROOT/'src'))
from kvq.store import ChunkStore

def main():
 assert not torch.cuda.is_initialized();target=p.HERE/'SELFTEST_V3.json';assert not target.exists()
 for path in p.HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
 api=p.load_api();hm=api.c.hadamard();g=torch.Generator().manual_seed(20261106)
 k=torch.randn(1,4680,12,128,generator=g).bfloat16();v=torch.randn(k.shape,generator=g).bfloat16()
 original=[p.tsha(k),p.tsha(v)];packet=encode_v(v,api);ref=api.c.encode_v(v)
 assert all(p.same(packet[n],ref[n]) for n in ('packed','scale'))
 assert p.same(api.restore_gpu(packet,hm),api.c.restore(ref))
 # CPU stand-in tests the adapter plumbing only; it is not a CUDA kernel test.
 calls=[]
 class CPUStandIn:
  def obq_pack(self,z,s,u):
   calls.append(float(u[0,0,1]));return api.c.pack(api.c.codes_obq(z,s,None,u))
 api.EXT=CPUStandIn();factors=torch.eye(128).repeat(30,12,1,1)
 factors[:,:,0,1]=torch.arange(30)[:,None]*.001
 context={'chunk':1};events=[];checks=[]
 with patch.object(p,'guard',lambda:None):
  for arm in p.ARMS[1:]:
   qs=encoders(arm,factors if arm=='O_full_gpu' else None,hm,api,context,events.append)
   assert all(q.layer==l and (q.factor is None if arm!='O_full_gpu' else q.factor.data_ptr()==factors[l].data_ptr()) for l,q in enumerate(qs))
   for layer in (0,29):
    context['chunk']=1;one=qs[layer].encode_pair(k,v)
    before=[{n:t.clone() for n,t in s.packet.items() if isinstance(t,torch.Tensor)} for s in one]
    z,s=api.gpu_frames(k,hm);reference=api.v1.code_packet(z,s,factors[layer],'O_offline' if arm=='O_full_gpu' else arm)
    assert all(p.same(one[0].packet[n],reference[i]) for i,n in enumerate(('packed','scale','sentinel')))
    context['chunk']=2;two=qs[layer].encode_pair((-k*.75).bfloat16(),(-v*.5).bfloat16())
    assert not p.same(one[0].packet['packed'],two[0].packet['packed'])
    for index,old in enumerate(one):
     assert all(p.same(old.packet[n],t) for n,t in before[index].items())
     assert all(old.packet[n].data_ptr()!=two[index].packet[n].data_ptr() for n in before[index])
    checks.append({'arm':arm,'layer':layer,'CPU_packet_exact':True,'owned_output_preserved':True})
 assert calls==[float(factors[0,0,0,1])]*2+[float(factors[29,0,0,1])]*2
 calls_before_invalid=list(calls);invalid_checks=0
 with patch.object(p,'guard',lambda:None):
  for arm in p.ARMS[1:]:
   for fill in (0.,1e-10):
    context['chunk']=1;q=encoders(arm,factors if arm=='O_full_gpu' else None,hm,api,context,None)[0]
    try:q.encode_pair(torch.full_like(k,fill),v)
    except AssertionError as exc:assert 'stored FP16 scale' in str(exc);invalid_checks+=1
    else:raise AssertionError('Invalid scale reached encoder')
 assert invalid_checks==6 and calls==calls_before_invalid
 previous=os.getcwd()
 try:
  os.chdir(p.ROOT/'src')
  for path in ('wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth',
               'wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth','checkpoints/self_forcing_dmd.pt'):
   assert Path(path).is_file(),path
 finally:os.chdir(previous)
 assert [p.tsha(k),p.tsha(v)]==original
 # Independent expected logical cache: seven chunk slots, current uncompressed.
 class State:
  def __init__(self,t):self.t=t.clone()
  def restore(self):return self.t
 class Identity:
  def __init__(self):self.calls=0
  def encode_pair(self,k,v):self.calls+=1;return State(k),State(v)
 identity=Identity();store=ChunkStore(identity,capacity=7*4)
 for chunk in range(10):
  for call in range(5):
   x=torch.full((1,4,1,1),float(10*chunk+call));past=[torch.full_like(x,float(10*i+4)) for i in range(max(0,chunk-6),chunk)]
   expected=torch.cat(past+[x],1)
   def attention(q,k,v):assert torch.equal(k,expected) and torch.equal(v,expected);return k.clone()
   store.attend(x,x,x,chunk*4,attention)
   assert identity.calls==chunk and store.pending_start==chunk*4
   assert [s['start']//4 for s in store.segments]==list(range(max(0,chunk-6),chunk))
 # Fidelity uses accumulated SSE/energy, not a mean of chunk-relative errors.
 x=torch.zeros(1,126,1);reference=torch.arange(1,127).reshape(1,126,1).float()
 assert p.latent_metric(x,reference)['nse']==1 and p.latent_metric(reference,reference)['nse']==0
 def rows(full_nse=.8,dm=0.,dd=.01):
  return [{'input_id':item['id'],'arm':arm,'late_latent':{'nse':0. if arm=='BF16' else full_nse if arm=='O_full_gpu' else 1.},
   'windows':{'late':{'musiq':70.+(dm if arm=='O_full_gpu' else 0.),'dino_reference':.9+(dd if arm=='O_full_gpu' else 0.)}}}
   for item in p.prompts() for arm in p.ARMS]
 assert p.decide(rows())['decision']=='signal'
 assert p.decide(rows(full_nse=.95))['decision']=='undecided'
 assert p.decide(rows(full_nse=1.2))['decision']=='stop_quality'
 assert p.decide(rows(dm=-2.1))['decision']=='stop_quality'
 assert p.decide(rows(dd=-.021))['decision']=='stop_quality'
 assert p.decide(rows(dm=-.75))['decision']=='undecided'
 review={'root_execution_authorized':True,'root_code_review_accepted':True,'independent_codex_code_review_accepted':True,
  'root_stage2_accepted':True,'cost_review_accepted':True,'historical_fable_design_reviewed':True,
  'review_route':'root_accepted_deferred_review','review_status':'deferred_not_completed',
  'claude_final_source_reviewed':False,'claude_final_result_reviewed':False,'final_reviewer_model':'Opus 5.5',
  'claude_not_before':'2026-11-05T21:50:00+09:00','administrative_amendment_adopted':True}
 p.validate_review_status(review)
 for key,value in (('root_execution_authorized',False),('independent_codex_code_review_accepted',False),
                   ('claude_final_source_reviewed',True),('review_status','completed')):
  invalid=dict(review);invalid[key]=value
  try:p.validate_review_status(invalid)
  except AssertionError:pass
  else:raise AssertionError('Incorrect review status accepted: '+key)
 assert json.loads((p.HERE/'STAGE2_BINDING.json').read_text())['status']=='pending'
 assert not torch.cuda.is_initialized()
 result={'passed':True,'checks':checks,'CPU_V_packet_exact':True,'layer_factor_mapping_30':True,
  'logical_cache_10_chunks_5_calls':True,'decision_cases':6,'administrative_review_gate_cases':5,
  'invalid_zero_or_tiny_scale_rejected_before_encoder':invalid_checks,'src_cwd_relative_model_paths_present':True,
  'real_capture_reads':0,'model_loads':0,'GPU_calls':0,
  'OBQ_backend':'CPU reference stand-in only; actual fused kernel/native GPU integration remains pending',
  'source_sha256':{str(f):p.sha(f) for f in p.HERE.glob('*.py')},'preview_api_source':p.record(p.PREVIEW_API),
  'torch_version':torch.__version__,'time_utc':time.time()}
 p.save(target,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
