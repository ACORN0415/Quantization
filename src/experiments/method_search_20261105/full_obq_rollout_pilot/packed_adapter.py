"""Owned packet adapter; all K code and readers are imported from stage2."""
import time
import torch
import pilot_common as p

def encode_v(value,api):
 assert value.shape==(1,4680,12,128) and value.dtype==torch.bfloat16
 z=torch.cat((value,value.new_zeros(1,8,12,128)),1).reshape(1,293,16,12,128)
 scale=(z.abs().amax(2,keepdim=True)/7).clamp_min(1e-8)
 code=(z/scale).round().clamp(-8,7).to(torch.int8)
 packet={'kind':'V','shape':tuple(code.shape),'packed':api.c.pack(code),'scale':scale.half()}
 assert p.nbytes(packet)==4500480
 return packet

class PackedState:
 def __init__(self,packet,hm,api):self.packet,self.hm,self.api=packet,hm,api
 def restore(self):return self.api.restore_gpu(self.packet,self.hm)
 def bytes(self):
  n=p.nbytes(self.packet)
  return {'resident':n,'packed_estimate':n,'actual_packed':n}

class Quantizer:
 def __init__(self,arm,layer,factor,hm,api,context,event_sink=None):
  assert arm in p.ARMS[1:] and 0<=layer<30
  if arm=='O_full_gpu':assert factor.shape==(12,128,128) and factor.dtype==torch.float32
  else:assert factor is None
  self.arm,self.layer,self.factor,self.hm,self.api=arm,layer,factor,hm,api
  self.context,self.event_sink,self.pairs=context,event_sink,0
  self.timing=[]
 def encode_pair(self,key,value):
  tick=time.perf_counter();p.guard();self.pairs+=1;assert self.pairs==self.context['chunk']
  start=end=None
  if key.is_cuda:
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
  z,s=self.api.gpu_frames(key,self.hm)
  assert z.shape==(3,98,16,12,128) and z.dtype==torch.bfloat16 and z.is_contiguous()
  assert s.shape==(3,98,1,12,128) and s.dtype==torch.float16 and s.is_contiguous()
  # One common device-to-host validity decision per encode; no scale repair/fallback.
  assert bool(torch.isfinite(z).all() & torch.isfinite(s).all() & (s>0).all()), 'Invalid rotated K or nonpositive/nonfinite stored FP16 scale'
  if self.arm=='O_full_gpu':out=self.api.raw('T_code','OBQ_fused',(z,s,self.factor),self.hm)
  else:out=self.api.v1.code_packet(z,s,self.factor,self.arm)
  kp={'kind':'K','shape':(3,98,16,12,128),'packed':out[0],'scale':out[1],'sentinel':out[2]}
  vp=encode_v(value,self.api)
  assert kp['packed'].dtype==torch.uint8 and p.nbytes(kp)==4515841
  if end is not None:end.record()
  self.timing.append((time.perf_counter()-tick,start,end))
  if self.event_sink is not None:self.event_sink({'layer':self.layer,'reader_chunk':self.context['chunk'],
   'source_chunk':self.context['chunk']-1,'actual_encode_pair_count':self.pairs,'K_bytes':p.nbytes(kp),'V_bytes':p.nbytes(vp)})
  return PackedState(kp,self.hm,self.api),PackedState(vp,self.hm,self.api)
 def timing_after_arm_sync(self):
  return {'layer':self.layer,'host_enqueue_seconds':[x[0] for x in self.timing],
   'GPU_event_seconds':[x[1].elapsed_time(x[2])/1000 for x in self.timing if x[1] is not None],
   'per_call_explicit_synchronize':False,'common_scale_validity_host_barrier':True,
   'scope':'K/V encode, common scale-validation barrier, events and host instrumentation; not minimal production throughput'}
