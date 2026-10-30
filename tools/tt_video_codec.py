"""V-only, once-per-completed-chunk video codecs. K/current chunk stay BF16."""
import torch
import codec_order_core as core
from change_mixed_core import encode_uniform

class State:
 def __init__(self,payload,kind,owner):self.payload=payload;self.kind=kind;self.owner=owner
 def restore(self):
  if self.kind=='raw':return self.payload
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
  x=self.payload.restore() if self.kind=='quant' else core.decode(self.payload)
  x=x.to(torch.bfloat16);end.record();self.owner.events.append(('restore',start,end))
  return x
 def resident(self):
  if self.kind=='raw':return self.payload.untyped_storage().nbytes()
  if self.kind=='quant':return self.payload.bytes()['resident']
  return core.payload_bytes(self.payload)

class VCodec:
 def __init__(self,arm,bits=2):
  assert arm in ('identity','tt','quant','quant_tt','tt_quant')
  self.arm=arm;self.bits=bits;self.events=[];self.encode_calls=0;self.ranks=set()
 def encode_pair(self,k,v):
  self.encode_calls+=1
  if self.arm=='identity':return State(k,'raw',self),State(v,'raw',self)
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
  if self.arm=='quant':state=State(encode_uniform(v,bits=self.bits,group=16),'quant',self)
  else:
   if v.shape[0]!=1 or v.shape[1]%3:raise ValueError('requires one batch, three latent frames per chunk')
   _,n,h,d=v.shape;shape=(h,3,n//3,d)
   budget=(n*h*d*self.bits+7)//8+((n+15)//16)*h*d*2
   storage_bits=self.bits if self.arm=='tt_quant' else None
   rank=core.choose_rank('tt',shape,budget,storage_bits);self.ranks.add(rank)
   source=encode_uniform(v,bits=self.bits,group=16).restore() if self.arm=='quant_tt' else v.float()
   x=source[0].reshape(3,n//3,h,d).permute(2,0,1,3).contiguous()
   state=State(core.store(core.decompose(x,'tt',rank),storage_bits),'tt',self)
   assert state.resident()<=budget
  end.record();self.events.append(('encode',start,end))
  return State(k,'raw',self),state
 def drain(self):
  # Caller synchronizes once per completed chunk.
  result={'encode_seconds':0.,'restore_seconds':0.,'encode_events':0,'restore_events':0}
  for kind,a,b in self.events:
   result[kind+'_seconds']+=a.elapsed_time(b)/1000
   result[kind+'_events']+=1
  self.events.clear();return result

def cache_bytes(cache):
 total=kbytes=vbytes=pending=0
 for c in cache:
  if 'store' not in c:
   kbytes+=c['k'].untyped_storage().nbytes();vbytes+=c['v'].untyped_storage().nbytes();continue
  store=c['store']
  for seg in store.segments:
   if seg['state'] is None:
    a,b=seg['raw'];kbytes+=a.untyped_storage().nbytes();vbytes+=b.untyped_storage().nbytes()
   else:
    a,b=seg['state'];kbytes+=a.resident();vbytes+=b.resident()
  if store.pending:
   a,b=store.pending;ka=a.untyped_storage().nbytes();vb=b.untyped_storage().nbytes()
   kbytes+=ka;vbytes+=vb;pending+=ka+vb
 return dict(k_bytes=kbytes,v_bytes=vbytes,total_bytes=kbytes+vbytes,pending_kv_bytes=pending)

def history_error(cache):
 error=norm=0.
 for c in cache:
  if 'store' not in c:continue
  for s in c['store'].segments:
   if s['state'] is not None:
    e,n=s['error_stats'][1];error+=e;norm+=n
 return (error/max(norm,1e-30))**.5
