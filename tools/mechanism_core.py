"""New storage adapters; the registered kvq implementation stays unchanged."""
import math
import torch
from kvq.store import RollingStore, ChunkStore
from kvq.quant import Quantizer, Quantized
from kvq.memory import memory_report


class BorrowedState:
    def __init__(self,x):self.tensor=x
    def restore(self):return self.tensor


class RawHistory(RollingStore):
    def commit(self,k,v):self.state=(BorrowedState(k),BorrowedState(v))


class CountedRolling(RollingStore):
    def reset(self):
        super().reset();self.version_counts={};self.count=None

    def write(self,key,value,start,**kwargs):
        k,v=super().write(key,value,start,**kwargs)
        self.count=key.shape[1];self.version_counts[start]=0
        first=self.global_end-self.local_end
        self.version_counts={s:n for s,n in self.version_counts.items() if s>=first}
        return k,v

    def commit(self,k,v):
        if hasattr(self.quantizer,'live_tokens'):self.quantizer.live_tokens=self.local_end
        super().commit(k,v)
        for s in self.version_counts:self.version_counts[s]+=1


class ChunkAdapter:
    """Same pending/eviction rules as ChunkStore, with optional history refresh.

    Re-encode existing segments before sealing the newly previous pending,
    avoiding an extra encode of that new history segment on its first read.
    No insertion-error statistics are silently calculated in performance mode.
    """
    def __init__(self,quantizer,capacity,repeat=False):
        self.quantizer=quantizer;self.capacity=capacity;self.repeat=repeat;self.reset()

    def reset(self):
        self.segments=[];self.pending=None;self.pending_start=None
        self.global_end=self.local_end=0;self.chunk_tokens=None

    def attend(self,query,key,value,start,attention,*,local=True,record=None):
        count=key.shape[1]
        if self.chunk_tokens is None:self.chunk_tokens=count
        if count!=self.chunk_tokens or self.capacity%count:raise ValueError('unaligned chunk/cache')
        if self.pending_start is not None and start!=self.pending_start:
            if start!=self.pending_start+count:raise ValueError('nonsequential chunk')
            # Eviction precedes reads. Encoding a discarded segment cannot
            # affect any surviving segment because groups are chunk-local.
            while len(self.segments)>=self.capacity//count-1:self.segments.pop(0)
        if self.repeat:
            for seg in self.segments:
                pair=tuple(s.restore() for s in seg['state'])
                seg['state']=self.quantizer.encode_pair(*pair);seg['encodes']+=1
        if self.pending_start is not None and start!=self.pending_start:
            self.segments.append(dict(start=self.pending_start,state=self.quantizer.encode_pair(*self.pending),encodes=1))
        self.pending_start=start;self.pending=(key.clone(),value.clone())
        self.global_end=start+count
        pairs=[tuple(s.restore() for s in seg['state']) for seg in self.segments]
        k=torch.cat([p[0] for p in pairs]+[key],1)
        v=torch.cat([p[1] for p in pairs]+[value],1)
        self.local_end=k.shape[1]
        if getattr(self,'history_observer',None) is not None:self.history_observer(k[:,:-count],v[:,:-count])
        return attention(query,k,v)


class ObservedQuantizer:
    def __init__(self,group=16,bits=4,observe=True):
        self.target=Quantizer('rtn',bits,bits,group);self.observe=observe;self.events=[];self.calls=0
        self.live_tokens=None

    def encode_pair(self,k,v):
        states=self.target.encode_pair(k,v);self.calls+=1
        if self.observe:
            for kind,x,state in zip(('k','v'),(k,v),states):
                n=x.shape[1] if self.live_tokens is None else min(self.live_tokens,x.shape[1])
                restored=state.restore();diff=(restored[:,:n].float()-x[:,:n].float())
                scale=state.scale[:,:math.ceil(n/self.target.group)]
                flat=scale.float().flatten();quantiles=torch.quantile(flat,flat.new_tensor([.5,.95]))
                # Per-event innovation is distinct from total insertion error.
                item=dict(kind=kind,elements=x.numel(),innovation_norm=float(diff.norm()),
                          input_norm=float(x[:,:n].float().norm()),zero_scale=int((scale==0).sum()),
                          scale_elements=scale.numel(),scale_min=float(flat.min()),scale_max=float(flat.max()),
                          scale_median=float(quantiles[0]),scale_p95=float(quantiles[1]),live_tokens=n)
                # Codes for padding are excluded from the zero-code fraction.
                valid=state.codes.reshape(x.shape[0],-1,*x.shape[2:])[:,:n]
                item['valid_zero_code_fraction']=float((valid==0).float().mean())
                self.events.append(item)
        return states


def make(arm,group=16,bits=4,capacity=32760,observe=False,identity=None):
    q=identity if identity is not None else ObservedQuantizer(group,bits,observe)
    if arm=='L':return CountedRolling(q,capacity)
    if arm=='C_repeat':return ChunkAdapter(q,capacity,True)
    if arm=='C_once':return ChunkAdapter(q,capacity,False)
    raise ValueError(arm)


def histories(store):
    if isinstance(store,RollingStore):
        return {s:n for s,n in store.version_counts.items() if s<store.global_end-store.count}
    return {s['start']:s['encodes'] for s in store.segments}


def state_bytes(store):
    payload=getattr(store,'state',None) if isinstance(store,RollingStore) else [s['state'] for s in store.segments]
    pending=getattr(store,'pending',None)
    def walk(x):
        if isinstance(x,Quantized):yield x
        elif isinstance(x,(tuple,list)):
            for v in x:yield from walk(v)
    valid=padding=scale=0
    for q in walk(payload):
        logical=math.prod(q.shape);valid+=logical*q.codes.element_size()
        padding+=(q.codes.numel()-logical)*q.codes.element_size()
        scale+=q.scale.numel()*q.scale.element_size()
    report=memory_report([payload,pending,getattr(store,'residual',None)])
    pending_bytes=memory_report(pending)['resident_tensor_bytes']
    report.update(valid_code_bytes=valid,padding_code_bytes=padding,scale_bytes=scale,
                  bf16_pending_bytes=pending_bytes,bf16_residual_bytes=0)
    if report['resident_tensor_bytes']!=valid+padding+scale+pending_bytes:
        raise RuntimeError('storage component total mismatch')
    return report
