"""QVG read-only reuse adapter. The external codec/attention are unchanged."""
from contextlib import contextmanager
import torch
from quant_videogen.kv_cache import ChunkedKVCache, ChunkState
from quant_videogen.functions import triton_prq_dequantize_tensor


def state_version(state):
    def walk(x):
        if isinstance(x,torch.Tensor):return (id(x),x._version,tuple(x.shape),str(x.dtype),str(x.device))
        if isinstance(x,(list,tuple)):return tuple(walk(v) for v in x)
        if isinstance(x,dict):return tuple((k,walk(v)) for k,v in x.items() if k!='info')
        return x
    return walk(state)


class ReusePool:
    def __init__(self, capacity, device='cuda'):
        self.capacity = capacity
        self.storage = torch.empty(capacity//2, dtype=torch.bfloat16, device=device) if capacity else None
        self.used = 0
        self.entries = {}
        self.requests = 0
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def get(self, layer, kind, span, decode):
        self.requests += 1
        shape = (1, (span['end_chunk']-span['start_chunk'])*1560, 12, 128)
        n = shape[0]*shape[1]*shape[2]*shape[3]
        key = (layer,span['start_chunk'],span['end_chunk'])
        entry = self.entries.get(key)
        if entry is None:
            # Reserve the complete K/V pair, never a dangling half pair.
            if self.used+4*n > self.capacity:
                self.misses += 1
                return decode()
            offset = self.used//2
            self.used += 4*n
            entry = {'views': [self.storage[offset:offset+n].view(shape),
                               self.storage[offset+n:offset+2*n].view(shape)],
                     'states': [None,None], 'versions': [None,None]}
            self.entries[key] = entry
        index = 0 if kind=='k' else 1
        state = span['quant_data']
        # Keep a strong reference as well as identity to prevent id reuse.
        version=state_version(state)
        if entry['states'][index] is state and entry['versions'][index]==version:
            self.hits += 1
            return entry['views'][index]
        self.misses += 1
        value = decode()
        entry['views'][index].copy_(value)
        entry['states'][index] = state
        entry['versions'][index] = version
        return entry['views'][index]

    def invalidate(self, layer, kind, start, end):
        index = 0 if kind=='k' else 1
        for (l,s,e),entry in self.entries.items():
            if l==layer and s<end and e>start and entry['states'][index] is not None:
                entry['states'][index] = None
                self.invalidations += 1
        # Slots stay reserved in this first fixed-span policy. No dynamic allocator.

    def stats(self):
        return dict(capacity_bytes=self.capacity,reserved_pair_bytes=self.used,
                    requests=self.requests,hits=self.hits,misses=self.misses,
                    invalidations=self.invalidations,pairs=len(self.entries))


class ReadController:
    def __init__(self, arm, pool=None, profile=False):
        self.arm=arm; self.pool=pool; self.profile=profile
        self.decode_calls=0; self.read_calls=0; self.events=[]; self.layers={}
        self.stage='generation'

    @contextmanager
    def event(self, name):
        if not self.profile:
            yield
            return
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record()
        try: yield
        finally:
            b.record(); self.events.append((name,a,b))

    def decode(self, state):
        self.decode_calls+=1
        with self.event('decoder'):
            return triton_prq_dequantize_tensor(state,64,2,output_dtype=torch.bfloat16)

    def summarize(self):
        times={}
        for name,a,b in self.events:times[name]=times.get(name,0)+a.elapsed_time(b)/1000
        return dict(arm=self.arm,decode_calls=self.decode_calls,read_calls=self.read_calls,
                    event_seconds=times,event_interpretation='nested GPU intervals, not additive or an upper bound',
                    pool=self.pool.stats() if self.pool else None)


def make_store_class(control):
    class ReuseStore(ChunkedKVCache):
        serial=0
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.layer=ReuseStore.serial//2
            self.kind='k' if ReuseStore.serial%2==0 else 'v'
            ReuseStore.serial+=1

        def _invalidate(self,start,end):
            if control.pool:control.pool.invalidate(self.layer,self.kind,
                                  self._token_to_chunk(start),self._token_to_chunk(end))

        def write(self,start,end,data):
            sc=self._token_to_chunk(start);ec=self._token_to_chunk(end)
            # Native partial overwrite removes the entire span and leaves dangling
            # QUANTIZED flags. Reject unsupported partial overwrite before mutation.
            for span in self.quantized_spans:
                s,e=span['start_chunk'],span['end_chunk']
                if s<ec and e>sc and not (sc<=s and e<=ec):
                    raise ValueError('Partial overwrite of a quantized span is unsupported')
            self._invalidate(start,end)
            return super().write(start,end,data)

        def store_quantized(self,start,end,data):
            self._invalidate(start,end)
            return super().store_quantized(start,end,data)

        def evict_range(self,start,end):
            # Only fully removed native spans lose their held value.
            sc=self._token_to_chunk(start);ec=self._token_to_chunk(end)
            for span in self.quantized_spans:
                if sc<=span['start_chunk'] and span['end_chunk']<=ec:
                    self._invalidate(span['start_chunk']*self.frame_seq_length,
                                     span['end_chunk']*self.frame_seq_length)
            return super().evict_range(start,end)

        def read(self,start,end):
            control.read_calls+=1
            with control.event('read_total'):
                if control.arm=='U':return super().read(start,end)
                sc=self._token_to_chunk(start);ec=self._token_to_chunk(end)
                parts=[];ci=sc
                while ci<ec:
                    if self.chunk_state[ci]==ChunkState.BF16:
                        parts.append(self.chunks[ci]);ci+=1
                    elif self.chunk_state[ci]==ChunkState.QUANTIZED:
                        span=self._find_span(ci)
                        def decode():
                            with control.event('decode_and_layout'):
                                value=control.decode(span['quant_data'])
                                return value.permute(0,2,1,3).contiguous() if self.layout=='BSHD' else value
                        if control.pool:
                            assert self.layout=='BSHD'
                            value=control.pool.get(self.layer,self.kind,span,decode)
                        else:value=decode()
                        off=(ci-span['start_chunk'])*self.frame_seq_length
                        length=(min(span['end_chunk'],ec)-ci)*self.frame_seq_length
                        parts.append(value[:,off:off+length] if self.seq_dim==1 else value[:,:,off:off+length])
                        ci=min(span['end_chunk'],ec)
                    else:raise ValueError('Read of empty chunk')
                with control.event('concat'):
                    return torch.cat(parts,dim=self.seq_dim)
    return ReuseStore
