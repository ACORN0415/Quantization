"""Per-layer storage. No global state; new prompts reset each store."""
import math
import torch


def relative(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


class RollingStore:
    def __init__(self, quantizer, capacity, sink_tokens=0, residual_tokens=0):
        if capacity <= 0 or not 0 <= sink_tokens < capacity or residual_tokens < 0:
            raise ValueError("invalid capacity, sink or residual")
        self.quantizer, self.capacity = quantizer, capacity
        self.sink_tokens, self.residual_tokens = sink_tokens, residual_tokens
        self.reset()

    def reset(self):
        self.state = self.residual = self.source = None
        self.global_end = self.local_end = 0

    def restore(self, like):
        if self.state is None:
            return like.new_zeros(like.shape[0], self.capacity, *like.shape[2:]), like.new_zeros(like.shape[0], self.capacity, *like.shape[2:])
        k, v = (s.restore() for s in self.state)
        if self.residual is not None:
            lo, hi, kr, vr = self.residual
            k[:, lo:hi], v[:, lo:hi] = kr, vr
        return k, v

    def write(self, key, value, start, *, local=True, track_source=False):
        track_source = track_source or self.source is not None
        k, v = self.restore(key)
        if track_source and self.source is None:
            if self.state is not None:
                raise ValueError('source logging must start at the first cache write')
            self.source = (torch.zeros_like(k), torch.zeros_like(v))
        count = key.shape[1]
        end = start + count
        evict = 0
        if local and end > self.global_end and count + self.local_end > self.capacity:
            evict = count + self.local_end - self.capacity
            rolled = self.local_end - evict - self.sink_tokens
            if rolled < 0:
                raise ValueError("chunk and sink exceed cache capacity")
            for buf in ((k, v) + self.source if track_source else (k, v)):
                buf[:, self.sink_tokens:self.sink_tokens + rolled] = buf[:, self.sink_tokens + evict:self.sink_tokens + evict + rolled].clone()
        last = self.local_end + end - self.global_end - evict
        first = last - count
        if first < 0 or last > self.capacity:
            raise ValueError("cache write out of bounds")
        k[:, first:last], v[:, first:last] = key, value
        if track_source:
            self.source[0][:, first:last], self.source[1][:, first:last] = key, value
        self.global_end, self.local_end = end, last
        return k, v

    def commit(self, k, v):
        # Preserve the newest live tokens, rather than the physical buffer tail.
        n = min(self.residual_tokens, self.local_end)
        self.residual = (self.local_end - n, self.local_end,
                         k[:, self.local_end-n:self.local_end].clone(),
                         v[:, self.local_end-n:self.local_end].clone()) if n else None
        self.state = self.quantizer.encode_pair(k, v)

    def attend(self, query, key, value, start, attention, *, local=True, record=None):
        k, v = self.write(key, value, start, local=local, track_source=record is not None)
        lo = max(0, self.local_end - self.capacity)
        if record is not None:
            history_end=self.local_end-key.shape[1]
            ke=relative(k[:,lo:history_end],self.source[0][:,lo:history_end])
            ve=relative(v[:,lo:history_end],self.source[1][:,lo:history_end])
            record.update(k_rel=ke,v_rel=ve,read_history_k_rel=ke,read_history_v_rel=ve,
                          read_history_tokens=history_end-lo,read_current_tokens=key.shape[1],
                          kv_error_phase='pre_attention_history',kv_error_reference='insertion_bf16_history',
                          kv_error_schema=2)
        out = attention(query, k[:, lo:self.local_end], v[:, lo:self.local_end])
        self.commit(k, v)
        if record is not None:
            kr, vr = self.restore(key)
            record.update(legacy_post_k_rel=relative(kr[:, :self.local_end], k[:, :self.local_end]),
                          legacy_post_v_rel=relative(vr[:, :self.local_end], v[:, :self.local_end]),
                          tokens=self.local_end,
                          legacy_post_reference='pre_requantization_buffer',
                          source_error_phase='post_storage',
                          source_k_rel=relative(kr[:, :self.local_end],self.source[0][:, :self.local_end]),
                          source_v_rel=relative(vr[:, :self.local_end],self.source[1][:, :self.local_end]),
                          source_reference='insertion_bf16',
                          diagnostic_bf16_bytes=sum(t.numel()*t.element_size() for t in self.source))
        return out


class ChunkStore:
    def __init__(self, quantizer, capacity, sink_tokens=0, residual_tokens=0):
        if capacity <= 0 or not 0 <= sink_tokens < capacity or residual_tokens < 0:
            raise ValueError("invalid capacity, sink or residual")
        self.quantizer, self.capacity = quantizer, capacity
        self.sink_tokens, self.residual_tokens = sink_tokens, residual_tokens
        self.reset()

    def reset(self):
        self.segments = []
        self.pending = None
        self.pending_start = None
        self.global_end = self.local_end = 0
        self.chunk_tokens = None

    def attend(self, query, key, value, start, attention, *, local=True, record=None):
        count = key.shape[1]
        if self.chunk_tokens is None:
            self.chunk_tokens = count
        if count != self.chunk_tokens or self.capacity % count or self.sink_tokens % count:
            raise ValueError("chunk policy requires fixed chunks and aligned capacity/sink")
        if count + self.sink_tokens > self.capacity:
            raise ValueError("sink and current chunk exceed capacity")
        if self.pending_start is not None and start != self.pending_start:
            if start != self.pending_start + count:
                raise ValueError("chunks must arrive in order")
            self.segments.append({"start": self.pending_start, "raw": self.pending, "state": None})
            while len(self.segments) >= self.capacity // count:
                if not local:
                    raise ValueError("cache capacity exceeded with local eviction disabled")
                self.segments.pop(self.sink_tokens // count)
        self.pending_start, self.pending = start, (key.clone(), value.clone())
        # Complete recent chunks can remain BF16 for a residual window. Encode
        # each only once when it leaves that window. Pending is always BF16.
        cutoff = start - self.residual_tokens
        for seg in self.segments:
            if seg["state"] is None and seg["start"] + count <= cutoff:
                seg["state"] = self.quantizer.encode_pair(*seg["raw"])
                seg["error_stats"] = []
                for original, encoded in zip(seg["raw"],seg["state"]):
                    x=original.float(); restored=encoded.restore().float()
                    seg["error_stats"].append(((restored-x).square().sum().item(),x.square().sum().item()))
                seg["raw"] = None
        pairs = [s["raw"] if s["state"] is None else tuple(q.restore() for q in s["state"]) for s in self.segments]
        k = torch.cat([p[0] for p in pairs] + [key], 1)
        v = torch.cat([p[1] for p in pairs] + [value], 1)
        self.global_end, self.local_end = start + count, k.shape[1]
        if record is not None:
            errors=[0.,0.]; norms=[0.,0.]
            for seg in self.segments:
                if seg["state"] is None:
                    for index,x in enumerate(seg["raw"]): norms[index]+=x.float().square().sum().item()
                else:
                    for index,(error,norm) in enumerate(seg["error_stats"]):
                        errors[index]+=error;norms[index]+=norm
            ke=math.sqrt(errors[0])/max(math.sqrt(norms[0]),1e-12)
            ve=math.sqrt(errors[1])/max(math.sqrt(norms[1]),1e-12)
            stored_norms=[norms[0]+key.float().square().sum().item(),
                          norms[1]+value.float().square().sum().item()]
            record.update(tokens=k.shape[1], segment_starts=[s["start"] for s in self.segments],
                          k_rel=ke,v_rel=ve,read_history_k_rel=ke,read_history_v_rel=ve,
                          read_history_tokens=k.shape[1]-count,read_current_tokens=count,
                          source_k_rel=math.sqrt(errors[0])/max(math.sqrt(stored_norms[0]),1e-12),
                          source_v_rel=math.sqrt(errors[1])/max(math.sqrt(stored_norms[1]),1e-12),
                          kv_error_phase='pre_attention_history',kv_error_reference='insertion_bf16_history',
                          kv_error_schema=2,source_error_phase='post_storage',
                          source_reference='insertion_bf16',diagnostic_bf16_bytes=0)
        return attention(query, k, v)
