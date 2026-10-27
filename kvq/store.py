"""Per-layer storage. No global state; new prompts reset each store."""
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
        self.state = self.residual = None
        self.global_end = self.local_end = 0

    def restore(self, like):
        if self.state is None:
            return like.new_zeros(like.shape[0], self.capacity, *like.shape[2:]), like.new_zeros(like.shape[0], self.capacity, *like.shape[2:])
        k, v = (s.restore() for s in self.state)
        if self.residual is not None:
            lo, hi, kr, vr = self.residual
            k[:, lo:hi], v[:, lo:hi] = kr, vr
        return k, v

    def write(self, key, value, start, *, local=True):
        k, v = self.restore(key)
        count = key.shape[1]
        end = start + count
        evict = 0
        if local and end > self.global_end and count + self.local_end > self.capacity:
            evict = count + self.local_end - self.capacity
            rolled = self.local_end - evict - self.sink_tokens
            if rolled < 0:
                raise ValueError("chunk and sink exceed cache capacity")
            for buf in (k, v):
                buf[:, self.sink_tokens:self.sink_tokens + rolled] = buf[:, self.sink_tokens + evict:self.sink_tokens + evict + rolled].clone()
        last = self.local_end + end - self.global_end - evict
        first = last - count
        if first < 0 or last > self.capacity:
            raise ValueError("cache write out of bounds")
        k[:, first:last], v[:, first:last] = key, value
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
        k, v = self.write(key, value, start, local=local)
        lo = max(0, self.local_end - self.capacity)
        out = attention(query, k[:, lo:self.local_end], v[:, lo:self.local_end])
        self.commit(k, v)
        if record is not None:
            kr, vr = self.restore(key)
            record.update(k_rel=relative(kr[:, :self.local_end], k[:, :self.local_end]),
                          v_rel=relative(vr[:, :self.local_end], v[:, :self.local_end]),
                          tokens=self.local_end)
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
                seg["raw"] = None
        pairs = [s["raw"] if s["state"] is None else tuple(q.restore() for q in s["state"]) for s in self.segments]
        k = torch.cat([p[0] for p in pairs] + [key], 1)
        v = torch.cat([p[1] for p in pairs] + [value], 1)
        self.global_end, self.local_end = start + count, k.shape[1]
        if record is not None:
            record.update(tokens=k.shape[1], segment_starts=[s["start"] for s in self.segments])
        return attention(query, k, v)
