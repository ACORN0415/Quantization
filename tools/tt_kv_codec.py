"""Final KV test: compress completed post-RoPE K and V, pending stays BF16.

Both tensors use the same independent codec, without sharing factors/scales.
Direct RTN groups token positions in blocks of 16 for each head/channel. TT
factorization and packing reuse the frozen V implementation without alteration.
"""
import torch
from change_mixed_core import encode_uniform
from tt_rank_codec import encode_payload
from tt_video_codec import State, VCodec


def pair_payloads(k, v, bits, rank):
    """CPU-testable payload construction; returns kind plus independent K/V payloads."""
    if k.shape != v.shape or k.device != v.device:
        raise ValueError('K and V must have equal shapes and device')
    if bits not in (2, 3, 4) or not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError('expected bits 2/3/4 and nonnegative integer rank')
    if rank == 0:
        return 'quant', (encode_uniform(k, bits=bits, group=16),
                         encode_uniform(v, bits=bits, group=16))
    return 'tt', (encode_payload(k, bits, rank), encode_payload(v, bits, rank))


class KVCodec(VCodec):
    def __init__(self, bits, rank):
        if bits == 0 and rank == 0:
            arm = 'identity'
        elif bits in (2, 3, 4) and isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0:
            arm = 'quant' if rank == 0 else 'tt_quant'
        else:
            raise ValueError('identity requires (0,0); compression requires bits 2/3/4 and integer rank >=0')
        super().__init__(arm, bits)
        self.rank = rank

    def encode_pair(self, k, v):
        self.encode_calls += 1
        if self.arm == 'identity':
            return State(k, 'raw', self), State(v, 'raw', self)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kind, payloads = pair_payloads(k, v, self.bits, self.rank)
        if self.rank:
            self.ranks.add(self.rank)
        result = tuple(State(payload, kind, self) for payload in payloads)
        end.record()
        self.events.append(('encode', start, end))
        return result
