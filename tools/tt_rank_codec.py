"""Fixed-rank V-only TT -> packed-factor RTN for the memory sweep.

The frozen tt_video codec chooses rank from a byte budget; this adapter instead
spends the savings from a smaller rank on a smaller resident payload. K and the
pending chunk stay BF16. Decomposition, factor scales and reconstruction are
identical to the frozen implementation.
"""
import torch
import codec_order_core as core
from tt_video_codec import State, VCodec


def encode_payload(v, bits, rank):
    if bits not in (2, 3, 4):
        raise ValueError('supported factor bits: 2, 3, 4')
    if v.ndim != 4 or v.shape[0] != 1 or v.shape[1] % 3:
        raise ValueError('requires [1, three-frame token count, heads, dim]')
    _, n, h, d = v.shape
    shape = (h, 3, n // 3, d)
    cap = min(shape[2], shape[1] * shape[3])
    if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= cap:
        raise ValueError(f'rank must be an integer in [1, {cap}]')
    x = v.float()[0].reshape(3, n // 3, h, d).permute(2, 0, 1, 3).contiguous()
    payload = core.store(core.decompose(x, 'tt', rank), bits)
    if core.payload_bytes(payload) != core.expected_bytes('tt', shape, rank, bits):
        raise RuntimeError('fixed-rank payload byte accounting mismatch')
    return payload


class FixedRankVCodec(VCodec):
    def __init__(self, bits, rank):
        super().__init__('tt_quant', bits)
        if bits not in (2, 3, 4) or not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError('invalid fixed-rank codec configuration')
        self.rank = rank

    def encode_pair(self, k, v):
        self.encode_calls += 1
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        state = State(encode_payload(v, self.bits, self.rank), 'tt', self)
        self.ranks.add(self.rank)
        end.record()
        self.events.append(('encode', start, end))
        return State(k, 'raw', self), state
