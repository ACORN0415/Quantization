"""CPU one-head INT8 / eleven-head INT4 baseline, no fitting or capture I/O.

Arithmetic follows causal_obq/common.py R_fp32: FP32 H128 rotation -> BF16,
frame-local groups, BF16 max/qmax scale -> FP16, FP32 division/round-even,
BF16 code*scale, FP32 inverse H128 -> BF16. Shared metadata is separately owned.
"""
from dataclasses import dataclass
import math
import struct
import torch

HEADER = struct.Struct('<4sIIIII')  # magic, frames, frame_tokens, heads, dim, group
HEADS, DIM, GROUP = 12, 128, 16


def require(ok, why):
    if not ok:
        raise ValueError(why)


def tensor_bytes(x):
    return x.contiguous().view(torch.uint8).numpy().tobytes()


def _read(data, dtype, shape):
    return torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)


def _owned(x):
    require(x.device.type == 'cpu' and x.is_contiguous(), 'CPU contiguous storage required')
    require(not x.requires_grad and x.grad_fn is None, 'autograd state forbidden')
    require(x.storage_offset() == 0 and x.untyped_storage().nbytes() == x.numel()*x.element_size(),
            'compact owned-sized storage required')


@dataclass
class MixedHeadState:
    header: torch.Tensor
    selected_head: torch.Tensor

    def validate(self):
        for t in (self.header, self.selected_head):
            _owned(t)
            require(t.dtype == torch.uint8, 'state dtype')
        require(self.header.shape == (HEADER.size,) and self.selected_head.shape == (1,), 'state shapes')
        magic, frames, tokens, heads, dim, group = HEADER.unpack(tensor_bytes(self.header))
        require(magic == b'H8K1' and frames > 0 and tokens > 0, 'state header')
        require((heads, dim, group) == (HEADS, DIM, GROUP), 'fixed H128/12head/g16 contract')
        head = int(self.selected_head.item())
        require(0 <= head < HEADS, 'selected head')
        return frames, tokens, math.ceil(tokens/GROUP), head

    def nbytes(self):
        self.validate()
        return self.header.numel()+self.selected_head.numel()

    def storage_bytes(self):
        self.validate()
        return sum(t.untyped_storage().nbytes() for t in (self.header, self.selected_head))

    def to_bytes(self):
        self.validate()
        return tensor_bytes(self.header)+tensor_bytes(self.selected_head)

    @classmethod
    def from_bytes(cls, data):
        require(isinstance(data, bytes) and len(data) == HEADER.size+1, 'state byte length')
        obj = cls(_read(data[:HEADER.size], torch.uint8, (HEADER.size,)),
                  _read(data[HEADER.size:], torch.uint8, (1,)))
        obj.validate()
        return obj


@torch.no_grad()
def prepare_state(selected_head, frame_tokens=1560, frames=3):
    require(isinstance(selected_head, int) and 0 <= selected_head < HEADS, 'selected head integer')
    require(isinstance(frames, int) and 0 < frames < 2**32, 'frame count')
    require(isinstance(frame_tokens, int) and 0 < frame_tokens < 2**32, 'frame tokens')
    raw = HEADER.pack(b'H8K1', frames, frame_tokens, HEADS, DIM, GROUP)
    state = MixedHeadState(_read(raw, torch.uint8, (HEADER.size,)),
                           torch.tensor([selected_head], dtype=torch.uint8))
    state.validate()
    return state


def hadamard():
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < DIM:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h/math.sqrt(DIM)


def _pack4(codes):
    require(codes.dtype == torch.int8 and not codes.numel()%2, 'INT4 code shape')
    require(bool(((codes >= -8) & (codes <= 7)).all()), 'INT4 code range')
    raw = codes.contiguous().view(torch.uint8).flatten()
    return ((raw[::2] & 15) | ((raw[1::2] & 15) << 4)).contiguous().clone()


def _unpack4(packed, shape):
    raw = torch.empty(packed.numel()*2, dtype=torch.uint8)
    raw[::2] = packed & 15
    raw[1::2] = packed >> 4
    signed = raw.to(torch.int16)
    signed[signed >= 8] -= 16
    return signed.to(torch.int8).reshape(shape)


@dataclass
class MixedHeadPacket:
    packed4: torch.Tensor
    codes8: torch.Tensor
    scale: torch.Tensor
    sentinel: torch.Tensor

    def validate(self, state):
        frames, tokens, groups, head = state.validate()
        for t in (self.packed4, self.codes8, self.scale, self.sentinel):
            _owned(t)
        count = frames*groups*GROUP*DIM
        require(self.packed4.dtype == torch.uint8 and self.packed4.shape == (count*11//2,), 'INT4 packed shape')
        require(self.codes8.dtype == torch.int8 and self.codes8.shape == (count,), 'INT8 packed shape')
        require(self.scale.dtype == torch.float16 and self.scale.shape == (frames, groups, 1, HEADS, DIM), 'scale shape')
        require(bool((torch.isfinite(self.scale) & (self.scale > 0)).all()), 'positive finite FP16 scales required')
        require(self.sentinel.dtype == torch.uint8 and self.sentinel.shape == (1,) and int(self.sentinel[0]) == 255, 'sentinel')
        return frames, tokens, groups, head

    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in (self.packed4, self.codes8, self.scale, self.sentinel))

    def storage_bytes(self):
        for t in (self.packed4, self.codes8, self.scale, self.sentinel):
            _owned(t)
        return sum(t.untyped_storage().nbytes() for t in (self.packed4, self.codes8, self.scale, self.sentinel))

    def to_bytes(self, state):
        self.validate(state)
        return b''.join(tensor_bytes(t) for t in (self.sentinel, self.packed4, self.codes8, self.scale))

    @classmethod
    def from_bytes(cls, data, state):
        f, n, g, h = state.validate(); count = f*g*GROUP*DIM
        b4, b8, bs = count*11//2, count, f*g*HEADS*DIM*2
        require(isinstance(data, bytes) and len(data) == 1+b4+b8+bs, 'packet byte length')
        obj = cls(_read(data[1:1+b4], torch.uint8, (b4,)),
                  _read(data[1+b4:1+b4+b8], torch.int8, (b8,)),
                  _read(data[1+b4+b8:], torch.float16, (f,g,1,HEADS,DIM)),
                  _read(data[:1], torch.uint8, (1,)))
        obj.validate(state)
        return obj


def _finite_bf16(k, expected=None):
    require(isinstance(k, torch.Tensor) and k.device.type == 'cpu' and k.dtype == torch.bfloat16, 'CPU BF16 key required')
    if expected is not None:
        require(tuple(k.shape) == expected, 'key shape')
    require(bool(torch.isfinite(k).all()), 'nonfinite key')


@torch.no_grad()
def encode(key, state):
    frames, tokens, groups, head = state.validate()
    _finite_bf16(key, (1,frames*tokens,HEADS,DIM))
    z = (key.float() @ hadamard()).bfloat16()[0].reshape(frames,tokens,HEADS,DIM)
    require(bool(torch.isfinite(z).all()), 'rotated key overflow')
    pad = groups*GROUP-tokens
    z = torch.cat((z,z.new_zeros(frames,pad,HEADS,DIM)),1).reshape(frames,groups,GROUP,HEADS,DIM)
    # Preserve the original BF16 scale arithmetic for every non-promoted head.
    maximum = z.abs().amax(2, keepdim=True)
    s = (maximum/7).clamp_min(1e-8)
    s[:,:,:,head:head+1,:] = (maximum[:,:,:,head:head+1,:]/127).clamp_min(1e-8)
    sf = s.to(torch.float16)
    require(bool((torch.isfinite(sf) & (sf > 0)).all()), 'zero/tiny/overflow FP16 scale')
    require(torch.equal(sf.bfloat16(),s), 'FP16 scale does not preserve BF16 scale')
    other = [h for h in range(HEADS) if h != head]
    code4 = (z[:,:,:,other,:].float()/sf[:,:,:,other,:].float()).round().clamp(-8,7).to(torch.int8)
    code8 = (z[:,:,:,head,:].float()/sf[:,:,:,head,:].float()).round().clamp(-128,127).to(torch.int8)
    packet = MixedHeadPacket(_pack4(code4), code8.flatten().contiguous().clone(),
                             sf.contiguous().clone(), torch.tensor([255],dtype=torch.uint8))
    packet.validate(state)
    return packet


@torch.no_grad()
def restore(packet, state):
    frames,tokens,groups,head = packet.validate(state)
    other = [h for h in range(HEADS) if h != head]
    c = torch.empty(frames,groups,GROUP,HEADS,DIM,dtype=torch.int8)
    c[:,:,:,other,:] = _unpack4(packet.packed4,(frames,groups,GROUP,11,DIM))
    c[:,:,:,head,:] = packet.codes8.reshape(frames,groups,GROUP,DIM)
    z = c.bfloat16()*packet.scale.bfloat16()
    z = z.reshape(frames,groups*GROUP,HEADS,DIM)[:,:tokens].reshape(1,frames*tokens,HEADS,DIM)
    result = (z.float() @ hadamard().t()).bfloat16()
    require(bool(torch.isfinite(result).all()), 'restored key overflow')
    return result


@torch.no_grad()
def select_head(raw_moment, original_key, rtn_restored_key):
    """External p0-only original-coordinate moment and K/RTN error; lowest-index ties.

    This function never chooses rank, fits a moment, or accesses evaluation queries.
    Scores are full query-weighted K error, not post-Wo/attention output error.
    """
    require(isinstance(raw_moment,torch.Tensor) and raw_moment.device.type == 'cpu' and
            raw_moment.dtype == torch.float64 and raw_moment.shape == (HEADS,DIM,DIM), 'raw original moment FP64 [12,128,128]')
    require(bool(torch.isfinite(raw_moment).all()), 'nonfinite moment')
    require(torch.allclose(raw_moment,raw_moment.transpose(-1,-2),rtol=1e-10,atol=1e-10), 'moment not symmetric')
    eig = torch.linalg.eigvalsh(raw_moment)
    require(bool((eig[:,0] >= -1e-10*eig.abs().amax(-1).clamp_min(1)).all()), 'moment not PSD')
    _finite_bf16(original_key); _finite_bf16(rtn_restored_key)
    require(original_key.shape == rtn_restored_key.shape, 'K pair shape')
    require((original_key.ndim == 3 and original_key.shape[1:] == (HEADS,DIM)) or
            (original_key.ndim == 4 and original_key.shape[0] == 1 and original_key.shape[2:] == (HEADS,DIM)), 'K [N,12,128] or [1,N,12,128]')
    err = (original_key.float()-rtn_restored_key.float()).reshape(-1,HEADS,DIM).double()
    require(err.shape[0] > 0, 'empty K')
    scores = torch.einsum('nhd,hde,nhe->h',err,raw_moment,err)
    require(bool(torch.isfinite(scores).all()), 'nonfinite head scores')
    return int(torch.argmax(scores)), scores.detach().clone().contiguous()
