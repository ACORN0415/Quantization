"""Fixed offline original-coordinate K mean removal, CPU baseline only.

No online mean update. This public/simple baseline is not a new method.
Shared state is exactly FP32 [12,128]; the original R_fp32 packet is unchanged.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import torch

COMMON_PATH = Path(__file__).resolve().parent.parent/'causal_obq/common.py'
COMMON_SHA256 = '436f7b2762397ecce840b91feb535094a9008c54066e4e796c3df248f9071c35'
if hashlib.sha256(COMMON_PATH.read_bytes()).hexdigest() != COMMON_SHA256:
    raise RuntimeError('Frozen R_fp32 common source changed')
_spec = importlib.util.spec_from_file_location('_mean_frozen_causal_common', COMMON_PATH)
c = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = c
_spec.loader.exec_module(c)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_bytes(x):
    return x.contiguous().view(torch.uint8).numpy().tobytes()


@dataclass
class MeanState:
    mean: torch.Tensor

    def validate(self):
        m=self.mean
        require(isinstance(m,torch.Tensor) and m.device.type=='cpu' and m.dtype==torch.float32 and m.shape==(12,128), 'mean must be CPU FP32 [12,128]')
        require(bool(torch.isfinite(m).all()), 'nonfinite mean')
        require(m.is_contiguous() and m.storage_offset()==0 and m.untyped_storage().nbytes()==6144, 'compact mean storage required')
        require(not m.requires_grad and m.grad_fn is None, 'mean must not retain autograd graph')

    def nbytes(self):
        self.validate()
        return 6144

    def storage_bytes(self):
        self.validate()
        return self.mean.untyped_storage().nbytes()

    def to_bytes(self):
        self.validate()
        return tensor_bytes(self.mean)

    @classmethod
    def from_bytes(cls,data):
        require(isinstance(data,bytes) and len(data)==6144, 'mean byte length')
        state=cls(torch.frombuffer(bytearray(data),dtype=torch.float32).clone().reshape(12,128))
        state.validate()
        return state


@torch.no_grad()
def prepare_mean(mean):
    require(isinstance(mean,torch.Tensor), 'mean tensor required')
    state=MeanState(mean.detach().clone().contiguous())
    state.validate()
    return state


def _key(key, production=False):
    require(isinstance(key,torch.Tensor) and key.device.type=='cpu' and key.dtype==torch.bfloat16, 'CPU BF16 K required')
    require(bool(torch.isfinite(key).all()), 'nonfinite K')
    if production:
        require(key.shape==(1,4680,12,128), 'one completed production chunk required')
    else:
        require((key.ndim==3 and key.shape[1:]==(12,128)) or
                (key.ndim==4 and key.shape[0]==1 and key.shape[2:]==(12,128)), 'K [N,12,128] or [1,N,12,128]')
        require(key.numel()>0, 'empty key')


@torch.no_grad()
def make_mean(keys):
    """Token-weighted FP64 offline accumulation -> owned FP32 mean.

    Caller supplies only the registered p0 past K. No data access or provenance
    inference occurs here. Accept a single tensor or iterable of BF16 tensors.
    """
    if isinstance(keys,torch.Tensor):
        keys=(keys,)
    total=torch.zeros(12,128,dtype=torch.float64); count=0
    for key in keys:
        _key(key)
        flat=key.reshape(-1,12,128)
        total += flat.sum(0,dtype=torch.float64)
        count += flat.shape[0]
    require(count>0, 'no calibration tokens')
    mean=(total/count).float()
    return prepare_mean(mean)


def _packet(packet):
    require(isinstance(packet,dict) and packet.get('kind')=='K', 'original K packet required')
    require(tuple(packet.get('shape',()))==(3,98,16,12,128), 'K packet shape')
    require(isinstance(packet.get('packed'),torch.Tensor) and packet['packed'].device.type=='cpu' and
            packet['packed'].dtype==torch.uint8 and packet['packed'].shape==(3612672,), 'packed INT4 shape')
    scale=packet.get('scale')
    require(isinstance(scale,torch.Tensor) and scale.device.type=='cpu' and scale.dtype==torch.float16 and
            scale.shape==(3,98,1,12,128), 'FP16 scale shape')
    require(bool((torch.isfinite(scale)&(scale>0)).all()), 'zero/negative/tiny/nonfinite stored scale')
    require(torch.equal(scale.bfloat16().half(),scale), 'stored scale not representable in original BF16 scale contract')
    sent=packet.get('sentinel')
    require(isinstance(sent,torch.Tensor) and sent.device.type=='cpu' and sent.dtype==torch.uint8 and
            sent.shape==(1,) and int(sent[0])==255, 'K sentinel')
    for t in (packet['packed'],scale,sent):
        require(t.is_contiguous() and not t.requires_grad and t.grad_fn is None, 'invalid packet storage/graph')


@torch.no_grad()
def encode(key,state):
    state.validate();_key(key,production=True)
    # Exact zero state explicitly preserves the original byte path, including signed zero.
    centered=key if not bool(state.mean.ne(0).any()) else (key.float()-state.mean[None,None]).bfloat16()
    require(bool(torch.isfinite(centered).all()), 'centered BF16 overflow')
    try:
        packet=c.encode_k(centered,'R_fp32')
    except AssertionError as error:
        raise ValueError('Original quantizer shape/FP16 scale roundtrip contract failed') from error
    _packet(packet)
    return packet


@torch.no_grad()
def restore(packet,state):
    state.validate();_packet(packet)
    restored=c.restore(packet)
    require(bool(torch.isfinite(restored).all()), 'base restore overflow')
    if not bool(state.mean.ne(0).any()):
        return restored
    result=(restored.float()+state.mean[None,None]).bfloat16()
    require(bool(torch.isfinite(result).all()), 'restored mean addition overflow')
    return result
