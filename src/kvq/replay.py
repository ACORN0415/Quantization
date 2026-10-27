"""Development helpers for reference-write replay and storage-noise controls."""
import copy
import torch
from .store import RollingStore, ChunkStore


class ReferenceReplay:
    """One layer, driven only by unmodified teacher K/V writes."""
    def __init__(self, store):
        self.store = store
        self.calls = 0

    def __call__(self, key, value, start, *, local=True):
        if isinstance(self.store, RollingStore):
            k, v = self.store.write(key, value, start, local=local)
            self.store.commit(k, v)
        else:
            self.store.attend(key, key, value, start, lambda q,k,v: None, local=local)
        self.calls += 1

    def snapshot(self):
        return copy.deepcopy(self.store)


def store_cache(stores, device):
    return [dict(store=s, k=torch.empty(0,device=device,dtype=torch.bfloat16),
                 v=torch.empty(0,device=device,dtype=torch.bfloat16), layer=i,
                 global_end_index=torch.tensor([s.global_end],device=device),
                 local_end_index=torch.tensor([s.local_end],device=device))
            for i,s in enumerate(stores)]


class RawState:
    def __init__(self, tensor): self.tensor = tensor.clone()
    def restore(self): return self.tensor


class StorageNoiseQuantizer:
    """Match each Q(x)-x norm; persistent isotropic noise, private RNG.

    This control does NOT match accumulated read-history error. BF16 rounding
    changes the realized norm, which is recorded rather than silently ignored.
    The target quantizer runs on this arm's input, not another trajectory.
    """
    def __init__(self, target, seed=987654, support='all'):
        if support not in ('all','quantization_error'):
            raise ValueError('unknown noise support')
        self.target, self.seed, self.support = target, seed, support
        self.generators = {}
        self.records = []

    def encode_pair(self, k, v):
        targets = self.target.encode_pair(k, v)
        states = []
        for name,x,encoded in zip(('k','v'),(k,v),targets):
            error = encoded.restore().float()-x.float()
            target_norm = error.norm()
            if target_norm.item() == 0:
                result=x.clone()
            else:
                device=str(x.device)
                if device not in self.generators:
                    self.generators[device]=torch.Generator(device=x.device).manual_seed(self.seed)
                noise=torch.randn(x.shape,device=x.device,dtype=torch.float32,
                                  generator=self.generators[device])
                if self.support=='quantization_error':
                    noise.mul_(error.ne(0))
                noise.mul_(target_norm/noise.norm().clamp_min(1e-30))
                result=(x.float()+noise).to(x.dtype)
            actual=(result.float()-x.float()).norm().item()
            self.records.append(dict(kind=name,target_norm=target_norm.item(),actual_norm=actual,
                                     support=self.support))
            states.append(RawState(result))
        return tuple(states)


def make_store(policy, quantizer, capacity, sink=0):
    return {'rolling':RollingStore,'chunk':ChunkStore}[policy](quantizer,capacity,sink)
