"""Independent small-shape gates for V-only full-video storage adapters.

CPU tests exercise actual ChunkStore, including repeated current-chunk writes and
window eviction. CUDA tests exercise every compressed arm without loading a model.
Run CPU alone with --cpu-only while another worker owns the GPU.
"""
import gc
import math
import sys
import unittest
import weakref
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kvq.store import ChunkStore
from tt_video_codec import VCodec,cache_bytes,history_error
from change_mixed_core import encode_uniform
import codec_order_core as core

CPU_ONLY='--cpu-only' in sys.argv
if CPU_ONLY:sys.argv.remove('--cpu-only')


def dense_attention(q,k,v):
    q=q.float().transpose(1,2);k=k.float().transpose(1,2);v=v.float().transpose(1,2)
    weights=(q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])).softmax(-1)
    return (weights@v).transpose(1,2)


def tensors(value):
    """Walk payload only, avoiding the owner's profiling metadata."""
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,dict):
        for x in value.values():yield from tensors(x)
    elif isinstance(value,(list,tuple)):
        for x in value:yield from tensors(x)
    elif is_dataclass(value):
        for f in fields(value):yield from tensors(getattr(value,f.name))


def physical_bytes(value):
    seen=set();total=0
    for x in tensors(value):
        storage=x.untyped_storage();key=(str(x.device),storage.data_ptr(),storage.nbytes())
        if key not in seen:total+=storage.nbytes();seen.add(key)
    return total


class CPUStorageGates(unittest.TestCase):
    def test_identity_repeated_write_eviction_and_bytes(self):
        torch.manual_seed(317)
        count,heads,dim,window=6,2,4,3
        quant=VCodec('identity');store=ChunkStore(quant,window*count)
        completed=[]
        for chunk in range(8):
            for call in range(5):
                key=torch.randn(1,count,heads,dim).to(torch.bfloat16)
                value=torch.randn_like(key)
                query=torch.randn(1,2,heads,dim).to(torch.bfloat16)
                reference=(completed+[(key,value)])[-window:]
                allkey=torch.cat([x[0] for x in reference],1)
                allvalue=torch.cat([x[1] for x in reference],1)
                expected=dense_attention(query,allkey,allvalue)
                result=store.attend(query,key,value,chunk*count,dense_attention)
                self.assertTrue(torch.equal(result,expected),(chunk,call))
                self.assertEqual(quant.encode_calls,chunk)
                self.assertEqual(store.global_end,(chunk+1)*count)
                self.assertEqual(store.local_end,min(chunk+1,window)*count)
                self.assertEqual([x['start'] for x in store.segments],
                                 [c*count for c in range(max(0,chunk-window+1),chunk)])
                self.assertTrue(all(x['raw'] is None for x in store.segments))
                # Mutating caller buffers after the write must not alter pending.
                key_saved=key.clone();value_saved=value.clone()
                key.zero_();value.zero_()
                self.assertTrue(torch.equal(store.pending[0],key_saved))
                self.assertTrue(torch.equal(store.pending[1],value_saved))
                memory=cache_bytes([{'store':store}])
                each=count*heads*dim*2
                self.assertEqual(memory['k_bytes'],min(chunk+1,window)*each)
                self.assertEqual(memory['v_bytes'],min(chunk+1,window)*each)
                self.assertEqual(memory['pending_kv_bytes'],2*each)
                self.assertEqual(history_error([{'store':store}]),0.)
            completed.append((key_saved,value_saved))
            completed=completed[-window:]
        self.assertEqual(quant.drain(),dict(encode_seconds=0.,restore_seconds=0.,encode_events=0,restore_events=0))

    def test_evicted_state_is_released(self):
        codec=VCodec('identity');store=ChunkStore(codec,12)
        for chunk in range(2):
            x=torch.full((1,6,1,2),float(chunk),dtype=torch.bfloat16)
            store.attend(x,x,x,chunk*6,lambda q,k,v:v.sum())
        old=weakref.ref(store.segments[0]['state'][1])
        x=torch.ones(1,6,1,2,dtype=torch.bfloat16)
        store.attend(x,x,x,12,lambda q,k,v:v.sum())
        gc.collect()
        self.assertIsNone(old(), 'evicted value state retained')
        self.assertEqual(codec.encode_calls,2)

    def test_bf16_baseline_accounting(self):
        key=torch.zeros(1,18,2,4,dtype=torch.bfloat16);value=torch.zeros_like(key)
        report=cache_bytes([{'k':key,'v':value}])
        expected=2*18*2*4*2
        self.assertEqual(report['total_bytes'],expected)
        self.assertEqual(report['pending_kv_bytes'],0)


@unittest.skipUnless(not CPU_ONLY and torch.cuda.is_available(), 'CUDA skipped or unavailable')
class CUDAAdapterGates(unittest.TestCase):
    def test_all_bits_and_arms_payload_and_source_lifetime(self):
        torch.manual_seed(923)
        for bits in (2,3,4):
            for arm in ('quant','tt','quant_tt','tt_quant'):
                with self.subTest(bits=bits,arm=arm):
                    codec=VCodec(arm,bits)
                    k=torch.randn(1,48,2,8,device='cuda',dtype=torch.bfloat16)
                    v=torch.randn_like(k);original=v.clone()
                    source_ref=weakref.ref(v);source_ptr=v.untyped_storage().data_ptr()
                    ks,vs=codec.encode_pair(k,v)
                    self.assertTrue(torch.equal(ks.restore(),k))
                    restored=vs.restore()
                    self.assertEqual(restored.dtype,torch.bfloat16)
                    self.assertEqual(restored.shape,v.shape)
                    self.assertTrue(torch.isfinite(restored).all())
                    budget=encode_uniform(original,bits=bits,group=16).bytes()['resident']
                    self.assertLessEqual(vs.resident(),budget)
                    self.assertEqual(vs.resident(),physical_bytes(vs.payload))
                    self.assertTrue(all(x.untyped_storage().data_ptr()!=source_ptr for x in tensors(vs.payload)))
                    if arm=='quant':
                        self.assertTrue(torch.equal(restored,encode_uniform(original,bits=bits,group=16).restore().bfloat16()))
                    else:
                        # Reconstruct the declared route independently from payload rank.
                        rank=vs.payload['rank']
                        source=(encode_uniform(original,bits=bits,group=16).restore()
                                if arm=='quant_tt' else original.float())
                        tensor=source[0].reshape(3,16,2,8).permute(2,0,1,3).contiguous()
                        expected=core.store(core.decompose(tensor,'tt',rank),bits if arm=='tt_quant' else None)
                        self.assertTrue(torch.equal(restored,core.decode(expected).bfloat16()))
                    del v
                    gc.collect()
                    self.assertIsNone(source_ref(), 'compressed V retains source tensor')
                    torch.cuda.synchronize();profile=codec.drain()
                    self.assertEqual(profile['encode_events'],1)
                    self.assertEqual(profile['restore_events'],1)
                    self.assertGreaterEqual(profile['encode_seconds'],0.)
                    self.assertGreaterEqual(profile['restore_seconds'],0.)
                    self.assertEqual(codec.events,[])

    def test_compressed_store_last_write_and_once_only(self):
        torch.manual_seed(117)
        for arm in ('quant','tt','quant_tt','tt_quant'):
            codec=VCodec(arm,2);store=ChunkStore(codec,3*48)
            completed=[]
            for chunk in range(5):
                for call in range(2):
                    key=torch.randn(1,48,2,8,device='cuda',dtype=torch.bfloat16)
                    value=torch.randn_like(key)
                    expected_past=[]
                    for k0,v0 in completed[-2:]:
                        # Separate adapter constructs the expected encoded read once.
                        checker=VCodec(arm,2);a,b=checker.encode_pair(k0,v0)
                        expected_past.append((a.restore(),b.restore()))
                    ek=torch.cat([p[0] for p in expected_past]+[key],1)
                    ev=torch.cat([p[1] for p in expected_past]+[value],1)
                    seen=[]
                    store.attend(key,key,value,chunk*48,lambda q,k,v:seen.extend((k.clone(),v.clone())))
                    self.assertTrue(torch.equal(seen[0],ek))
                    self.assertTrue(torch.equal(seen[1],ev))
                    self.assertEqual(codec.encode_calls,chunk)
                    self.assertTrue(torch.equal(store.pending[1],value))
                completed.append((key.clone(),value.clone()))
            torch.cuda.synchronize();codec.drain()


if __name__=='__main__':unittest.main()
