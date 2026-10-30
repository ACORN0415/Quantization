"""Bounded QVG compatibility/correctness probe; no video or speed claims.

Uses an unmodified, revision-pinned external QVG checkout. Result/budget files
are append-preserving: at most two attempts, 600 total wall seconds.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REV = '0601468f2dbba6a17ac7086faec6d41527cad188'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path.home()/'sfq_data/qvg-reuse-reference')
    parser.add_argument('--deps', type=Path, default=Path.home()/'sfq_data/qvg-reuse-deps')
    parser.add_argument('--output', type=Path, default=ROOT/'src/experiments/kv_reuse_preflight')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    budget_file = args.output/'budget.json'
    now = time.time()
    budget = json.loads(budget_file.read_text()) if budget_file.exists() else {
        'start_unix': now, 'wall_limit_s': 600, 'max_attempts': 2, 'attempts': []}
    remaining = 600 - (now-budget['start_unix'])
    if remaining <= 0 or len(budget['attempts']) >= 2:
        raise RuntimeError('Preflight budget exhausted; do not reset for another attempt')
    attempt = len(budget['attempts'])+1
    budget['attempts'].append({'attempt': attempt, 'start_unix': now})
    write(budget_file, budget)
    def expired(*_):
        raise TimeoutError('600-second total wall budget reached')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(remaining)))
    result = {'attempt': attempt, 'scope': 'synthetic codec/read/attention compatibility only',
              'source_revision': REV, 'generation_count': 0, 'performance_claim': False,
              'source': str(args.source), 'status': 'running'}
    try:
        actual = subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.source, text=True).strip()
        assert actual == REV, actual
        assert not subprocess.check_output(['git','status','--porcelain'], cwd=args.source, text=True).strip()
        pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
        assert not pids, f'GPU occupied by {pids}'
        os.environ['TIME_BENCH'] = '0'
        sys.path[:0] = [str(args.deps), str(args.source)]
        import torch
        import triton
        import flash_attn
        from flash_attn import flash_attn_func
        from quant_videogen import (compress_kv_cache, get_quantize_fn,
                                    QuantizeConfig, ChunkedKVCache)
        uc = importlib.import_module('quant_videogen.uncompress')
        torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
        torch.set_num_threads(2)
        torch.manual_seed(20261031)
        result['environment'] = {'torch': torch.__version__, 'triton': triton.__version__,
                                 'flash_attn': flash_attn.__version__, 'cuda': torch.version.cuda,
                                 'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability()}
        result['source_sha256'] = {str(p.relative_to(args.source)): sha(p)
                                  for p in sorted((args.source/'quant_videogen').rglob('*.py'))}
        result['probe_sha256'] = sha(Path(__file__))
        result['plan_sha256'] = sha(ROOT/'src/experiments/KV_REUSE_PREFLIGHT.md')
        cfg = QuantizeConfig(quant_type='triton-nstages-kmeans-int2',
                             quant_block_size=64, num_prq_stages=1,
                             kmeans_max_iters=2, cache_num_k_centroids=256,
                             cache_num_v_centroids=256, asymmetric=False)
        result['codec_config'] = vars(cfg)
        shape = (1,12,4680,128)
        result['input_shape_bhsd'] = shape
        k = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        states = compress_kv_cache(k, v, cfg.quant_type, cfg, get_quantize_fn(cfg.quant_type, cfg))
        del k, v
        def tensors(obj, prefix=''):
            if isinstance(obj, torch.Tensor):
                yield prefix, obj
            elif isinstance(obj, dict):
                for key, value in obj.items(): yield from tensors(value, prefix+'/'+str(key))
            elif isinstance(obj, (list, tuple)):
                for key, value in enumerate(obj): yield from tensors(value, prefix+'/'+str(key))
        def payload_hash():
            h = hashlib.sha256()
            for name, t in tensors(states):
                h.update(name.encode())
                h.update(str((tuple(t.shape),t.dtype)).encode())
                h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
            return h.hexdigest()
        storage = {}
        result['payload_tensors'] = []
        for name, t in tensors(states):
            storage[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
            result['payload_tensors'].append({'name': name, 'shape': list(t.shape),
                                              'dtype': str(t.dtype), 'storage_bytes': t.untyped_storage().nbytes()})
        result['packed_pair_storage_bytes'] = sum(storage.values())
        result['bf16_pair_bytes'] = 2*2*1*12*4680*128
        for state in states:
            assert state['residual_quant'].dtype == torch.uint8
            assert list(state['residual_quant'].shape) == [1,12,4680,32]
            state['info'] = {'quant_config': cfg, 'output_dtype': torch.bfloat16}
        before = payload_hash()

        def exact(a,b):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            return a.shape==b.shape and a.dtype==b.dtype and torch.equal(
                a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

        def reference(state):
            # Independent PyTorch unpack of the fixed INT2/one-stage format.
            packed = state['residual_quant'].cpu().to(torch.int32)
            codes = torch.stack([(packed >> n)&3 for n in (6,4,2,0)],-1).reshape(shape)-1
            scale = state['scales'].cpu().float().repeat_interleave(64,-1)
            centroids = state['centroids_list'][0].cpu().float()
            indices = state['cluster_ids_list'][0].cpu().long()
            gathered = torch.gather(centroids,2,indices[...,None].expand(*shape))
            return (codes.float()*scale+gathered).bfloat16()

        real_decode = uc.triton_prq_dequantize_tensor
        phase = 'reference_validation'
        counts = {}
        def observed(*a, **kw):
            counts[phase] = counts.get(phase,0)+1
            return real_decode(*a, **kw)
        uc.triton_prq_dequantize_tensor = observed
        references = []
        for state in states:
            out = observed(state,64,2,output_dtype=torch.bfloat16)
            expected = reference(state)
            assert exact(out.cpu(),expected), 'Official decoder differs from independent INT2 reconstruction'
            references.append(expected.permute(0,2,1,3).contiguous())
            del out, expected
        result['independent_decode_byte_equal'] = True
        stores = [ChunkedKVCache(1,1560,12,128,3,torch.bfloat16,torch.device('cuda'),layout='BSHD')
                  for _ in range(2)]
        for store,state in zip(stores,states): store.store_quantized(0,4680,state)
        held = None
        result['reads'] = []
        for step in range(5):
            query = torch.randn((1,4680,12,128),device='cuda',dtype=torch.bfloat16)
            reference_output = None
            for arm in ('upstream','single_decode','retain'):
                phase = arm
                if arm=='upstream':
                    pair = tuple(s.read(0,4680) for s in stores)
                else:
                    if arm=='single_decode' or held is None:
                        decoded = tuple(observed(s,64,2,output_dtype=torch.bfloat16)
                                        .permute(0,2,1,3).contiguous() for s in states)
                        if arm=='retain': held = decoded
                    else:
                        decoded = held
                    # Preserve the native one-span read's concatenation/copy.
                    pair = tuple(torch.cat([x],dim=1) for x in decoded)
                    del decoded
                assert all(exact(x.cpu(),y) for x,y in zip(pair,references)), (step,arm)
                out = flash_attn_func(query,*pair,dropout_p=0.0,causal=False)
                if reference_output is None: reference_output=out.detach().cpu()
                else: assert exact(out.cpu(),reference_output), (step,arm,'attention mismatch')
                result['reads'].append({'step':step,'arm':arm,'kv_byte_equal':True,'attention_byte_equal':True})
                del pair,out
            del query,reference_output
        expected_counts = {'reference_validation':2,'upstream':20,'single_decode':10,'retain':2}
        result['actual_decode_calls'] = counts
        assert counts == expected_counts, counts
        assert before == payload_hash(), 'Packed representation changed'
        result['packed_payload_unchanged'] = True
        result['payload_sha256'] = before
        result['max_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['max_reserved_bytes'] = torch.cuda.max_memory_reserved()
        result['memory_scope'] = 'whole synthetic probe incl. correctness temporaries; not an arm comparison'
        result['full_generation_latent_validated'] = False
        result['status'] = 'pass'
        print(json.dumps({key:result[key] for key in ('status','packed_pair_storage_bytes',
              'bf16_pair_bytes','actual_decode_calls','max_allocated_bytes')},indent=2),flush=True)
    except BaseException as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        signal.alarm(0)
        result['elapsed_s'] = time.time()-now
        result['completed_unix'] = time.time()
        write(args.output/f'attempt_{attempt:02d}.json',result)
        budget['attempts'][-1].update(status=result['status'],elapsed_s=result['elapsed_s'])
        budget['elapsed_s'] = time.time()-budget['start_unix']
        write(budget_file,budget)


if __name__=='__main__': main()
