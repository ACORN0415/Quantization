"""Bounded regression of an unchanged, pinned RobustGEMQ merge kernel.

This tests synthetic merge inputs, not a model or a serving accuracy contract.
Run with --cpu-only to validate the public-option witness without CUDA.
"""
import argparse
import bisect
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import threading
import time
import traceback

import numpy as np
import torch
import triton

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / 'src/experiments/moe_chunk_merge_gpu'
HALF_VALUES = [struct.unpack('<e', struct.pack('<H', b))[0] for b in range(0x7c00)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def rounded_exact_half(q):
    """Nearest finite half using exact rational distances; ties to even."""
    sign = 0x8000 if q < 0 else 0
    q = abs(q)
    assert q <= Fraction.from_float(HALF_VALUES[-1])
    i = bisect.bisect_left(HALF_VALUES, float(q))
    candidates = range(max(0, i-1), min(len(HALF_VALUES), i+2))
    bits = min(candidates, key=lambda b: (abs(Fraction.from_float(HALF_VALUES[b])-q), b & 1))
    return struct.unpack('<e', struct.pack('<H', bits | sign))[0]


def fixture(reg, case):
    f = reg['fixtures']
    t, k, h = f['tokens'], f['top_k'], f['hidden']
    ids = np.tile(np.array(f['expert_ids_each_token'], dtype=np.int64), (t, 1))
    weights = np.tile(np.array(f['routing_weights_each_token'], dtype=np.float16), (t, 1))
    values = np.zeros((t, k, h), dtype=np.float16)
    base = np.array(f['token_0_vectors'][case], dtype=np.float16)
    values[0, :, 0::4] = base[:, None]
    values[0, :, 1::4] = np.arange(1, k+1, dtype=np.float16)[:, None]
    values[0, :, 3::4] = -base[:, None]
    assert all(len(set(row)) == k for row in ids.tolist())
    assert np.all(weights.astype(np.float64).sum(axis=1) == 1)
    order = np.argsort(ids.reshape(-1), kind='stable')
    inverse = np.empty_like(order)
    inverse[order] = np.arange(t*k)
    bank = values.reshape(t*k, h)[order].copy()
    return ids, weights, values, bank, inverse.reshape(t, k), order


def cpu_reference(values, weights, ids, inverse, chunk_tokens, canonical=False):
    t, k, h = values.shape
    result = np.zeros((t, h), dtype=np.float16)
    limit = chunk_tokens * k
    for token in range(t):
        slots = sorted(range(k), key=lambda s: int(ids[token, s])) if canonical else list(range(k))
        # Four repeated coordinate families make checking each coordinate explicit and cheap.
        for lane in range(4):
            a = 0.0
            for start in range(0, t*k, limit):
                for slot in slots:
                    v = float(values[token, slot, lane]) if start <= inverse[token, slot] < start+limit else 0.0
                    product = v * float(weights[token, slot])
                    assert fp32(product) == product  # Exact for the registered operands.
                    a = fp32(a + product)
            result[token, lane::4] = np.float16(a)
    return result


def exact_reference(values, weights):
    t, k, h = values.shape
    result = np.zeros((t, h), dtype=np.float16)
    sums = []
    for token in range(t):
        row = []
        for lane in range(4):
            q = sum((Fraction.from_float(float(values[token, s, lane])) *
                     Fraction.from_float(float(weights[token, s])) for s in range(k)), Fraction())
            result[token, lane::4] = rounded_exact_half(q)
            row.append(dict(numerator=q.numerator, denominator=q.denominator,
                            decimal=float(q), correctly_rounded_fp16=rounded_exact_half(q)))
        sums.append(row)
    return result, sums


def equal_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def cpu_checks(reg):
    rows = {}
    for case in reg['fixtures']['token_0_vectors']:
        ids, w, v, bank, inv, order = fixture(reg, case)
        exact, sums = exact_reference(v, w)
        slots = cpu_reference(v, w, ids, inv, 4)
        local = {c: cpu_reference(v, w, ids, inv, c) for c in [1, 2, 4]}
        canonical = {c: cpu_reference(v, w, ids, inv, c, True) for c in [1, 2, 4]}
        assert not equal_bytes(local[1], slots)
        assert equal_bytes(local[2], slots) and equal_bytes(local[4], slots)
        assert all(equal_bytes(x, canonical[1]) for x in canonical.values())
        assert equal_bytes(canonical[1], exact)  # This fixture only; not a general accuracy theorem.
        rows[case] = dict(exact_sums=sums, original_slot_value=float(slots[0, 0]),
                          chunk_values={c: float(x[0, 0]) for c, x in local.items()},
                          expert_order_values={c: float(x[0, 0]) for c, x in canonical.items()},
                          inverse_order=inv.tolist(), sorted_assignment_order=order.tolist())
    assert rows['rounding_boundary']['exact_sums'][0][0]['decimal'] == 1.0004885792732239
    return rows


def run_gpu(reg, cpu):
    deadline = datetime.fromisoformat(reg['deadline_utc'].replace('Z', '+00:00')).timestamp()
    stage_start = time.monotonic()
    finished = threading.Event()
    calls, rows = [], []
    context = {}

    def guard():
        while not finished.wait(0.2):
            if time.time() >= deadline or time.monotonic()-stage_start >= reg['caps']['gpu_stage_wall_seconds']:
                save_json(RECORD/'resource_stop.json', dict(reason='wall cap', calls=calls, completed_rows=rows))
                os._exit(90)

    threading.Thread(target=guard, daemon=True).start()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(2 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    source = REPO / reg['source']['local_path']
    assert sha(source) == reg['source']['sha256']
    spec = importlib.util.spec_from_file_location('pinned_robustgemq_merge', source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    source_kernels = ['stable_count_offsets_kernel', 'stable_scatter_dispatch_kernel',
                      'fused_weighted_unpermute_reduce_kernel']
    restored = []
    for name in source_kernels:
        kernel = getattr(upstream, name)
        original = kernel.run
        restored.append((kernel, original))

        def wrap(*args, _original=original, _name=name, **kwargs):
            if kwargs.get('warmup', False):
                raise AssertionError('Unregistered warmup launch')
            count = sum(c['kernel'] == 'fused_weighted_unpermute_reduce_kernel' for c in calls)
            if _name == 'fused_weighted_unpermute_reduce_kernel' and count >= reg['caps']['merge_kernel_launches']:
                raise RuntimeError('Merge launch cap')
            row = dict(index=len(calls), kernel=_name, context=dict(context),
                       monotonic_since_stage_start=time.monotonic()-stage_start,
                       status='entered', grid=list(kwargs['grid']))
            if _name == 'fused_weighted_unpermute_reduce_kernel':
                row.update(observed_chunk_start=int(args[6]), observed_chunk_end=int(args[7]),
                           expert_output_shape=list(args[0].shape), expert_output_dtype=str(args[0].dtype),
                           gate_dtype=str(args[2].dtype), accumulator_dtype=str(args[3].dtype))
            calls.append(row)
            try:
                answer = _original(*args, **kwargs)
                row['status'] = 'launched'
                return answer
            except Exception as error:
                row.update(status='error', error=repr(error))
                raise

        kernel.run = wrap

    raw_dir = RECORD/'raw'
    raw_dir.mkdir(exist_ok=True)
    raw_manifest = {}
    repeated = {}
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                 torch.profiler.ProfilerActivity.CUDA])
    error = None
    try:
        with profiler:
            ids, weights, _, _, inverse, order = fixture(reg, 'rounding_boundary')
            ids_gpu = torch.from_numpy(ids.copy()).cuda()
            context.update(phase='public_dispatch')
            sorted_tokens_gpu, inverse_gpu, offsets_gpu = upstream.stable_expert_dispatch(ids_gpu, 64)
            assert np.array_equal(inverse_gpu.cpu().numpy().reshape(4, 8), inverse)
            assert np.array_equal(sorted_tokens_gpu.cpu().numpy(), order//8)
            expected_offsets = np.r_[0, np.cumsum(np.bincount(ids.ravel(), minlength=64))].astype(np.int32)
            assert np.array_equal(offsets_gpu.cpu().numpy(), expected_offsets)
            weights_gpu = torch.from_numpy(weights.copy()).cuda()
            # A simple alternate contract, not a change to the upstream kernel.
            perm_gpu = torch.argsort(ids_gpu, dim=1, stable=True)
            canonical_inverse = torch.gather(inverse_gpu.reshape(4, 8), 1, perm_gpu).contiguous().flatten()
            canonical_weights = torch.gather(weights_gpu, 1, perm_gpu).contiguous()
            context.clear()
            for case in reg['fixtures']['token_0_vectors']:
                ids, weights, values, bank, inverse, order = fixture(reg, case)
                slot_reference = cpu_reference(values, weights, ids, inverse, 4)
                exact, _ = exact_reference(values, weights)
                bank_gpu = torch.from_numpy(bank).cuda()
                for key, array in [('bank',bank), ('weights',weights), ('ids',ids), ('inverse',inverse)]:
                    name = f'{case}_input_{key}.bin'
                    (raw_dir/name).write_bytes(array.tobytes())
                    raw_manifest[name] = dict(shape=list(array.shape), dtype=str(array.dtype), sha256=sha(raw_dir/name))
                for repeat in range(2):
                    modes = reg['runs'][f'order_repeat_{repeat}']
                    chunks = reg['runs'][f'chunk_order_repeat_{repeat}']
                    for mode in modes:
                        for chunk_tokens in chunks:
                            context.update(case=case, repeat=repeat, mode=mode, chunk_tokens=chunk_tokens)
                            accumulator = torch.zeros((4, 256), dtype=torch.float32, device='cuda')
                            output = torch.full((4, 256), float('nan'), dtype=torch.float16, device='cuda')
                            limit = chunk_tokens * 8
                            if mode == 'slot_buffer':
                                buffer = torch.full_like(bank_gpu, float('nan'))
                                for start in range(0, 32, limit):
                                    buffer[start:start+limit].copy_(bank_gpu[start:start+limit])
                                upstream.fused_chunk_unpermute_reduce(buffer, inverse_gpu, weights_gpu,
                                                                     accumulator, 0, 32, final_output=output)
                                expected = slot_reference
                            else:
                                inv_gpu = canonical_inverse if mode == 'expert_order' else inverse_gpu
                                gates_gpu = canonical_weights if mode == 'expert_order' else weights_gpu
                                for start in range(0, 32, limit):
                                    end = min(32, start+limit)
                                    upstream.fused_chunk_unpermute_reduce(
                                        bank_gpu[start:end], inv_gpu, gates_gpu, accumulator, start, end,
                                        final_output=output if end == 32 else None)
                                expected = cpu_reference(values, weights, ids, inverse, chunk_tokens,
                                                         canonical=mode == 'expert_order')
                            got = output.cpu().numpy()
                            name = f'{case}_{mode}_chunk{chunk_tokens}_repeat{repeat}.bin'
                            (raw_dir/name).write_bytes(got.tobytes())
                            raw_manifest[name] = dict(shape=list(got.shape), dtype=str(got.dtype), sha256=sha(raw_dir/name))
                            row = dict(case=case, mode=mode, chunk_tokens=chunk_tokens, repeat=repeat,
                                       first_value=float(got[0, 0]), first_fp16_bytes=got[0, 0].tobytes().hex(),
                                       matches_path_cpu_reference=equal_bytes(got, expected),
                                       matches_original_slot_reference=equal_bytes(got, slot_reference),
                                       matches_correctly_rounded_exact_sum=equal_bytes(got, exact),
                                       different_elements_vs_slot=int(np.count_nonzero(got.view(np.uint16) != slot_reference.view(np.uint16))),
                                       max_abs_difference_vs_slot=float(np.abs(got.astype(np.float64)-slot_reference.astype(np.float64)).max()),
                                       merge_only_passes_documented_allclose=bool(np.allclose(got.astype(np.float64),slot_reference.astype(np.float64),atol=.002,rtol=.002)),
                                       raw_file='raw/'+name)
                            rows.append(row)
                            assert np.isfinite(got).all()
                            assert row['matches_path_cpu_reference'], row
                            key = (case, mode, chunk_tokens)
                            if repeat:
                                assert equal_bytes(got, repeated[key]), row
                            else:
                                repeated[key] = got.copy()
                assert equal_bytes(bank_gpu.cpu().numpy(), bank)
            torch.cuda.synchronize()
    except Exception:
        error = traceback.format_exc()
    finally:
        for kernel, original in restored:
            kernel.run = original
        save_json(RECORD/'calls.json', calls)
        save_json(RECORD/'raw_manifest.json', raw_manifest)

    trace_path = RECORD/'cuda_trace.json'
    profiler.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())
    device_kernels = Counter(e.get('name','') for e in trace['traceEvents'] if e.get('cat') == 'kernel')
    target_device_counts = {name: sum(n for key,n in device_kernels.items() if name in key) for name in source_kernels}
    host_counts = dict(Counter(c['kernel'] for c in calls if c['status'] == 'launched'))
    result = dict(status='FAILED' if error else 'PASS', error=error,
                  started_at_utc=datetime.fromtimestamp(time.time()-(time.monotonic()-stage_start),timezone.utc).isoformat(),
                  completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  gpu_stage_wall_seconds=time.monotonic()-stage_start,
                  source_sha256=sha(source), regression_code_sha256=sha(__file__), registration_sha256=sha(RECORD/'registration.json'),
                  environment=dict(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                                   device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability()),
                  host_launch_counts=host_counts, cuda_profiler_target_counts=target_device_counts,
                  all_cuda_kernel_counts=dict(device_kernels), rows=rows,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                  tensor_payload_bytes=dict(common_sorted_expert_outputs=32*256*2, extra_slot_buffer=32*256*2,
                                            accumulator=4*256*4, final_output=4*256*2,
                                            canonical_inverse=4*8*8, canonical_weights=4*8*2),
                  limits='Synthetic merge unit only; resident fixture bank is common. Payloads and process peak are not deployment peak or a performance comparison.',
                  cpu_reference=cpu, raw_trace_sha256=sha(trace_path), model_forwards=0)
    if not error:
        assert len(rows) == 36
        assert host_counts['fused_weighted_unpermute_reduce_kernel'] == 68
        assert sum(host_counts.get(k,0) for k in source_kernels[:2]) == 2
        assert all(host_counts[k] == target_device_counts[k] for k in source_kernels)
    save_json(RECORD/'result.json', result)
    finished.set()
    if error:
        raise RuntimeError(error)
    print(json.dumps({k:result[k] for k in ['status','gpu_stage_wall_seconds','host_launch_counts',
                                           'cuda_profiler_target_counts','peak_cuda_allocated_bytes']},indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    reg = json.loads((RECORD/'registration.json').read_text())
    assert datetime.now(timezone.utc).timestamp() < datetime.fromisoformat(reg['deadline_utc'].replace('Z','+00:00')).timestamp()
    assert sha(REPO/reg['source']['local_path']) == reg['source']['sha256']
    cpu = cpu_checks(reg)
    if args.cpu_only:
        save_json(RECORD/'cpu_reference.json', dict(cases=cpu, code_sha256=sha(__file__), registration_sha256=sha(RECORD/'registration.json')))
        print(json.dumps({case:{k:v for k,v in row.items() if k.endswith('value') or k.endswith('values')} for case,row in cpu.items()},indent=2))
    else:
        run_gpu(reg, cpu)


if __name__ == '__main__':
    main()
