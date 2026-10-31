"""Bounded, synthetic attention execution audit; does not generate videos."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import time
import traceback


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--registration', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / 'results.json'
    if result_path.exists():
        raise RuntimeError('Do not overwrite a previous experiment')
    reg = json.loads(args.registration.read_text())
    result = dict(status='running', started_at=stamp(), pid=os.getpid(),
                  registration_sha256=sha(args.registration), script_sha256=sha(__file__),
                  gpu_model_generations=0, small_gates=[], shapes=[], telemetry=[],
                  replay_rounds=[], call_counts={}, errors=[])
    start = time.monotonic()
    hard_end = min(start + reg['gpu_seconds_limit'], start + max(0,
        datetime.fromisoformat(reg['overall_deadline']).timestamp() - time.time()))
    counts = Counter()
    trace = args.out / 'progress.jsonl'

    def progress(event, **kw):
        row = dict(time=stamp(), event=event, **kw)
        with trace.open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)
        result['call_counts'] = dict(counts)
        dump(result_path, result)

    def deadline():
        if time.monotonic() >= hard_end:
            raise TimeoutError('Registered GPU/overall wall-clock limit reached')

    def telemetry(label):
        fields = 'name,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw'
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=' + fields,
                                      '--format=csv,noheader'], text=True).strip()
        proc = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True).strip()
        foreign = [int(x.strip()) for x in proc.splitlines()
                   if x.strip().isdigit() and int(x.strip()) != os.getpid()]
        result['telemetry'].append(dict(time=stamp(), label=label, fields=fields,
                                        values=gpu, other_compute_pids=foreign))
        if foreign:
            raise RuntimeError('Another compute process is using the GPU')

    try:
        # This timer includes imports that may initialize CUDA, validation and profiling.
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.65, 0)
        source = Path(reg['attention_source'])
        if sha(source) != reg['attention_source_sha256']:
            raise RuntimeError('Upstream attention source changed after registration')
        spec = importlib.util.spec_from_file_location('qvg_attention_frozen', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not module.FLASH_ATTN_2_AVAILABLE or module.FLASH_ATTN_3_AVAILABLE:
            raise RuntimeError('The registered external FA2 dispatch is not available')
        telemetry('start')
        result['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
            flash_attn=version('flash-attn'), triton=version('triton'),
            gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
            sdpa_flags={k: getattr(torch.backends.cuda, k)() for k in
                ['flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled',
                 'cudnn_sdp_enabled']})
        explicit = dict(sdpa_flash=SDPBackend.FLASH_ATTENTION,
                        sdpa_efficient=SDPBackend.EFFICIENT_ATTENTION)
        variants = ['external_fa2', 'sdpa_flash', 'sdpa_efficient', 'sdpa_auto']
        candidates = variants[:3]
        phase = 'small_gate'

        def invoke(name, q, k, v):
            deadline()
            counts[phase + ':' + name] += 1
            if name == 'external_fa2':
                return module.attention(q, k, v)
            ctx = sdpa_kernel(explicit[name]) if name in explicit else nullcontext()
            with ctx:
                y = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    attn_mask=None, dropout_p=0.0, is_causal=False)
            return y.transpose(1, 2).contiguous()

        def check(actual, reference):
            a, b = actual.float(), reference.float()
            diff = (a - b).abs()
            tol = reg['atol'] + reg['rtol'] * b.abs()
            return dict(finite=bool(torch.isfinite(a).all()),
                        shape_matches=list(a.shape) == list(b.shape),
                        max_abs=float(diff.max()),
                        relative_l2=float(torch.linalg.vector_norm(a-b) /
                                          torch.linalg.vector_norm(b).clamp_min(1e-30)),
                        tolerance_violations=int((diff > tol).sum()),
                        passed=bool(torch.isfinite(a).all() and (diff <= tol).all()))

        generator = torch.Generator(device='cpu').manual_seed(reg['seed'])
        for label, qn, kn, scale in [('normal', 17, 37, 1), ('zero_query', 9, 23, 0),
                                      ('sharp', 31, 67, 4)]:
            # FP32 reference uses the same BF16-rounded operands, on CPU.
            q0 = (torch.randn(1, qn, 12, 128, generator=generator) * scale).bfloat16()
            k0 = (torch.randn(1, kn, 12, 128, generator=generator) * max(scale, 1)).bfloat16()
            v0 = torch.randn(1, kn, 12, 128, generator=generator).bfloat16()
            logits = q0.float().transpose(1, 2) @ k0.float().transpose(1, 2).transpose(-1, -2)
            ref = (torch.softmax(logits / 128**0.5, -1) @
                   v0.float().transpose(1, 2)).transpose(1, 2).contiguous().cuda()
            q, k, v = q0.cuda(), k0.cuda(), v0.cuda()
            for name in variants:
                out = invoke(name, q, k, v)
                info = check(out, ref)
                result['small_gates'].append(dict(case=label, backend=name, **info))
                if not info['passed']:
                    raise RuntimeError(f'Fixed numerical gate failed: {label}/{name}')
            del out, ref, q, k, v
        progress('small_gates_passed', count=len(result['small_gates']))

        events = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]
        for event in events:
            event.record()
        torch.cuda.synchronize()

        def timed(name, inputs):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            events[0].record()
            out = invoke(name, *inputs)
            events[1].record()
            events[1].synchronize()
            wall = (time.perf_counter() - tick) * 1000
            return out, dict(wall_ms=wall, cuda_interval_ms=events[0].elapsed_time(events[1]))

        # Instrumentation floor, not subtracted from any observation.
        floors = []
        for _ in range(20):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            events[0].record(); events[1].record(); events[1].synchronize()
            floors.append((time.perf_counter() - tick) * 1000)
        result['empty_event_sync_wall_ms'] = floors
        inputs_by_length = {}
        choices = {}

        for index, kn in enumerate(reg['k_lengths']):
            telemetry(f'before_shape_{kn}')
            rng = torch.Generator(device='cuda').manual_seed(reg['seed'] + kn)
            q = torch.randn(1, 4680, 12, 128, device='cuda', dtype=torch.bfloat16, generator=rng)
            k = torch.randn(1, kn, 12, 128, device='cuda', dtype=torch.bfloat16, generator=rng)
            v = torch.randn(1, kn, 12, 128, device='cuda', dtype=torch.bfloat16, generator=rng)
            inputs = (q, k, v)
            inputs_by_length[kn] = inputs
            record = dict(k_tokens=kn, q_tokens=4680, calls_in_original_schedule=reg['shape_calls'][str(kn)],
                inputs=[dict(shape=list(x.shape), stride=list(x.stride()), dtype=str(x.dtype)) for x in inputs],
                first={}, large_gates={}, selection={}, evaluation={}, profiles={})
            result['shapes'].append(record)
            phase = 'shape_first_call'
            outputs = {}
            order = variants[index % 4:] + variants[:index % 4]
            for name in order:
                outputs[name], record['first'][name] = timed(name, inputs)
            for name in variants:
                info = check(outputs[name], outputs['external_fa2'])
                record['large_gates'][name] = info
                if not info['passed']:
                    raise RuntimeError(f'Large-input numerical gate failed: {kn}/{name}')
            del outputs

            # This is the actual simple selector. Its clock includes warming all
            # three candidates, timing, synchronization and the choice itself.
            selection_tick = time.perf_counter()
            for name in candidates:
                record['selection'][name] = dict(warmup=[], samples=[])
            for stage, count in [('warmup', reg['warmup']), ('samples', reg['selection_samples'])]:
                phase = 'selector_' + stage
                for repeat in range(count):
                    offset = (index + repeat) % len(candidates)
                    for name in candidates[offset:] + candidates[:offset]:
                        out, measurement = timed(name, inputs)
                        record['selection'][name][stage].append(measurement)
                        del out
            chosen = min(candidates, key=lambda name: st.median(
                x['wall_ms'] for x in record['selection'][name]['samples']))
            record['selector_elapsed_ms'] = (time.perf_counter() - selection_tick) * 1000
            record['selector_first_probe_ms'] = sum(record['first'][name]['wall_ms'] for name in candidates)
            record['selected_before_evaluation'] = chosen
            choices[kn] = chosen
            progress('selection_frozen', k_tokens=kn, chosen=chosen)

            phase = 'auto_warmup'
            for _ in range(reg['warmup']):
                out, _ = timed('sdpa_auto', inputs)
                del out
            phase = 'evaluation'
            record['evaluation'] = {name: [] for name in variants}
            for repeat in range(reg['evaluation_samples']):
                offset = (index + repeat) % len(variants)
                for name in variants[offset:] + variants[:offset]:
                    out, measurement = timed(name, inputs)
                    record['evaluation'][name].append(measurement)
                    del out
            progress('shape_completed', k_tokens=kn,
                     medians_ms={name: st.median(x['wall_ms'] for x in record['evaluation'][name])
                                 for name in variants})

        # All decisions for the deployable selector were persisted before these
        # evaluation observations. The posthoc path is only a comparison bound.
        posthoc = {r['k_tokens']: min(variants, key=lambda name:
            st.median(x['wall_ms'] for x in r['evaluation'][name])) for r in result['shapes']}
        policies = {
            'fixed_external_fa2': {kn: 'external_fa2' for kn in inputs_by_length},
            'sdpa_auto': {kn: 'sdpa_auto' for kn in inputs_by_length},
            'measured_selector': choices,
            'posthoc_reference': posthoc,
        }
        result['policies'] = {p: {str(k): v for k, v in mapping.items()} for p, mapping in policies.items()}
        result['selector_overhead_ms'] = sum(
            r['selector_first_probe_ms'] + r['selector_elapsed_ms'] for r in result['shapes'])
        result['fixed_first_shape_premium_ms'] = {}
        for policy in ['fixed_external_fa2', 'sdpa_auto', 'posthoc_reference']:
            result['fixed_first_shape_premium_ms'][policy] = sum(max(0.0,
                r['first'][policies[policy][r['k_tokens']]]['wall_ms'] -
                st.median(x['wall_ms'] for x in r['evaluation'][policies[policy][r['k_tokens']]]))
                for r in result['shapes'])
        progress('replay_begin', policies=list(policies))

        for repeat in range(reg['replay_rounds']):
            telemetry(f'before_replay_{repeat}')
            names = list(policies)
            order = names[repeat % len(names):] + names[:repeat % len(names)]
            row = dict(round=repeat, order=order, policies={})
            for policy in order:
                phase = 'replay_' + policy
                torch.cuda.synchronize()
                tick = time.perf_counter()
                events[0].record()
                calls = 0
                for kn, inputs in inputs_by_length.items():
                    for _ in range(reg['shape_calls'][str(kn)]):
                        out = invoke(policies[policy][kn], *inputs)
                        calls += 1
                        del out
                events[1].record(); events[1].synchronize()
                elapsed = (time.perf_counter() - tick) * 1000
                overhead = (result['selector_overhead_ms'] if policy == 'measured_selector'
                            else result['fixed_first_shape_premium_ms'][policy])
                row['policies'][policy] = dict(calls=calls, replay_wall_ms=elapsed,
                    replay_cuda_interval_ms=events[0].elapsed_time(events[1]),
                    one_time_cost_ms=overhead, cost_inclusive_ms=elapsed + overhead)
            result['replay_rounds'].append(row)
            progress('replay_round_completed', round=repeat,
                     cost_inclusive_ms={k: v['cost_inclusive_ms'] for k, v in row['policies'].items()})

        # Kernel attribution is deliberately after timing, never used to select.
        expected = {'sdpa_flash': 'aten::_scaled_dot_product_flash_attention',
                    'sdpa_efficient': 'aten::_scaled_dot_product_efficient_attention'}
        for r in result['shapes']:
            kn = r['k_tokens']
            for name in variants:
                phase = 'kernel_attribution'
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    out = invoke(name, *inputs_by_length[kn])
                    torch.cuda.synchronize()
                operators = sorted({e.key for e in prof.key_averages()
                                    if 'attention' in e.key or 'flash' in e.key})
                kernels = sorted({e.name for e in prof.events() if 'CUDA' in str(e.device_type)})
                attribution = dict(operators=operators, cuda_kernels=kernels)
                r['profiles'][name] = attribution
                if name in expected and expected[name] not in operators:
                    raise RuntimeError(f'Forced backend attribution mismatch: {kn}/{name}')
                if not kernels:
                    raise RuntimeError(f'No CUDA kernel evidence: {kn}/{name}')
                del out
            progress('kernel_attribution_completed', k_tokens=kn,
                     auto_operators=r['profiles']['sdpa_auto']['operators'])

        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        result['upstream_source_unchanged'] = sha(source) == reg['attention_source_sha256']
        assert result['upstream_source_unchanged']
        telemetry('completed')
        result['status'] = 'completed'
    except BaseException as exc:
        result['status'] = 'invalid_or_incomplete'
        result['errors'].append(dict(type=type(exc).__name__, message=str(exc),
                                     traceback=traceback.format_exc()))
        print(traceback.format_exc(), flush=True)
    finally:
        result['finished_at'] = stamp()
        result['elapsed_seconds'] = time.monotonic() - start
        result['call_counts'] = dict(counts)
        dump(result_path, result)
        progress('finished', status=result['status'], elapsed_seconds=result['elapsed_seconds'])
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
