"""One initial chunk, then at most three captured linear shapes. No model replay."""
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'src/experiments/linear_w8a8_stage1'
DATA = Path('/home/acorn0415/sfq_data/linear_w8a8_stage1')
START = datetime.fromisoformat('2026-11-04T13:10:04+00:00').timestamp()
DEADLINE = START + 3600
STATE = {'status': 'preparing', 'model_forward_started': 0, 'model_forward_finished': 0,
         'start_utc': START, 'deadline_utc': DEADLINE, 'benchmarks': []}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def check():
    if time.time() >= DEADLINE:
        raise TimeoutError('registered 60-minute deadline reached')


def emit(message):
    STATE['phase'] = message
    STATE['updated_utc'] = time.time()
    save(OUT / 'results.json', STATE)
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)
    check()


def gpu_check():
    check()
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], text=True)
    others = [int(x) for x in raw.split() if int(x) != os.getpid()]
    if others:
        raise RuntimeError(f'Other compute processes: {others}')


def integer_gate(torch):
    gen = torch.Generator().manual_seed(303)
    cases = []
    for kind in ('random', 'zero', 'extreme'):
        a = torch.randint(-127, 128, (32, 32), dtype=torch.int8, generator=gen)
        b = torch.randint(-127, 128, (32, 16), dtype=torch.int8, generator=gen)
        if kind == 'zero':
            a.zero_()
        if kind == 'extreme':
            a.fill_(127); b.fill_(-127)
        result = torch._int_mm(a.cuda(), b.cuda()).cpu()
        ref = a.long() @ b.long()
        ok = result.dtype == torch.int32 and torch.equal(result.long(), ref)
        cases.append({'case': kind, 'all_elements_equal': ok})
        assert ok, kind
    STATE['integer_gate'] = cases


def capture_model(torch, native):
    emit('model_and_text_preparation')
    t = time.perf_counter()
    prompt = (ROOT / 'src/experiments/policy_study_prompts.txt').read_text().splitlines()[0]
    # Reuse frozen loading code, but redirect ALL output/work paths to this experiment.
    native.DATA = DATA
    native.OUT = OUT
    native.setup_paths()
    pipe, cond = native.build({'prompt': prompt})
    torch.cuda.synchronize()
    STATE['model_and_text_preparation_s'] = time.perf_counter() - t
    STATE['prompt'] = prompt
    STATE['preparation_scope'] = 'text encoding + model loading jointly; not separately instrumented'
    gpu_check()
    events, captures, stack, handles = [], {}, {}, []
    pos = {'forward': -1}

    def before_forward(module, args, kwargs):
        check()
        pos['forward'] += 1
        assert pos['forward'] < 5
        STATE['model_forward_started'] += 1
        emit(f'model_forward_{pos["forward"]}_started')
        stack['forward'] = torch.cuda.Event(enable_timing=True)
        stack['forward'].record()

    def after_forward(module, args, output):
        end = torch.cuda.Event(enable_timing=True); end.record()
        events.append({'kind': 'forward', 'forward': pos['forward'],
                       'begin': stack.pop('forward'), 'end': end})
        STATE['model_forward_finished'] += 1

    handles.append(pipe.generator.register_forward_pre_hook(before_forward, with_kwargs=True))
    handles.append(pipe.generator.register_forward_hook(after_forward))
    for name, mod in pipe.generator.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue

        def pre(module, args, name=name):
            x = args[0]
            shape = (x.numel() // x.shape[-1], module.out_features, module.in_features)
            key = 'x'.join(map(str, shape))
            ctx = torch.profiler.record_function('linear_leaf/' + name)
            ctx.__enter__()
            begin = torch.cuda.Event(enable_timing=True); begin.record()
            stack[name] = (begin, ctx, key, shape)

        def post(module, args, y, name=name):
            end = torch.cuda.Event(enable_timing=True); end.record()
            begin, ctx, key, shape = stack.pop(name)
            ctx.__exit__(None, None, None)
            events.append({'kind': 'linear', 'name': name, 'forward': pos['forward'],
                           'shape_MNK': list(shape), 'key': key, 'begin': begin, 'end': end})
            if pos['forward'] == 1 and shape[0] == 4680 and key not in captures:
                x = args[0]
                if x.dtype != torch.bfloat16 or module.weight.dtype != torch.bfloat16:
                    return
                with torch.profiler.record_function('diagnostic_capture/' + key):
                    captures[key] = {'x': x.detach().clone(), 'w': module.weight.detach().clone(),
                                     'bias': None if module.bias is None else module.bias.detach().clone(),
                                     'y': y.detach().clone(), 'name': name, 'shape_MNK': list(shape)}
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))
    torch.manual_seed(303)
    noise = torch.randn((1, 3, 16, 60, 104), device='cuda', dtype=torch.bfloat16)
    pipe.kv_cache1 = None; pipe.crossattn_cache = None
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, with_stack=False) as profiler:
            _, latent = pipe.inference(noise=noise, text_prompts=[prompt], return_latents=True,
                                       profile=False, low_memory=False)
        torch.cuda.synchronize()
        STATE['instrumented_chunk_wall_s'] = time.perf_counter() - start
        STATE['instrumented_model_peak_allocated'] = torch.cuda.max_memory_allocated()
        STATE['instrumented_model_peak_reserved'] = torch.cuda.max_memory_reserved()
        assert STATE['model_forward_finished'] == STATE['model_forward_started'] == 5
        profiler.export_chrome_trace(str(DATA / 'trace.json'))
        rows = []
        for e in profiler.key_averages(group_by_input_shape=True):
            rows.append({'key': e.key, 'input_shapes': e.input_shapes, 'count': e.count,
                         'self_device_time_us': e.self_device_time_total,
                         'device_time_us_inclusive': e.device_time_total,
                         'self_cpu_time_us': e.self_cpu_time_total})
        save(OUT / 'operators.json', rows)
        times = []
        totals = {}
        for row in events:
            row['gpu_ms'] = row.pop('begin').elapsed_time(row.pop('end'))
            times.append(row)
            if row['kind'] == 'linear' and row['forward'] >= 1 and row['key'] in captures:
                totals[row['key']] = totals.get(row['key'], 0.) + row['gpu_ms']
        selected = sorted(totals, key=lambda k: (-totals[k], k))[:3]
        STATE['shape_ranking_ms'] = totals
        STATE['selected_shapes'] = selected
        save(OUT / 'linear_events.json', times)
        STATE['captures'] = []
        for key in selected:
            sample = {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                      for k, v in captures[key].items()}
            file = DATA / (key + '.pt')
            torch.save(sample, file)
            STATE['captures'].append({'path': str(file), 'sha256': sha(file), 'bytes': file.stat().st_size,
                                      'module': sample['name'], 'shape_MNK': sample['shape_MNK']})
        assert sum(x['bytes'] for x in STATE['captures']) < 2**30
        emit('one_chunk_profile_and_capture_complete')
    finally:
        for handle in handles:
            handle.remove()
    del pipe, cond, noise, latent, captures, events, profiler, mod
    gc.collect(); torch.cuda.empty_cache()


def quantize(torch, value):
    fp = value.float()
    scale = fp.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.
    return torch.round(fp / scale).clamp(-127, 127).to(torch.int8), scale


def metrics(torch, y, ref):
    yf, rf = y.float(), ref.float()
    delta = yf - rf
    return {'rel_l2': (delta.norm() / rf.norm().clamp_min(1e-30)).item(),
            'max_abs': delta.abs().max().item(), 'finite': bool(torch.isfinite(yf).all())}


def bench_shape(torch, item):
    import numpy as np
    gpu_check()
    sample = torch.load(item['path'], map_location='cpu', weights_only=True)
    x, w, y = (sample[k].cuda() for k in ('x', 'w', 'y'))
    bias = None if sample['bias'] is None else sample['bias'].cuda()
    m, n, k = sample['shape_MNK']; x2 = x.reshape(m, k)
    result = {'shape_MNK': [m, n, k], 'module': sample['name']}
    baseline = lambda: torch.nn.functional.linear(x, w, bias)
    replay = baseline()
    result['native_replay_bytes_equal'] = torch.equal(replay.contiguous().view(torch.uint8),
                                                    y.contiguous().view(torch.uint8))
    assert result['native_replay_bytes_equal'], 'native replay validity failure'
    torch.cuda.synchronize(); start = time.perf_counter()
    qw, sw = quantize(torch, w); wt = qw.t().contiguous()
    torch.cuda.synchronize(); result['static_weight_prepare_s'] = time.perf_counter() - start

    def candidate():
        qx, sx = quantize(torch, x2)
        acc = torch._int_mm(qx, wt)
        out = acc.float() * sx * sw.t()
        if bias is not None:
            out = out + bias.float()
        return out.to(torch.bfloat16).reshape(*x.shape[:-1], n)

    qx, sx = quantize(torch, x2)
    acc = torch._int_mm(qx, wt)
    rows = sorted(set(list(range(min(4, m))) + list(range(max(0, m - 4), m))))
    exact = qx[rows].cpu().long() @ qw[:16].cpu().long().t()
    sampled = torch.equal(acc[rows, :16].cpu().long(), exact)
    int_range = bool((qx.int().abs() <= 127).all() and (qw.int().abs() <= 127).all())
    assert sampled and int_range and k * 127**2 < 2**31
    # Independent FP32 GEMM of the same integer operands, with TF32 disabled.
    # Do not assume arbitrary INT32 sums are exactly FP32-representable.
    float_acc = qx.float() @ wt.float()
    result['fp32_integer_reference_max_abs'] = (float_acc - acc.float()).abs().max().item()
    ref = float_acc * sx * sw.t()
    if bias is not None:
        ref = ref + bias.float()
    output = candidate()
    qmetrics = metrics(torch, output.reshape(m, n), ref)
    pointwise = bool(((output.reshape(m, n).float() - ref).abs() <= .02 + .02 * ref.abs()).all())
    qvalid = qmetrics['finite'] and qmetrics['rel_l2'] <= .002 and pointwise
    quality = metrics(torch, output, y)
    result.update(integer_sample_equal=sampled, integer_sample_shape=[len(rows), min(16, n)],
                  integer_range_valid=int_range, accumulator_worst_bound=k * 127**2,
                  quantized_reference=qmetrics, quantized_pointwise_valid=pointwise,
                  candidate_matches_quantized_reference=qvalid, vs_native_bf16=quality,
                  engineering_error_gate=quality['finite'] and quality['rel_l2'] <= .05,
                  output_shape_valid=output.shape == y.shape,
                  output_dtype_valid=output.dtype == torch.bfloat16,
                  weight_storage_bf16=w.numel() * 2,
                  weight_storage_w8a8=wt.numel() + sw.numel() * 4)
    assert qvalid and result['output_shape_valid'] and result['output_dtype_valid']
    del replay, output, ref, float_acc, qx, sx, acc, exact, qw
    gc.collect(); torch.cuda.empty_cache()
    for fn in (baseline, candidate):
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    timing = []
    for round_id in range(10):
        check()
        pair = {'round': round_id, 'order': ['BF16', 'W8A8'] if round_id % 2 == 0 else ['W8A8', 'BF16']}
        for arm in pair['order']:
            fn = baseline if arm == 'BF16' else candidate
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter(); begin.record()
            for _ in range(10):
                value = fn()
                del value
            end.record(); torch.cuda.synchronize()
            pair[arm] = {'wall_ms_per_call': (time.perf_counter() - start) * 1000 / 10,
                         'gpu_ms_per_call': begin.elapsed_time(end) / 10,
                         'common_resident_bytes': resident,
                         'incremental_peak_bytes': torch.cuda.max_memory_allocated() - resident}
        timing.append(pair)
    reductions = np.array([100 * (1 - p['W8A8']['wall_ms_per_call'] / p['BF16']['wall_ms_per_call']) for p in timing])
    rng = np.random.default_rng(20261104)
    boot = np.median(rng.choice(reductions, size=(10000, 10), replace=True), axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    result.update(timing=timing, paired_wall_reduction_percent=reductions.tolist(),
                  median_reduction_percent=float(np.median(reductions)),
                  bootstrap_median_95ci_percent=[float(lo), float(hi)])
    result['promising'] = bool(result['engineering_error_gate'] and np.median(reductions) >= 5 and lo > 0)
    return result


def main():
    check()
    if (OUT / 'results.json').exists() or (DATA / 'execution_started.json').exists():
        raise RuntimeError('Refusing to overwrite or repeat an already started experiment')
    DATA.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('deadline alarm')))
    signal.alarm(max(1, int(DEADLINE - time.time())))
    save(DATA / 'execution_started.json', {'utc': time.time(), 'pid': os.getpid()})
    import kv_reuse_stage1 as native
    native.DATA = DATA; native.OUT = OUT; native.setup_paths()
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    STATE.update(torch=torch.__version__, cuda=torch.version.cuda, pid=os.getpid(),
                 git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 code_sha256=sha(Path(__file__)), loader_sha256=sha(ROOT / 'tools/kv_reuse_stage1.py'),
                 registration_sha256=sha(OUT / 'REGISTRATION_KO.txt'))
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=native.SOURCE, text=True).strip() == native.REV
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=native.SOURCE, text=True).strip()
    gpu_check()
    STATE['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                          '--format=csv,noheader'], text=True).strip()
    try:
        with torch.inference_mode():
            emit('integer_gate'); integer_gate(torch)
            capture_model(torch, native)
            for item in STATE['captures']:
                emit('benchmark_' + 'x'.join(map(str, item['shape_MNK'])))
                STATE['benchmarks'].append(bench_shape(torch, item))
                gc.collect(); torch.cuda.empty_cache()
                emit('benchmark_complete')
        STATE['status'] = 'completed'
    except BaseException as exc:
        STATE['status'] = 'failed'
        STATE['error'] = repr(exc); STATE['traceback'] = traceback.format_exc()
        raise
    finally:
        STATE['finished_utc'] = time.time()
        STATE['elapsed_from_registered_start_s'] = time.time() - START
        save(OUT / 'results.json', STATE)
        print('FINAL_STATUS', STATE['status'], flush=True)


if __name__ == '__main__':
    main()
