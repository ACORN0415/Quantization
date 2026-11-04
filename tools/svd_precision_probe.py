"""CPU-only, fixed TT rank: numerical SVD precision versus packed storage loss."""
import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
import torch
import codec_order_core as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'src/experiments/svd_precision_stage1'
DATA = Path('/home/acorn0415/sfq_data/svd_precision_stage1')
DEADLINE = 1793830735.0  # 2026-11-04 22:18:55 UTC
torch.set_num_threads(4)
torch.set_num_interop_threads(1)


def check_time():
    if time.time() >= DEADLINE:
        raise TimeoutError('C 30-minute limit')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f'Refuse overwrite {path}')
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def record(path):
    return {'path': str(path), 'sha256': sha(path), 'file_bytes': Path(path).stat().st_size}


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def provenance():
    return {p: sha(ROOT / p) for p in (
        'tools/svd_precision_probe.py', 'tools/codec_order_core.py',
        'tools/change_mixed_core.py', 'src/experiments/svd_precision_stage1/REGISTRATION_KO.txt')}


def decode_factor_numpy(state):
    n = math.prod(state['shape']); bits = state['bits']; qmax = (1 << (bits - 1)) - 1
    stream = state['stream'].numpy()
    shifts = (np.arange(n, dtype=np.int64) * bits) % 8
    code = ((stream[(np.arange(n) * bits) // 8] >> shifts) & ((1 << bits) - 1)).astype(np.float32)
    assert bits in (2, 4)  # registered widths divide bytes; no cross-byte fields
    maxima = np.repeat(state['maximum'].numpy().astype(np.float32), 16)[:n]
    return (((code - np.float32(qmax)) * maxima) / np.float32(qmax)).reshape(state['shape'])


def canonical(raw):
    a, b, c = [v.clone() for v in raw['factors']]
    idx = a.abs().argmax(dim=1, keepdim=True)
    s1 = a.gather(1, idx).sign().squeeze(1); s1[s1 == 0] = 1
    a = a * s1[:, None, :]; b = b * s1[:, :, None, None]
    flat = b.reshape(b.shape[0], -1, b.shape[-1])
    idx = flat.abs().argmax(dim=1, keepdim=True)
    s2 = flat.gather(1, idx).sign().squeeze(1); s2[s2 == 0] = 1
    b = b * s2[:, None, None, :]; c = c * s2[:, :, None]
    return {**raw, 'factors': [a, b, c]}, {'first_flips': int((s1 < 0).sum()), 'second_flips': int((s2 < 0).sum())}


def errors(y, ref):
    y = y.double(); ref = ref.double(); sse = float((y - ref).square().sum())
    norm = float(ref.square().sum())
    return {'sse': sse, 'mse': sse / ref.numel(), 'normalized_squared_error': sse / norm,
            'relative_l2': math.sqrt(sse / norm)}


def attention_setup(p):
    q = p['query'][0].double().permute(1, 0, 2)
    k = p['key'][0].double().permute(1, 2, 0)
    probs = (q @ k / math.sqrt(128)).softmax(-1)
    assert probs.shape == (12, 32, 32760)
    return probs[:, :, -4680:], p['wo'].double()


def output(v, probs, wo):
    z = probs @ v[0].double().permute(1, 0, 2)
    return z.permute(1, 0, 2).reshape(32, 1536) @ wo.T


def unit():
    check_time(); gen = torch.Generator().manual_seed(771)
    x = torch.randn((2, 3, 16, 8), generator=gen, dtype=torch.float64)
    raw = core.decompose(x, 'tt', 16)
    ref = x.permute(1, 2, 0, 3).reshape(1, 48, 2, 8)
    err = errors(core.reconstruct('tt', raw['shape'], raw['factors']), ref)
    assert err['relative_l2'] <= 1e-12
    counts = 0
    for dtype in (torch.float32, torch.float64):
        rr = {**raw, 'factors': [a.to(dtype) for a in raw['factors']]}
        can, _ = canonical(rr)
        for bits in (2, 4):
            state = core.store(rr, bits); cs = core.store(can, bits); counts += 2
            assert core.payload_bytes(state) == core.expected_bytes('tt', raw['shape'], 16, bits)
            for f in state['factors']:
                assert same(torch.from_numpy(decode_factor_numpy(f)), core.unpack_factor(f))
            assert same(core.decode(state), core.decode(cs))
    save(OUT / 'unit.json', {'status': 'passed', 'gpu_calls': 0, 'synthetic_svd_calls': 2,
         'synthetic_store_calls': counts, 'full_rank_relative_l2': err['relative_l2'], 'provenance': provenance()})


def worker(prompt, precision):
    check_time(); assert not torch.cuda.is_initialized()
    assert json.loads((OUT / 'unit.json').read_text())['provenance'] == provenance()
    folder = DATA / f'p{prompt:02d}_fp{precision}'; folder.mkdir(exist_ok=False)
    path = Path(f'/home/acorn0415/sfq_data/postwo_rounding/p{prompt:02d}_s101/l29_c6.pt')
    expected = json.loads((ROOT / 'src/experiments/mixed_precision_world_review_20261104/input_availability.json').read_text())
    expected = next(a['sha256'] for a in expected['files'] if a['path'] == str(path))
    assert sha(path) == expected
    p = torch.load(path, map_location='cpu', weights_only=True)
    v = p['new_value']; assert v.shape == (1, 4680, 12, 128) and v.dtype == torch.bfloat16
    assert p['chunk'] == 6 and p['call'] == 4 and torch.isfinite(v).all()
    dtype = torch.float32 if precision == 32 else torch.float64
    x = v.to(dtype)[0].reshape(3, 1560, 12, 128).permute(2, 0, 1, 3).contiguous()
    assert same(x.to(torch.bfloat16), v[0].reshape(3, 1560, 12, 128).permute(2, 0, 1, 3))
    timings = []; svd_calls = []; first = None; first_spectra = []
    original = torch.linalg.svd
    def observed(a, *args, **kwargs):
        check_time(); assert a.dtype == dtype and a.device.type == 'cpu'
        start = time.perf_counter(); result = original(a, *args, **kwargs); elapsed = time.perf_counter() - start
        svd_calls.append({'shape': list(a.shape), 'dtype': str(a.dtype), 'seconds': elapsed})
        if len(svd_calls) <= 2:
            first_spectra.append(result.S.clone())
        return result
    torch.linalg.svd = observed
    try:
        for repeat in range(3):
            check_time(); started = time.perf_counter(); raw = core.decompose(x, 'tt', 192)
            timings.append(time.perf_counter() - started)
            if first is None:
                first = raw
            else:
                assert all(same(a, b) for a, b in zip(first['factors'], raw['factors'])), 'SVD repeat differs'
            print(prompt, precision, 'decomposition', repeat + 1, timings[-1], flush=True)
    finally:
        torch.linalg.svd = original
    assert len(svd_calls) == 6
    raw = first; can, flips = canonical(raw)
    raw64 = core.reconstruct('tt', raw['shape'], [a.double() for a in raw['factors']])
    probs, wo = attention_setup(p); ref_o = output(v, probs, wo)
    raw_o = output(raw64, probs, wo)
    rr = {'prompt': prompt, 'precision': precision, 'status': 'completed', 'input': record(path),
          'provenance': provenance(), 'rank': 192, 'shape': list(x.shape), 'gpu_calls': 0,
          'decompose_seconds': timings, 'svd_calls': svd_calls, 'repeat_factor_byte_equal': True,
          'sign_flips': flips, 'raw_v_error': errors(raw64, v),
          'raw_output_error': errors(raw_o, ref_o), 'reference_chunk_output_norm_sq': float(ref_o.square().sum()),
          'rank_boundary_relative_gap_per_head': ((first_spectra[0][:, 191] - first_spectra[0][:, 192]) / first_spectra[0][:, 191]).tolist(),
          'arms': []}
    rawfile = folder / 'raw.pt'; torch.save({'raw': raw, 'spectra': first_spectra}, rawfile); rr['raw'] = record(rawfile)
    for bits in (2, 4):
        check_time(); start = time.perf_counter(); state = core.store(raw, bits); store_s = time.perf_counter() - start
        start = time.perf_counter(); decoded = core.decode(state); restored = decoded.to(torch.bfloat16); decode_s = time.perf_counter() - start
        assert torch.isfinite(restored).all()
        assert core.payload_bytes(state) == core.expected_bytes('tt', raw['shape'], 192, bits)
        for f in state['factors']:
            assert same(torch.from_numpy(decode_factor_numpy(f)), core.unpack_factor(f))
        assert same(core.decode(core.store(can, bits)), decoded), 'Canonical signs changed decoded result'
        final_o = output(restored, probs, wo)
        arm = {'bits': bits, 'payload_tensor_bytes': core.payload_bytes(state), 'store_seconds': store_s,
               'decode_cast_seconds': decode_s, 'sign_invariant_byte_equal': True,
               'decoded_fp32_v_error': errors(decoded, v), 'restored_bf16_v_error': errors(restored, v),
               'output_error': errors(final_o, ref_o),
               'storage_change_from_raw': errors(restored, raw64)}
        dest = folder / f'b{bits}.pt'
        torch.save({'state': state, 'restored': restored, 'output': final_o, 'reference_output': ref_o}, dest)
        loaded = torch.load(dest, map_location='cpu', weights_only=True)
        assert same(core.decode(loaded['state']).to(torch.bfloat16), restored)
        arm['artifact'] = record(dest); rr['arms'].append(arm)
    rr.update(finished_utc=time.time(), max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    check_time(); assert not torch.cuda.is_initialized(); assert sha(path) == expected
    save(OUT / f'p{prompt:02d}_fp{precision}.json', rr)


def decision(rows):
    groups = {(r['prompt'], r['precision']): r for r in rows}; comparisons = []
    for prompt in (0, 4):
        for bits in (2, 4):
            a, b = [next(x for x in groups[prompt, precision]['arms'] if x['bits'] == bits) for precision in (32, 64)]
            assert a['payload_tensor_bytes'] == b['payload_tensor_bytes']
            comparisons.append({'prompt': prompt, 'bits': bits, 'payload_tensor_bytes': a['payload_tensor_bytes'],
                'v_sse_reduction': 1 - b['restored_bf16_v_error']['sse'] / a['restored_bf16_v_error']['sse'],
                'output_mse_reduction': 1 - b['output_error']['mse'] / a['output_error']['mse']})
    successful = [bits for bits in (2, 4) if all(c['v_sse_reduction'] >= .01 and c['output_mse_reduction'] >= .01 for c in comparisons if c['bits'] == bits)]
    negligible = all(abs(c[k]) < .01 for c in comparisons for k in ('v_sse_reduction', 'output_mse_reduction'))
    return {'comparisons': comparisons, 'successful_bits': successful, 'all_changes_below_one_percent': negligible,
            'decision': 'precision_effect_followup_justified' if successful else ('close_precision_upgrading_for_tested_cases' if negligible else 'mixed_effect_not_adopted')}


def summarize():
    check_time(); rows = [json.loads((OUT / f'p{p:02d}_fp{t}.json').read_text()) for p in (0, 4) for t in (32, 64)]
    distances = []
    for p in (0, 4):
        aa, bb = [torch.load(DATA / f'p{p:02d}_fp{t}/raw.pt', map_location='cpu', weights_only=True)['raw'] for t in (32, 64)]
        a, b = [core.reconstruct('tt', z['shape'], [f.double() for f in z['factors']]) for z in (aa, bb)]
        distances.append({'prompt': p, 'fp32_vs_fp64_raw_reconstruction': errors(a, b)})
    save(OUT / 'results.json', {'status': 'completed', 'provenance': provenance(), 'rows': rows,
         'raw_distances': distances, **decision(rows), 'finished_utc': time.time(), 'real_svd_calls': 24,
         'real_primary_store_calls': 8, 'real_sign_control_store_calls': 8, 'gpu_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['unit', 'worker', 'summarize'])
    parser.add_argument('--prompt', type=int, choices=[0, 4]); parser.add_argument('--precision', type=int, choices=[32, 64])
    args = parser.parse_args()
    if args.mode == 'unit': unit()
    elif args.mode == 'worker': worker(args.prompt, args.precision)
    else: summarize()
