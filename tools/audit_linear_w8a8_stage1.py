"""Read-only, CPU-only audit. Prints JSON; never overwrites the measured records."""
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'src/experiments/linear_w8a8_stage1'
DATA = Path('/home/acorn0415/sfq_data/linear_w8a8_stage1')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    r = json.loads((OUT / 'results.json').read_text())
    assert r['status'] == 'completed'
    assert r['model_forward_started'] == r['model_forward_finished'] == 5
    assert r['finished_utc'] < r['deadline_utc']
    for p, field in [(ROOT / 'tools/linear_w8a8_stage1.py', 'code_sha256'),
                     (ROOT / 'tools/kv_reuse_stage1.py', 'loader_sha256'),
                     (OUT / 'REGISTRATION_KO.txt', 'registration_sha256')]:
        assert sha(p) == r[field], field
    events = json.loads((OUT / 'linear_events.json').read_text())
    assert sorted(x['forward'] for x in events if x['kind'] == 'forward') == list(range(5))
    totals = {}
    for x in events:
        if x['kind'] == 'linear' and x['forward'] >= 1 and x['key'] in r['shape_ranking_ms']:
            totals[x['key']] = totals.get(x['key'], 0.) + x['gpu_ms']
    assert totals == r['shape_ranking_ms']
    assert sorted(totals, key=lambda k: (-totals[k], k))[:3] == r['selected_shapes']
    checks = []
    for cap, b in zip(r['captures'], r['benchmarks']):
        assert sha(Path(cap['path'])) == cap['sha256']
        sample = torch.load(cap['path'], map_location='cpu', weights_only=True)
        x, w, y = (sample[k] for k in ('x', 'w', 'y'))
        m, n, k = b['shape_MNK']
        assert x.numel() == m*k and w.shape == (n, k) and y.numel() == m*n
        assert x.dtype == w.dtype == y.dtype == torch.bfloat16
        assert torch.isfinite(x).all() and torch.isfinite(w).all() and torch.isfinite(y).all()
        # Reconstruct all outputs on CPU with FP64 integer matrix multiplication.
        # Every scalar product and integer sum is exactly representable in FP64.
        def q(value):
            f = value.float()
            s = f.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.
            return torch.round(f/s).clamp(-127, 127), s
        qx, sx = q(x.reshape(m, k)); qw, sw = q(w)
        acc = qx.double() @ qw.double().t()
        fp = acc.float() * sx * sw.t()
        if sample['bias'] is not None:
            fp = fp + sample['bias'].float()
        recovered = fp.bfloat16().float()
        # The CPU FP32 norm of tens of millions of entries loses measurable
        # accumulation precision. Audit scalar metrics with FP64 reductions.
        rel = ((recovered - y.reshape(m, n).float()).double().norm() /
               y.double().norm().clamp_min(1e-30)).item()
        # Audit agreement of scalar summaries, not a claim of GPU-output bit equality.
        print(json.dumps({'shape': [m,n,k], 'cpu_reconstruction_rel_l2': rel,
                          'saved_gpu_rel_l2': b['vs_native_bf16']['rel_l2'],
                          'difference': rel-b['vs_native_bf16']['rel_l2']}), file=__import__('sys').stderr)
        assert abs(rel - b['vs_native_bf16']['rel_l2']) < 1e-6
        reductions = np.array([100*(1-p['W8A8']['wall_ms_per_call']/p['BF16']['wall_ms_per_call'])
                               for p in b['timing']])
        assert np.array_equal(reductions, b['paired_wall_reduction_percent'])
        rng = np.random.default_rng(20261104)
        ci = np.quantile(np.median(rng.choice(reductions, (10000, 10)), axis=1), [.025, .975])
        assert np.array_equal(ci, b['bootstrap_median_95ci_percent'])
        assert b['median_reduction_percent'] == float(np.median(reductions))
        assert all(t['order'] == (['BF16', 'W8A8'] if i%2 == 0 else ['W8A8', 'BF16'])
                   for i, t in enumerate(b['timing']))
        assert len(b['timing']) == 10
        assert not b['promising']
        checks.append({'shape_MNK': [m,n,k], 'capture_hash_valid': True,
                       'cpu_full_reconstruction_relative_l2': rel,
                       'saved_gpu_relative_l2': b['vs_native_bf16']['rel_l2'],
                       'bootstrap_and_decision_recomputed': True})
    trace = json.loads((DATA / 'trace.json').read_text())
    kernels = [x for x in trace['traceEvents'] if x.get('cat') == 'kernel']
    def family(name):
        if 'gemm' in name.lower(): return 'GEMM'
        if 'flash_fwd' in name: return 'FlashAttention'
        return 'Other'
    families = {}
    for row in kernels:
        f = family(row['name'])
        families.setdefault(f, {'calls': 0, 'duration_ms': 0.})
        families[f]['calls'] += 1
        families[f]['duration_ms'] += row['dur']/1000
    print(json.dumps({'status': 'passed', 'gpu_calls': 0, 'model_forward_ledger_valid': True,
                      'frozen_hashes_valid': True, 'capture_and_selection_checks': checks,
                      'trace_sha256': sha(DATA / 'trace.json'),
                      'kernel_categories': families,
                      'kernel_duration_note': 'Only cat=kernel; sum of durations, not critical-path time. Name-based families.',
                      'scope_limit': 'No saved GPU output array to byte-compare. CPU reconstructed scalar errors agree; GPU native replay claim comes from execution records.'},
                     indent=2))


if __name__ == '__main__':
    main()
