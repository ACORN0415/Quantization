"""Read-only CPU audit of A-2, including its rejected BF16 baseline.

Prints a new audit to stdout; never rewrites the frozen measurement records.
No GPU initialization or current-time deadline check.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'src/experiments/linear_w8a8_fused_stage1'
ARMS = ['BF16_eager', 'BF16_tuned', 'W8A8_default', 'W8A8_fused']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def metrics(y, ref):
    d = y.float() - ref.float()
    return {'rel_l2': float(d.double().norm() / ref.double().norm()),
            'max_abs': float(d.abs().max()),
            'pointwise_valid': bool((d.abs() <= .02 + .02 * ref.float().abs()).all()),
            'finite': bool(torch.isfinite(y).all())}


def trace_arms(path):
    events = json.loads(Path(path).read_text())['traceEvents']
    labels = [e for e in events if e.get('cat') == 'user_annotation' and e['name'] in ARMS]
    assert len(labels) == 4
    launches = {e['args']['correlation']: e for e in events
                if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'correlation' in e.get('args', {})}
    result = {a: [] for a in ARMS}
    for e in events:
        if e.get('cat') != 'kernel':
            continue
        launch = launches[e['args']['correlation']]
        parent = [p for p in labels if p['ts'] <= launch['ts'] < p['ts'] + p['dur']]
        assert len(parent) == 1
        result[parent[0]['name']].append(e['name'])
    assert len(result['W8A8_fused']) == 3
    assert any(n.startswith('triton_tem_') for n in result['W8A8_fused'])
    assert not any('ampere_igemm' in n for n in result['W8A8_fused'])
    assert any('ampere_igemm' in n for n in result['W8A8_default'])
    return result


def main():
    torch.set_num_threads(4)
    r = json.loads((OUT / 'results.json').read_text())
    assert r['status'] == 'partial_or_invalid' and r['model_forwards'] == 0
    assert r['finished_utc'] < r['deadline_utc']
    assert sha(ROOT / 'tools/linear_w8a8_fused_stage1.py') == r['code_sha256']
    assert sha(OUT / 'REGISTRATION_KO.txt') == r['registration_sha256']
    rows = []
    for s in r['shapes']:
        assert s['finished_utc'] - s['started_utc'] < 300
        assert not s['process_timeout'] and s['process_exit_code'] == 0
        cap, raw = s['capture'], s['validation_artifact']
        assert sha(cap['path']) == cap['sha256'] and sha(raw['path']) == raw['sha256']
        inp = torch.load(cap['path'], map_location='cpu', weights_only=True)
        bank = torch.load(raw['path'], map_location='cpu', weights_only=True)
        outputs = bank['outputs']
        assert eq(outputs['BF16_eager'], inp['y'])
        assert eq(outputs['BF16_tuned'], inp['y']) == s['compiled_bf16_bytes_equal']
        bm = metrics(outputs['BF16_tuned'], inp['y'])
        bfvalid = (bm['finite'] and bm['rel_l2'] <= .002 and bm['pointwise_valid']
                   and outputs['BF16_tuned'].shape == inp['y'].shape
                   and outputs['BF16_tuned'].dtype == inp['y'].dtype)
        assert bfvalid == s['validity']['compiled_bf16_numerically_valid']
        assert abs(bm['rel_l2'] - s['compiled_bf16_metric']['rel_l2']) < 1e-6
        exact = bank['qx'].double() @ bank['qw'].double().t()
        assert torch.equal(exact, bank['int32_eager'].double())
        assert eq(bank['int32_eager'], bank['int32_compiled'])
        assert bank['qx'].int().abs().max() <= 127 and bank['qw'].int().abs().max() <= 127
        assert s['shape_MNK'][2] * 127**2 < 2**31
        qdiff = int((bank['qx'] != bank['isolated_compiled_qx']).sum())
        assert qdiff == s['isolated_compiled_quant']['int8_different_elements']
        assert float((bank['sx'] - bank['isolated_compiled_sx']).abs().max()) == s['isolated_compiled_quant']['scale_max_abs']
        ref = exact.float() * bank['sx'] * bank['sw'].t()
        if inp['bias'] is not None:
            ref = ref + inp['bias'].float()
        output_metrics = {}
        for arm in ('W8A8_default', 'W8A8_fused'):
            y = outputs[arm]
            qm, ym = metrics(y.reshape_as(ref), ref), metrics(y, inp['y'])
            valid = (qm['finite'] and qm['rel_l2'] <= .002 and qm['pointwise_valid']
                     and ym['rel_l2'] <= .05 and y.shape == inp['y'].shape and y.dtype == inp['y'].dtype)
            assert valid == s['validity'][arm]
            assert abs(ym['rel_l2'] - s['metrics'][arm]['vs_bf16']['rel_l2']) < 1e-6
            output_metrics[arm] = {'vs_quantized_reference': qm, 'vs_bf16': ym}
        assert eq(outputs['W8A8_default'], outputs['W8A8_fused'])
        assert all(s['validity'].values()) == s['all_valid']
        source_root = Path(raw['path']).parent / 'inductor'
        full_sources = []
        for file in sorted(source_root.rglob('*.py')):
            text = file.read_text()
            if 'def call(args):' in text and 'fused_int_mm' in text:
                call = text.split('def call(args):', 1)[1].split('def benchmark_compiled_module', 1)[0]
                assert 'extern_kernels._int_mm' not in call and 'torch.int32' not in call
                assert 'tl.dot(a, b' in text and 'tmp1 = acc * tmp0' in text
                assert re.search(r'ACC_TYPE\s*:\s*tl\.constexpr\s*=\s*tl\.int32', text)
                m, n, _ = s['shape_MNK']
                assert f'empty_strided_cuda(({m}, {n}), ({n}, 1), torch.float32)' in call
                kernels = re.findall(r'\b(triton_\w+)\.run\(', call)
                assert len(kernels) == 3
                full_sources.append({'path': str(file), 'sha256': sha(file), 'kernel_names': kernels,
                                     'int32_intermediate': False, 'fp32_MxN_intermediate': True})
        assert len(full_sources) == 1
        trace = None
        if s['status'] == 'completed':
            assert s['all_valid'] and len(s['timing']) == 12
            for i, t in enumerate(s['timing']):
                assert t['order'] == ARMS[i % 4:] + ARMS[:i % 4]
            for base in ('BF16_eager', 'BF16_tuned'):
                d = np.array([100 * (1 - t['W8A8_fused']['wall_ms'] / t[base]['wall_ms']) for t in s['timing']])
                rng = np.random.default_rng(20261104)
                ci = np.quantile(np.median(rng.choice(d, (10000, 12)), axis=1), [.025, .975])
                comparison = s['comparisons'][base]
                assert np.array_equal(d, comparison['reductions_percent'])
                assert np.array_equal(ci, comparison['bootstrap_median_95ci_percent'])
                assert float(np.median(d)) == comparison['median_percent']
                assert bool(np.median(d) >= 5 and ci[0] > 0) == comparison['passes']
            assert s['performance_screen_passes'] == all(d['passes'] for d in s['comparisons'].values())
            assert sha(s['trace']['path']) == s['trace']['sha256']
            trace = trace_arms(s['trace']['path'])
            assert trace['W8A8_fused'] == full_sources[0]['kernel_names']
            for c in s['generated_code']:
                assert sha(c['path']) == c['sha256']
        else:
            assert s['status'] == 'invalid' and not bfvalid and not s['timing']
            assert [k for k, v in s['validity'].items() if not v] == ['compiled_bf16_numerically_valid']
            assert 'trace' not in s and 'comparisons' not in s
        rows.append({'shape_MNK': s['shape_MNK'], 'measurement_status': s['status'],
                     'full_int32_cpu_reference_equal': True, 'native_bf16_bytes_equal': True,
                     'bf16_tuned_cpu_metrics': bm, 'bf16_tuned_valid': bfvalid,
                     'default_fused_w8a8_bytes_equal': True, 'isolated_quant_diff_count': qdiff,
                     'output_metrics': output_metrics, 'fused_generated_source': full_sources[0],
                     'trace_kernels_by_arm': trace,
                     'performance_screen_recomputed': s.get('performance_screen_passes'),
                     'measurement_invalidity_preserved': s['status'] == 'invalid'})
    print(json.dumps({'status': 'passed', 'meaning': 'records validated; not all experiment arms passed',
                      'gpu_calls': 0, 'audit_code_sha256': sha(__file__), 'shapes': rows}, indent=2))


if __name__ == '__main__':
    main()
