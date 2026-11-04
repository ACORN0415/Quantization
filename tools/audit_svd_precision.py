"""Read-only CPU audit of stored C artifacts. Never reruns SVD; no current-time gate."""
import argparse
import json
import math
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import numpy as np
import torch
import codec_order_core as core
import svd_precision_probe as probe


def close(a, b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert abs(float(a) - float(b)) <= 1e-11 + 2e-9 * max(abs(float(a)), abs(float(b))), (a, b)


def metrics(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    sse = float(np.sum((a - b) ** 2)); norm = float(np.sum(b ** 2))
    return dict(sse=sse, mse=sse / a.size, normalized_squared_error=sse / norm, relative_l2=math.sqrt(sse / norm))


def match_metrics(a, b):
    assert a.keys() == b.keys()
    for k in a:
        close(a[k], b[k])


def factor_audit(raw, packed):
    flat = raw.float().numpy().reshape(-1); n = flat.size; bits = packed['bits']; qmax = (1 << (bits - 1)) - 1
    padded = np.pad(flat, (0, (-n) % 16))
    maximum = np.maximum(np.max(np.abs(padded.reshape(-1, 16)), axis=1), np.float32(2**-24)).astype(np.float16)
    assert maximum.tobytes() == packed['maximum'].numpy().tobytes()
    scales = np.repeat(maximum.astype(np.float32), 16)[:n] / np.float32(qmax)
    code = (np.clip(np.rint(flat / scales), -qmax, qmax) + qmax).astype(np.uint8)
    padded_code = np.pad(code, (0, (-n) % (8 // bits))).reshape(-1, 8 // bits)
    stream = np.zeros(padded_code.shape[0], np.uint8)
    for j in range(8 // bits):
        stream |= padded_code[:, j] << (j * bits)
    assert stream.tobytes() == packed['stream'].numpy().tobytes()
    factor = probe.decode_factor_numpy(packed)
    assert factor.tobytes() == core.unpack_factor(packed).numpy().tobytes()
    return factor


def raw_reconstruction(raw):
    a, b, c = [f.double().numpy() for f in raw['factors']]
    h, t, s, d = raw['shape']; r = raw['rank']; rd = min(r, d)
    heads = []
    for i in range(h):
        middle = (b[i].reshape(r * t, rd) @ c[i]).reshape(r, t * d)
        heads.append((a[i] @ middle).reshape(s, t, d).transpose(1, 0, 2))
    return np.stack(heads).transpose(1, 2, 0, 3).reshape(1, t * s, h, d)


def attention(p):
    q = p['query'][0].double().numpy().transpose(1, 0, 2)
    k = p['key'][0].double().numpy().transpose(1, 2, 0)
    logits = q @ k / np.sqrt(128.)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs[:, :, -4680:], p['wo'].double().numpy()


def output(v, p, wo):
    z = p @ np.asarray(v, np.float64)[0].transpose(1, 0, 2)
    return z.transpose(1, 0, 2).reshape(32, 1536) @ wo.T


def audit():
    root = probe.OUT; r = json.loads((root / 'results.json').read_text())
    assert r['status'] == 'completed' and r['gpu_calls'] == 0 and r['finished_utc'] < probe.DEADLINE
    assert r['provenance'] == probe.provenance()
    execution = json.loads((root / 'execution.json').read_text()); assert execution['status'] == 'completed'
    assert [(a['prompt'], a['precision']) for a in execution['workers']] == [(0, 32), (0, 64), (4, 64), (4, 32)]
    assert all(a['returncode'] == 0 and a['finished_utc'] < probe.DEADLINE for a in execution['workers'])
    fact_count = payload_count = 0; scores = {}; raw_outputs = {}
    for row in r['rows']:
        prompt, precision = row['prompt'], row['precision']
        assert row == json.loads((root / f'p{prompt:02d}_fp{precision}.json').read_text())
        assert row['provenance'] == probe.provenance() and row['repeat_factor_byte_equal']
        assert len(row['svd_calls']) == 6
        assert [a['shape'] for a in row['svd_calls']] == [[12, 1560, 384], [12, 576, 128]] * 3
        assert all(a['dtype'] == f'torch.float{precision}' for a in row['svd_calls'])
        assert probe.sha(row['input']['path']) == row['input']['sha256']
        assert probe.sha(row['raw']['path']) == row['raw']['sha256']
        capture = torch.load(row['input']['path'], map_location='cpu', weights_only=True)
        blob = torch.load(row['raw']['path'], map_location='cpu', weights_only=True); raw = blob['raw']
        assert all(str(f.dtype) == f'torch.float{precision}' for f in raw['factors'])
        v = capture['new_value'].double().numpy(); probs, wo = attention(capture); ref_o = output(v, probs, wo)
        raw_v = raw_reconstruction(raw); raw_outputs[prompt, precision] = raw_v
        match_metrics(metrics(raw_v, v), row['raw_v_error'])
        match_metrics(metrics(output(raw_v, probs, wo), ref_o), row['raw_output_error'])
        close(float(np.sum(ref_o ** 2)), row['reference_chunk_output_norm_sq'])
        for arm in row['arms']:
            bits = arm['bits']; artifact = arm['artifact']; assert probe.sha(artifact['path']) == artifact['sha256']
            saved = torch.load(artifact['path'], map_location='cpu', weights_only=True); state = saved['state']
            assert state['bits'] == bits and state['rank'] == 192 and state['shape'] == (12, 3, 1560, 128)
            for f, packed in zip(raw['factors'], state['factors']):
                factor_audit(f, packed); fact_count += 1
            assert core.payload_bytes(state) == arm['payload_tensor_bytes'] == core.expected_bytes('tt', state['shape'], 192, bits)
            decoded = core.decode(state)
            assert probe.same(decoded.to(torch.bfloat16), saved['restored'])
            restored = saved['restored'].double().numpy(); final_o = output(restored, probs, wo)
            assert np.allclose(final_o, saved['output'].numpy(), atol=1e-10, rtol=2e-9)
            assert np.allclose(ref_o, saved['reference_output'].numpy(), atol=1e-10, rtol=2e-9)
            match_metrics(metrics(decoded.double().numpy(), v), arm['decoded_fp32_v_error'])
            match_metrics(metrics(restored, v), arm['restored_bf16_v_error'])
            match_metrics(metrics(restored, raw_v), arm['storage_change_from_raw'])
            err = metrics(final_o, ref_o); match_metrics(err, arm['output_error'])
            scores[prompt, precision, bits] = (metrics(restored, v)['sse'], err['mse'])
            payload_count += 1
    for rawrow in r['raw_distances']:
        p = rawrow['prompt']; match_metrics(metrics(raw_outputs[p, 32], raw_outputs[p, 64]), rawrow['fp32_vs_fp64_raw_reconstruction'])
    comps = []
    for c in r['comparisons']:
        p, bits = c['prompt'], c['bits']; a, b = scores[p, 32, bits], scores[p, 64, bits]
        vv, oo = 1 - b[0] / a[0], 1 - b[1] / a[1]
        close(vv, c['v_sse_reduction']); close(oo, c['output_mse_reduction']); comps.append((bits, vv, oo))
    successful = [b for b in (2, 4) if all(v >= .01 and o >= .01 for bits, v, o in comps if bits == b)]
    tiny = all(abs(v) < .01 and abs(o) < .01 for _, v, o in comps)
    expected = 'precision_effect_followup_justified' if successful else ('close_precision_upgrading_for_tested_cases' if tiny else 'mixed_effect_not_adopted')
    assert expected == r['decision'] and successful == r['successful_bits'] and tiny == r['all_changes_below_one_percent']
    total_bytes = sum(p.stat().st_size for p in probe.DATA.rglob('*') if p.is_file()); assert total_bytes <= 1024**3
    assert not torch.cuda.is_initialized()
    return {'status': 'passed', 'gpu_calls': 0, 'svd_reruns': 0, 'worker_records': 4,
            'actual_svd_calls_checked': 24, 'factor_streams_independently_requantized_and_decoded': fact_count,
            'payloads': payload_count, 'score_arms_recomputed': 8, 'raw_arms_recomputed': 4,
            'data_bytes': total_bytes, 'decision_recomputed': expected, 'audit_code_sha256': probe.sha(__file__),
            'note': 'Stored source factors, packing and scores audited; original SVD repeat logs/sign controls not rerun.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output'); args = parser.parse_args()
    result = audit()
    if args.output:
        probe.save(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
