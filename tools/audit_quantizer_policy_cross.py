"""Read-only arithmetic/artifact audit; never imports the GPU experiment worker."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def audit(data, results):
    decision = read(results/'decision.json')
    source = decision['source_sha256']
    rows = list(csv.DictReader((results/'cases.csv').open()))
    expected = set(itertools.product((0, 4), (101,), ('rtn', 'kivi'), ('L', 'C_repeat')))
    actual = {(int(r['prompt']), int(r['seed']), r['method'], r['arm']) for r in rows}
    assert len(rows) == 8 and actual == expected
    budgets = []
    for r in rows:
        folder = data/'development'/f"p{int(r['prompt']):02d}_s101_{r['method']}_{r['arm']}"
        s = read(folder/'sampling.json'); q = read(folder/'quality.json'); resource = read(folder/'quality_resource.json')
        assert all(v['source_sha256'] == source for v in (s, q, resource))
        assert (s['prompt'], s['seed'], s['method'], s['arm']) == (
            int(r['prompt']), int(r['seed']), r['method'], r['arm'])
        assert s['group'] == 16 and s['bits'] == 4
        assert q['latent_file_sha256'] == s['latent_file_sha256']
        # A cached evaluation must not replace the actual measured resources
        # with the cost of returning already-computed scores.
        measured_quality_phases = sum(q[name] for name in ('vae_decode_s', 'quality_eval_s', 'mp4_encode_s'))
        assert resource['seconds'] + 1e-6 >= measured_quality_phases
        scores = q['per_frame_musiq']
        assert len(scores) == 501 and all(math.isfinite(v) for v in scores)
        assert abs(statistics.mean(scores[249:501])-float(r['late_raw_musiq'])) < 1e-10
        for name, positions in (
                ('boundary_raw_musiq', [12*c-3+o for c in range(21,42) for o in range(-2,3)]),
                ('inside_raw_musiq', [12*c-3+o for c in range(21,42) for o in range(4,9)])):
            assert abs(statistics.mean(scores[i] for i in positions)-float(r[name])) < 1e-10
        components = ['valid_code_bytes', 'padding_code_bytes', 'scale_bytes', 'offset_bytes',
                      'bf16_pending_bytes', 'bf16_residual_bytes']
        assert sum(int(r[name]) for name in components) == int(r['resident_tensor_bytes'])
        if r['method'] == 'kivi':
            assert int(r['offset_bytes']) > 0
        else:
            assert int(r['offset_bytes']) == 0
        assert (int(r['bf16_pending_bytes']) > 0) == (r['arm'] == 'C_repeat')
        wall = s['setup_s']+s['generation_s']+resource['seconds']
        peak = max(s['peak']['allocated'], resource['peak_allocated_bytes'])
        cpu = max(s['cpu_lifetime_high_water_bytes'], resource['cpu_lifetime_high_water_bytes'])
        budget = wall < 600 and peak < 22*2**30 and cpu < 48*2**30
        assert budget == (r['budget_pass'] == 'True')
        budgets.append(budget)
        if int(r['prompt']) == 0 and r['method'] == 'rtn':
            assert s['protected_rtn_latent_bitexact']
    interactions = []
    for prompt in (0,4):
        cells = {(r['method'],r['arm']):float(r['late_raw_musiq']) for r in rows if int(r['prompt']) == prompt}
        effect = (cells['kivi','C_repeat']-cells['rtn','C_repeat'])-(cells['kivi','L']-cells['rtn','L'])
        interactions.append(effect)
        recorded = next(v['interaction'] for v in decision['interactions'] if v['prompt'] == prompt)
        assert abs(effect-recorded) < 1e-10
        replay = read(data/'replay'/f'p{prompt:02d}_s101.json')
        assert replay['source_sha256'] == source and replay['teacher_bitexact_chunks'] == 42
        assert len(replay['rows']) == 6*42*5*4
        ids = {(r['layer'],r['chunk'],r['call'],r['method'],r['arm']) for r in replay['rows']}
        assert len(ids) == len(replay['rows'])
        assert {r['layer'] for r in replay['rows']} == {0,5,10,15,20,29}
        for row in replay['rows']:
            assert row['history_tokens'] == min(row['chunk'],6)*4680
            assert all(math.isfinite(row[key]) for key in ('k_rel','v_rel'))
        for method, arm in itertools.product(('rtn', 'kivi'), ('L', 'C_repeat')):
            selected = [r for r in replay['rows'] if r['method'] == method and r['arm'] == arm and r['chunk'] >= 7]
            assert len(selected) == 6*35*5
            summary = [r for r in decision['replay_summary'] if
                       r['prompt'] == prompt and r['method'] == method and r['arm'] == arm]
            assert len(summary) == 1 and summary[0]['layers'] == replay['layers']
            for key in ('k_rel', 'v_rel'):
                assert abs(statistics.mean(r[key] for r in selected)-summary[0][key]) < 1e-12
    effect_pass = interactions[0]*interactions[1] > 0 and all(abs(v) >= 1 for v in interactions)
    assert decision['effect_pass'] == effect_pass
    assert decision['resource_pass'] == all(budgets)
    assert decision['advance'] == (effect_pass and all(budgets))
    assert decision['gates']['passed']
    assert [r['bitexact_chunks'] for r in decision['gates']['full_model']] == [42,42]
    expected_artifacts = {f"development/p{prompt:02d}_s101_{method}_{arm}/{name}"
        for prompt, method, arm, name in itertools.product((0,4), ('rtn','kivi'), ('L','C_repeat'),
            ('latent.pt','sampling.json','quality.json','quality_resource.json','video.mp4'))}
    assert len(decision['artifacts']) == 40
    assert {item['path'] for item in decision['artifacts']} == expected_artifacts
    for item in decision['artifacts']:
        path = data/item['path']
        assert path.stat().st_size == item['bytes']
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024), b''):
                h.update(block)
        assert h.hexdigest() == item['sha256'], item['path']
    print(json.dumps(dict(passed=True, cases=8, development_prompts=2, interactions=interactions,
        advance=decision['advance'], artifacts_verified=len(decision['artifacts']),
        note='same-author read-only recomputation, not independent external validation'), ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path.home()/'sfq_data/quantizer_policy_cross')
    ap.add_argument('--results', type=Path, default=ROOT/'src/experiments/quantizer_policy_cross_development')
    args = ap.parse_args()
    audit(args.data, args.results)
