"""Read-only result arithmetic and artifact audit; no torch/GPU worker imports.

This recomputes reported aggregates, checks capture/source manifests, and hashes
all captured files. It does not re-run the codec, attention, or generation and is
same-author verification rather than independent external validation.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = (0, 4)
CHUNKS = (0, 1, 6, 7)
LAYERS = (0, 14, 29)
ARMS = ('direct', 'same_position', 'shared_translation',
        'matched_border_same_position', 'per_layer_translation')
SOURCE_FILES = ('tools/motion_codec_probe.py', 'tools/motion_codec_core.py',
                'tools/test_motion_codec_core.py', 'src/experiments/MOTION_CODEC_DEVELOPMENT.md')


def require(condition, description):
    if not condition:
        raise ValueError(description)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def numeric(value, description, *, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value), f'nonfinite/non-numeric {description}')
    require(not nonnegative or value >= 0, f'negative {description}')
    return value


def close(actual, expected, description):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14),
            f'{description}: {actual!r} != {expected!r}')


def all_finite(value, description):
    if isinstance(value, dict):
        for key, item in value.items():
            all_finite(item, description+'.'+key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            all_finite(item, description+f'[{index}]')
    elif type(value) in (int, float):
        numeric(value, description)


def diagnostic_source_hash():
    h = hashlib.sha256()
    for rel in SOURCE_FILES:
        h.update(rel.encode())
        h.update((ROOT/rel).read_bytes())
    return h.hexdigest()


def model_source_hash():
    # Literal reconstruction of p3_tf_fr.source_hash, without importing torch.
    src = ROOT/'src'
    h = hashlib.sha256()
    for folder in ('kvq', 'pipeline', 'wan', 'utils'):
        for path in sorted((src/folder).rglob('*.py')):
            h.update(path.read_bytes())
    for rel in ('scripts/p3_tf_fr.py', 'scripts/anchor_identity_gate.py',
                'experiments/P3_TF_FR_PILOT.md'):
        h.update((src/rel).read_bytes())
    return h.hexdigest()


def audit(data, results):
    decision = read(results/'decision.json')
    all_finite(decision, 'decision')
    source = diagnostic_source_hash()
    model_source = model_source_hash()
    require(decision['source_sha256'] == source, 'decision source differs from current frozen files')
    with (results/'rows.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 240, 'expected 240 complete rows')
    expected = set(itertools.product(PROMPTS, CHUNKS, (0, 1), LAYERS, ARMS))
    cells = {}
    for row in rows:
        key = (int(row['prompt']), int(row['chunk']), int(row['pair']), int(row['layer']), row['arm'])
        require(key not in cells, f'duplicate row {key}')
        require(key in expected, f'unexpected row {key}')
        cells[key] = row
        for name in ('k_rel', 'v_rel', 'k_post_rope_rel', 'error_mean', 'search_s',
                     'encode_decode_metric_wall_s'):
            row[name] = numeric(float(row[name]), f'{key}/{name}', nonnegative=True)
        for name in ('pair_resident_payload_bytes', 'pair_packed_estimate_bytes',
                     'anchor_resident_bytes', 'reference_restore_workspace_bytes'):
            value = int(row[name])
            require(value > 0, f'nonpositive byte count {key}/{name}')
            row[name] = value
        close(row['error_mean'], (row['k_rel']+row['v_rel'])/2, f'{key}/error_mean')
        require(row['pair_packed_estimate_bytes'] <= row['pair_resident_payload_bytes'],
                f'packed estimate exceeds uint8 payload {key}')
        require(row['anchor_resident_bytes'] < row['pair_resident_payload_bytes'],
                f'anchor omitted or payload too small {key}')
        if row['arm'] == 'direct':
            require(row['dy'] == row['dx'] == '', f'direct arm must have no translation {key}')
        else:
            row['dy'], row['dx'] = int(row['dy']), int(row['dx'])
            require(all(-2 <= row[name] <= 2 for name in ('dy', 'dx')), f'out of search range {key}')
        if row['arm'] == 'same_position':
            require(row['dy'] == row['dx'] == 0, f'same-position translation changed {key}')
        if row['arm'] not in ('shared_translation', 'per_layer_translation'):
            require(row['search_s'] == 0, f'unexpected search charge {key}')
    require(set(cells) == expected, 'incomplete Cartesian product')

    for prompt, chunk, pair, layer in itertools.product(PROMPTS, CHUNKS, (0, 1), LAYERS):
        group = {arm: cells[prompt, chunk, pair, layer, arm] for arm in ARMS}
        direct = group['direct']
        shared = group['shared_translation']
        border = group['matched_border_same_position']
        shared_l0 = cells[prompt, chunk, pair, 0, 'shared_translation']
        require((shared['dy'], shared['dx']) == (shared_l0['dy'], shared_l0['dx']),
                f'shared translation not shared {prompt}/{chunk}/{pair}/{layer}')
        close(shared['search_s'], shared_l0['search_s'], 'shared search time repeated across layers')
        require((border['dy'], border['dx']) == (shared['dy'], shared['dx']),
                'border control mask differs from shared translation')
        for arm, row in group.items():
            require(row['anchor_resident_bytes'] == direct['anchor_resident_bytes'], 'anchor payload changed')
            require(row['reference_restore_workspace_bytes'] == direct['reference_restore_workspace_bytes'],
                    'reference workspace changed')
            # Same geometry, bit width and groups: only two K/V 4-byte shifts,
            # and the matched-border control's additional two masks, can differ.
            overhead = 0 if arm == 'direct' else 16 if arm == 'matched_border_same_position' else 8
            for name in ('pair_resident_payload_bytes', 'pair_packed_estimate_bytes'):
                require(row[name] == direct[name]+overhead, f'payload/map byte mismatch {arm}/{name}')
        require(direct['pair_resident_payload_bytes'] == 2*direct['anchor_resident_bytes'],
                'direct pair must store anchor and target with identical geometry')
        if shared['dy'] == shared['dx'] == 0:
            for arm in ('shared_translation', 'matched_border_same_position'):
                for name in ('k_rel', 'v_rel', 'k_post_rope_rel', 'error_mean'):
                    close(group[arm][name], group['same_position'][name], f'zero-shift equivalence {arm}/{name}')
        if layer == 0:
            oracle = group['per_layer_translation']
            require((oracle['dy'], oracle['dx']) == (shared['dy'], shared['dx']), 'layer0 search disagreement')
            for name in ('k_rel', 'v_rel', 'k_post_rope_rel', 'error_mean'):
                close(oracle[name], shared[name], f'layer0 shared/oracle metric {name}')

    require(len(decision['decisions']) == 2, 'expected two prompt-level decisions')
    require({item['prompt'] for item in decision['decisions']} == set(PROMPTS), 'prompt decision IDs')
    recomputed = []
    for prompt in PROMPTS:
        means = {}
        for arm in ARMS:
            selected = [r for key, r in cells.items() if key[0] == prompt and key[4] == arm]
            require(len(selected) == 24, f'wrong number of prompt/arm cells {prompt}/{arm}')
            means[arm] = statistics.mean(row['error_mean'] for row in selected)
        require(means['same_position'] > 0 and means['matched_border_same_position'] > 0,
                'zero denominator in progress rule')
        reduction = 1-means['shared_translation']/means['same_position']
        border_reduction = 1-means['shared_translation']/means['matched_border_same_position']
        passed = reduction >= .05 and border_reduction >= .05
        recorded = next(item for item in decision['decisions'] if item['prompt'] == prompt)
        require(set(recorded['means']) == set(ARMS), 'summary arm set changed')
        for arm in ARMS:
            close(recorded['means'][arm], means[arm], f'prompt {prompt} mean {arm}')
        close(recorded['shared_vs_same_relative_reduction'], reduction, 'same-position reduction')
        close(recorded['shared_vs_matched_border_relative_reduction'], border_reduction, 'border reduction')
        require(type(recorded['passes']) is bool and recorded['passes'] == passed, 'prompt progress rule')
        recomputed.append(dict(prompt=prompt, means=means, shared_vs_same_relative_reduction=reduction,
                               shared_vs_matched_border_relative_reduction=border_reduction, passes=passed))
    advance = all(item['passes'] for item in recomputed)
    require(type(decision['advance_restricted_variant']) is bool and
            decision['advance_restricted_variant'] == advance, 'overall both-prompt progress rule')

    require(len(decision['captures']) == 2, 'expected two capture manifests')
    require({item['prompt'] for item in decision['captures']} == set(PROMPTS), 'capture prompt IDs')
    verified = []
    for prompt in PROMPTS:
        manifest = read(data/f'p{prompt:02d}_s101/manifest.json')
        all_finite(manifest, f'manifest p{prompt}')
        require(manifest == next(item for item in decision['captures'] if item['prompt'] == prompt),
                f'embedded manifest differs from capture manifest p{prompt}')
        require(manifest['prompt'] == prompt and manifest['seed'] == 101, 'capture identity mismatch')
        require(manifest['source_sha256'] == source and manifest['source_model_sha256'] == model_source,
                'capture diagnostic/model source mismatch')
        require(manifest['teacher_bitexact_chunks'] == 8 and manifest['rope_identity_records'] == 12,
                'missing BF16 or RoPE gates')
        require(len(manifest['artifacts']) == 12, 'expected 12 files per prompt')
        expected_files = {f'p{prompt:02d}_s101/c{chunk:02d}_l{layer:02d}.pt': (chunk, layer)
                          for chunk, layer in itertools.product(CHUNKS, LAYERS)}
        seen = set()
        for item in manifest['artifacts']:
            rel = item['file']
            require(rel in expected_files and rel not in seen, f'duplicate/unexpected capture {rel}')
            seen.add(rel)
            require((item['chunk'], item['layer']) == expected_files[rel], 'capture file ID mismatch')
            path = data/rel
            require(path.stat().st_size == item['bytes'] and item['bytes'] > 0, f'artifact size {rel}')
            require(sha(path) == item['sha256'], f'artifact SHA256 {rel}')
            verified.append(rel)
        require(seen == set(expected_files), f'incomplete capture p{prompt}')
    require(len(verified) == len(set(verified)) == 24, 'expected exactly 24 unique verified artifacts')
    status = read(data/'status.json')
    require(status['state'] == 'pass' and status['advance_restricted_variant'] == advance,
            'final status does not match completed result')
    return dict(passed=True, rows=240, development_prompts=2, per_prompt_arm_cells=24,
                artifacts_verified=24, source_sha256=source, model_source_sha256=model_source,
                advance_restricted_variant=advance, recomputed=recomputed,
                limits=[
                    'same-author read-only arithmetic/hash audit; not external independent validation',
                    'capture gates are manifest claims; raw numerical gates are not re-executed here',
                    'actual uint8 tensor payload differs from hypothetical packed bytes; no deployment savings claim',
                    'K primary metric is pre-RoPE reconstruction, not attention/output or video quality',
                    'unwarmed codec+metric timing excludes anchor encoding and reference restore; not throughput',
                    'global +/-2 translation probe does not test dense motion or reproduce TempoKV'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path.home()/'sfq_data/motion_codec')
    parser.add_argument('--results', type=Path, default=ROOT/'src/experiments/motion_codec_development')
    args = parser.parse_args()
    print(json.dumps(audit(args.data, args.results), ensure_ascii=False, indent=2))
