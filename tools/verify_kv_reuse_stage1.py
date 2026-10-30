"""CPU-only audit of saved stage-1 evidence; does not replay GPU operations."""
import hashlib
import json
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'src/experiments/kv_reuse_stage1'
DATA = Path.home() / 'sfq_data/kv_reuse_stage1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    torch.set_num_threads(2)
    summary = read(OUT / 'summary.json')
    manifest = read(OUT / 'manifest.json')
    assert summary['reason'] == 'complete', 'Only audit a completed stage here'
    assert summary['completed'] == len(summary['results']) == 12
    assert len(summary['budget']['attempts']) == 12
    assert summary['budget']['elapsed_s'] <= 7200
    for path, expected in manifest['own_files'].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in manifest['upstream_files'].items():
        assert sha(Path(manifest['source']) / path) == expected, path
    assert sha(Path(manifest['prompt_file'])) == manifest['prompt_file_sha256']

    reference = torch.load(DATA / 'gate_U/latent.pt', map_location='cpu', weights_only=True)['latent']
    audited = []
    byname = {}
    for name in summary['results']:
        result = read(OUT / (name + '.json'))
        assert result == read(DATA / name / 'result.json')
        assert result['status'] == 'valid'
        path = DATA / name / 'latent.pt'
        assert sha(path) == result['latent_sha256']
        latent = torch.load(path, map_location='cpu', weights_only=True)['latent']
        assert latent.shape == reference.shape and latent.dtype == reference.dtype
        assert torch.equal(latent.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))
        assert result['counts']['decode_calls'] == {'U': 2400, 'B': 1200, 'R': 1048}[result['arm']]
        assert result['counts']['read_calls'] == 3660
        telemetry = read(DATA / name / 'telemetry.json')
        assert telemetry
        assert all(not x['external_pids'] and x['temp_c'] < 83 and x['process_bytes'] <= 20 * 2**30 for x in telemetry)
        assert max(x['process_bytes'] for x in telemetry) == result['process_sample_peak']
        assert result['peak_reserved'] <= 18 * 2**30
        audited.append(name)
        byname[name] = result
    for pair in summary['pairs']:
        i = pair['pair']
        assert pair['B_s'] == byname[f'pair{i}_B']['generation_s']
        assert pair['R_s'] == byname[f'pair{i}_R']['generation_s']
        assert pair['reduction_fraction'] == 1 - pair['R_s'] / pair['B_s']

    offset = 0
    raw_bytes = 0
    index = DATA / 'attention_tape/index.jsonl'
    records = [json.loads(line) for line in index.read_text().splitlines()]
    assert len(records) == 5400
    for i, row in enumerate(records):
        assert row['offset'] == offset
        assert row['key']['forward'] == i // 90
        assert row['key']['layer'] == (i // 3) % 30
        assert row['key']['tensor'] == ('q', 'k', 'v')[i % 3]
        offset += row['packed_bytes']
        raw_bytes += row['bytes']
    tape = DATA / 'attention_tape/inputs.zst'
    assert offset == tape.stat().st_size == byname['gate_U']['tape']['compressed_bytes']
    for name in ('gate_U', 'gate_B', 'gate_R'):
        assert byname[name]['tape']['records'] == len(records)
        assert byname[name]['tape']['compared_or_recorded_bytes'] == raw_bytes
    print('Saved latents, source fingerprints, counters, telemetry and index checked; hashing tape.', flush=True)
    audit = dict(
        completed_cases=audited,
        direct_latent_byte_comparisons_including_reference=len(audited),
        attention_tape_records=len(records),
        attention_raw_bytes_per_gate=raw_bytes,
        attention_tape_sha256=sha(tape),
        attention_index_sha256=sha(index),
        own_source_files=len(manifest['own_files']),
        upstream_source_files=len(manifest['upstream_files']),
        scope='CPU recheck of saved evidence. Counters are stored aggregates; no GPU replay or second full attention-input comparison.',
        seconds_since_stage_start=time.time() - summary['budget']['start_unix'],
    )
    (OUT / 'verification.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Saved evidence audit passed.', flush=True)


if __name__ == '__main__':
    main()
