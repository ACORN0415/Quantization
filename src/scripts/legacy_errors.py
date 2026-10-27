"""Reproduce historical aggregation boundaries without contaminating raw logs."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('raw', type=Path)
ap.add_argument('reference', type=Path)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()
raw = json.loads(a.raw.read_text())
if any(r.get('kv_error_reference','pre_requantization_buffer') != 'pre_requantization_buffer' for r in raw):
    raise SystemExit('Historical comparison requires rolling pre-requantization errors. For policy-neutral diagnostics compare source_k_rel/source_v_rel instead.')
if any('k_rel' not in r or 'v_rel' not in r for r in raw):
    raise SystemExit('Raw records are missing k_rel/v_rel; regenerate with KV logging enabled.')
rows, pending = [], []
for r in raw:
    pending.append(r)
    if r['call_phase'] == 'denoise_3' and r['layer'] == 29:
        rows.append({'prompt': r['prompt'], 'chunk_idx': r['chunk'],
                     'n_records': len(pending),
                     'k_rel_mean': sum(x['k_rel'] for x in pending) / len(pending),
                     'v_rel_mean': sum(x['v_rel'] for x in pending) / len(pending),
                     'tokens': pending[-1]['tokens']})
        pending = []
ref = json.loads(a.reference.read_text())['by_chunk']
errors = []
for i, (got, expected) in enumerate(zip(rows, ref)):
    for field in ('chunk_idx', 'n_records', 'tokens'):
        if got[field] != expected[field]:
            errors.append({'row': i, 'field': field, 'got': got[field], 'expected': expected[field]})
    for field in ('k_rel_mean', 'v_rel_mean'):
        rel = abs(got[field] - expected[field]) / max(abs(expected[field]), 1e-12)
        if rel > 1e-6:
            errors.append({'row': i, 'field': field, 'got': got[field], 'expected': expected[field], 'relative_difference': rel})
report = {'pass': len(rows) == len(ref) and not errors, 'n_rows': len(rows),
          'reference_rows': len(ref), 'mismatches': errors, 'by_chunk': rows}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k != 'by_chunk'}, indent=2))
raise SystemExit(0 if report['pass'] else 1)
