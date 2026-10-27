"""Compare full reference latents or per-chunk captures; nonzero on mismatch."""
import argparse
import json
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument('reference', type=Path)
ap.add_argument('candidate', type=Path)
ap.add_argument('--report', type=Path)
a = ap.parse_args()
results = []
for p in sorted(a.reference.glob('prompt*')):
    files = sorted(p.glob('chunk_*.pt'))
    full = not files
    if full:
        files = [p / 'full.pt']
    for f in files:
        ref = torch.load(f, map_location='cpu', weights_only=True)['latent']
        dest = a.candidate / p.name / f.name
        if dest.exists():
            got = torch.load(dest, map_location='cpu', weights_only=True)['latent']
        elif full:
            parts = sorted((a.candidate / p.name).glob('chunk_*.pt'))
            if not parts:
                results.append({'file': str(f.relative_to(a.reference)), 'equal': False, 'missing': True})
                continue
            got = torch.cat([torch.load(q, map_location='cpu', weights_only=True)['latent'] for q in parts], 1)
        else:
            results.append({'file': str(f.relative_to(a.reference)), 'equal': False, 'missing': True})
            continue
        equal = torch.equal(ref, got)
        results.append({'file': str(f.relative_to(a.reference)), 'equal': equal,
                        'max_abs': (ref-got).abs().max().item() if ref.shape == got.shape else None})
report = {'pass': bool(results) and all(r['equal'] for r in results), 'comparisons': results}
if a.report:
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['pass'] else 1)
