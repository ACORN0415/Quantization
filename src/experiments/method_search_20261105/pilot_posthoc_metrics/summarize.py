"""Descriptive views of frozen pilot outputs. No new model calls or gate changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'full_obq_rollout_pilot'
ARMS = ('BF16', 'R_bf16', 'R_fp32', 'O_full_gpu')
COLORS = ('black', '#2471a3', '#ca6f1e', '#148f77')
protected = {}

def protect(path, expected=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    protected[str(path)] = digest
    return path

def main():
    started = time.monotonic()
    assert not (HERE / 'RESULT.json').exists()
    quality = json.loads(protect(PILOT / 'QUALITY.json', '6cd48cdafb0e3f10029dfae05ceeb2a554fd2cf4dce2515496ced6ee8caa3d1d').read_text())
    assert quality['status'] == 'completed' and quality['decision']['decision'] == 'stop_quality'
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    rows = []
    for i, inp in enumerate(('coast_cycle', 'rain_market')):
        for arm, color in zip(ARMS, COLORS):
            entry = next(x for x in quality['rows'] if x['input_id'] == inp and x['arm'] == arm)
            sample = json.loads(protect(PILOT / inp / arm / 'sampling.json').read_text())
            features = torch.load(protect(entry['features']['path'], entry['features']['sha256']), map_location='cpu', weights_only=True)['features'].double()
            features = torch.nn.functional.normalize(features, dim=-1)
            assert features.shape[0] == 501 and torch.isfinite(features).all()
            late = features[249:501]
            # Trace of empirical covariance (denominator n) of normalized frame features.
            dispersion = float((late - late.mean(0)).square().sum(-1).mean())
            lag12 = float((late[12:] * late[:-12]).sum(-1).mean())
            lag60 = float((late[60:] * late[:-60]).sum(-1).mean())
            rows.append({'input_id': inp, 'arm': arm, 'late_feature_dispersion': dispersion,
                         'late_cosine_lag_12_frames': lag12, 'late_cosine_lag_60_frames': lag60,
                         'late_musiq': entry['windows']['late']['musiq'],
                         'late_dino_reference': entry['windows']['late']['dino_reference']})
            nses = [x.get('latent_metric', {}).get('nse', 0.0) for x in sample['chunks']]
            assert len(nses) == 42
            axes[i, 0].plot(range(42), nses, color=color, label=arm)
            # First latent chunk reconstructs 9 frames, subsequent chunks 12 frames.
            chunk_ids = np.minimum((np.arange(501) + 3) // 12, 41)
            for j, key in ((1, 'musiq_per_frame'), (2, 'dino_reference_per_frame')):
                values = np.asarray(entry[key], dtype=float)
                assert values.shape == (501,) and np.isfinite(values).all()
                means = [float(values[chunk_ids == c].mean()) for c in range(42)]
                axes[i, j].plot(range(42), means, color=color)
        for j, label in enumerate(('Latent NSE to BF16', 'MUSIQ', 'DINO cosine to BF16')):
            axes[i, j].set_title(f'{inp}: {label}')
            axes[i, j].set_xlabel('Chunk index (0-based)')
            axes[i, j].axvline(21, color='gray', linestyle=':', linewidth=1)
            axes[i, j].grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Frozen pilot: descriptive trajectories; registered decision remains stop_quality')
    fig.savefig(HERE / 'trajectories.png', dpi=150)
    fig.savefig(HERE / 'trajectories.svg')
    plt.close(fig)
    with (HERE / 'feature_summary.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for name, digest in protected.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    assert not torch.cuda.is_initialized()
    result = {'status': 'completed_descriptive_only', 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'original_decision': quality['decision']['decision'], 'decision_changed': False,
              'new_generation_calls': 0, 'new_metric_network_calls': 0, 'GPU_calls': 0,
              'protected_inputs': protected, 'rows': rows, 'elapsed_seconds': time.monotonic() - started,
              'interpretation': 'Feature dispersion and fixed-lag cosine are descriptive, not validated motion/quality measures; no independent-frame statistics or causal inference.'}
    (HERE / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'seconds': result['elapsed_seconds'], 'rows': rows}, indent=2))

if __name__ == '__main__':
    main()
