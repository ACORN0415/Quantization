"""Exact Bernoulli risk diagnostic, frozen BF16 K/new-V contribution only."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from change_mixed_core import PackedMixed
from postwo_round_core import _project
from postwo_round_probe import probabilities, free_gpu, sha, save

DATA = Path.home() / 'sfq_data/randomized_correction'
OUT = ROOT / 'src/experiments/randomized_correction_development'


def allocation(score, budget):
    """Minimize sum(score²/p), bounded p, by water filling (oracle for its score)."""
    if not 0.02 <= budget <= 1:
        raise ValueError('budget outside bounds')
    score = score.double().clamp_min(1e-30)
    lo, hi = 0., 1. / float(score.min())
    for _ in range(120):
        mid = (lo + hi) / 2
        if float((score * mid).clamp(.02, 1).mean()) < budget:
            lo = mid
        else:
            hi = mid
    p = (score * ((lo + hi) / 2)).clamp(.02, 1)
    if abs(float(p.mean()) - budget) > 1e-10:
        raise RuntimeError('allocation budget mismatch')
    return p


def risk(errors, p, gamma):
    """errors [windows,queries,output]; selection shared across output channels."""
    d = errors.double()
    p = torch.as_tensor(p, dtype=torch.float64).expand(d.shape[1])
    gamma = torch.as_tensor(gamma, dtype=torch.float64).expand_as(p)
    p, gamma = p[None, :, None], gamma[None, :, None]
    # Enumerate I=0,1 instead of relying on a noisy Monte Carlo mean.
    e0, e1 = d, (1-gamma/p)*d
    mean = (1-p)*e0 + p*e1
    mse = ((1-p)*e0.square() + p*e1.square()).mean()
    bias = mean.square().mean()
    variance = ((1-p)*(e0-mean).square() + p*(e1-mean).square()).mean()
    formula = ((1-2*gamma+gamma.square()/p)*d.square()).mean()
    if not torch.allclose(mse, formula, atol=1e-14, rtol=1e-12):
        raise RuntimeError('Bernoulli identity failed')
    if not torch.allclose(mse, bias+variance, atol=1e-14, rtol=1e-12):
        raise RuntimeError('bias variance identity failed')
    return dict(mse=float(mse), bias_squared=float(bias), variance=float(variance))


def status(detail, state='running'):
    print(detail, flush=True)
    save(DATA/'status.json', dict(state=state, detail=detail, pid=os.getpid(),
         updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))


def main():
    if OUT.exists():
        raise RuntimeError('refuse to overwrite published output')
    free_gpu()
    started = time.monotonic()
    source = sha(Path(__file__))
    protocol = sha(ROOT/'src/experiments/RANDOMIZED_CORRECTION_DEVELOPMENT.md')
    rows, inputs, artifacts = [], [], []
    previous = json.loads((ROOT/'src/experiments/change_mixed_development/decision.json').read_text())
    old_cost = list(csv.DictReader((ROOT/'src/experiments/change_mixed_development/costs.csv').open()))
    old_inputs = {a['file']: a['sha256'] for a in previous['input_artifacts']
                  if a['root'] == 'postwo_rounding'}
    for prompt in (0, 4):
        serial_file = Path.home()/f'sfq_data/change_mixed/p{prompt:02d}_uniform3_packed.pt'
        expected = next(r['state_sha256'] for r in old_cost
                        if int(r['prompt']) == prompt and r['arm'] == 'uniform3')
        if sha(serial_file) != expected:
            raise RuntimeError('packed input changed')
        inputs.append(dict(file=f'change_mixed/{serial_file.name}', sha256=expected))
        serial = torch.load(serial_file, weights_only=True, map_location='cuda')
        for layer in (0, 14, 29):
            status(f'p{prompt} layer{layer}: frozen input and exact risk')
            captured = []
            for chunk in (6, 7):
                rel = f'p{prompt:02d}_s101/l{layer:02d}_c{chunk}.pt'
                path = Path.home()/'sfq_data/postwo_rounding'/rel
                digest = sha(path)
                if digest != old_inputs[rel]:
                    raise RuntimeError('teacher input changed')
                inputs.append(dict(file=f'postwo_rounding/{rel}', sha256=digest))
                captured.append(torch.load(path, weights_only=True, map_location='cuda'))
            current, later = captured
            state = PackedMixed(**serial[str(layer)]['v'])
            error = (state.restore()[0]-current['new_value'][0].float()).permute(1,0,2)
            residuals = []
            for query in (current['query'], later['query']):
                a0, a1 = probabilities(query, current['key'])
                residuals.append(torch.stack([_project(a@error, current['wo'].float(), False)
                                             for a in (a0, a1)]).cpu())
            train, evaluation = residuals
            path = DATA/f'p{prompt:02d}_l{layer:02d}_errors.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(train=train, evaluation=evaluation), path)
            artifacts.append(dict(file=path.name, sha256=sha(path)))
            base = float(evaluation.double().square().mean())
            for budget in (.1, .25, .5, .75):
                policies = dict(uniform=torch.full((32,), budget, dtype=torch.float64),
                    causal_proxy=allocation(train.double().square().mean((0,2)).sqrt(), budget),
                    oracle=allocation(evaluation.double().square().mean((0,2)).sqrt(), budget))
                for policy, p in policies.items():
                    for arm, gamma in (('unbiased', torch.ones_like(p)),
                                       ('switch', p), ('gain_cap2', torch.minimum(2*p, torch.ones_like(p)))):
                        values = risk(evaluation, p, gamma)
                        rows.append(dict(prompt=prompt, layer=layer, budget=budget,
                            allocation=policy, arm=arm, baseline_mse=base,
                            mse_over_baseline=values['mse']/base, **values,
                            mean_p=float(p.mean()), min_p=float(p.min()), max_p=float(p.max()),
                            high_precision_storage_saved=False, scope='new-V-only post-Wo fixed teacher query'))
            del current, later, captured, error, state
        del serial
        torch.cuda.empty_cache()
    if len(rows) != 216 or source != sha(Path(__file__)):
        raise RuntimeError('incomplete run or source changed')
    OUT.mkdir(parents=True)
    with (OUT/'rows.csv').open('w') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    save(OUT/'evidence.json', dict(source_sha256=source, protocol_sha256=protocol,
         inputs=inputs, artifacts=artifacts, rows=len(rows), seconds=time.monotonic()-started,
         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
         scope='analytic local risk, not full cache attention, video, measured savings or novelty proof'))
    status('COMPLETE: 216 exact risk rows; analysis and report follow', 'pass')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    try:
        main()
    except Exception as exc:
        status(str(exc), 'failed')
        raise
