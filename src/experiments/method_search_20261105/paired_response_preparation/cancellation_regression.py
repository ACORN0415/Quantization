"""Synthetic regression after the independent Codex signed-cancellation finding.

The real Gaussian cases use internally consistent Q/K/V/native output/logZ.
Additional hand-built states isolate summation, not attainable model statistics.
"""
import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import torch
from core import ChunkState, PairedResponse

HERE = Path(__file__).resolve().parent


def run():
    torch.set_num_threads(4)
    started = time.monotonic()
    checks = []

    def check(name, passed, **data):
        checks.append(dict(name=name, passed=bool(passed), **data))

    for rank in (1, 32):
        for dtype in (torch.float64, torch.bfloat16):
            model = PairedResponse.gaussian(1, 128, rank, 17)
            center = model.omega[:, :, 0].double() * 128**.25
            delta = torch.zeros_like(center)
            delta[0, 0] = 128**.25
            key = torch.stack((center, center + delta), 1).to(dtype)
            khat = key.flip(1).contiguous()
            vp = torch.tensor([[[1.], [0.]]], dtype=dtype)
            for name, value in (('positive', vp), ('negative', -vp)):
                state, _ = model.summarize(name, key, khat, value)
                model.append(state)
            q = torch.zeros(1, 1, 128, dtype=dtype)
            native_k = torch.cat((khat, khat, torch.zeros_like(khat[:, :1])), 1)
            native_v = torch.cat((vp, -vp, vp.new_tensor([[[5.]]])), 1)
            scores = q.double() @ native_k.double().transpose(-1, -2) / math.sqrt(128)
            base = scores.softmax(-1) @ native_v.double()
            logz = scores.logsumexp(-1)
            plus, minus = model.states.values()
            check(f'gaussian_saved_cancel_r{rank}_{dtype}',
                  torch.equal(plus.dn, -minus.dn) and torch.equal(plus.offset, minus.offset)
                  and not bool(plus.dz.any()) and not bool(minus.dz.any()))
            got = model.read(q, base, logz, min_denominator_ratio=.1)
            check(f'gaussian_native_output_r{rank}_{dtype}',
                  torch.equal(got.output, base) and not bool(got.fallback.any()),
                  expected=base.tolist(), output=got.output.tolist(),
                  logz=logz.tolist(), fallback=got.fallback.tolist())

    # Scalar math.fsum reference: these are finite rounded coefficients of the
    # stored-state expression, not an exact exponential-kernel reference.
    for inside_one_chunk in (False, True):
        for numerator in (False, True):
            for order in itertools.permutations(range(3)):
                rank = 3 if inside_one_chunk else 1
                model = PairedResponse(torch.zeros(1, 1, rank))
                weights = (1., 1., -1.)
                offsets = (math.log(1e20), 0., math.log(1e20))
                if inside_one_chunk:
                    vals = torch.tensor([[weights[i] for i in order]], dtype=torch.float32)
                    z = torch.zeros_like(vals)
                    offset = torch.tensor([[offsets[i] + .5*math.log(rank) for i in order]], dtype=torch.float64)
                    model.append(ChunkState('one', z if numerator else vals,
                                            (vals if numerator else z)[..., None], offset))
                else:
                    for n, i in enumerate(order):
                        value = torch.tensor([[weights[i]]], dtype=torch.float32)
                        zero = torch.zeros_like(value)
                        model.append(ChunkState(str(n), zero if numerator else value,
                                                (value if numerator else zero)[..., None],
                                                torch.tensor([[offsets[i]]], dtype=torch.float64)))
                terms_n, terms_d = [], []
                for state in model.states.values():
                    for j in range(rank):
                        coefficient = math.exp(float(state.offset[0, j]) - .5*math.log(rank))
                        terms_n.append(coefficient * float(state.dn[0, j, 0]))
                        terms_d.append(coefficient * float(state.dz[0, j]))
                expected = math.fsum([1.] + terms_n) / math.fsum([1.] + terms_d)
                got = model.read(torch.zeros(1, 1, 1), torch.ones(1, 1, 1, dtype=torch.float64),
                                 torch.zeros(1, 1), min_denominator_ratio=.1)
                value = float(got.output.item())
                check(f'residual_{inside_one_chunk}_{numerator}_{order}',
                      math.isclose(value, expected, rel_tol=2e-13, abs_tol=2e-13)
                      and not bool(got.fallback.any()),
                      expected=expected, output=value, fallback=got.fallback.tolist())
    check('no_cuda', not torch.cuda.is_initialized())
    return dict(status='passed' if all(x['passed'] for x in checks) else 'failed',
                scope='synthetic numerical regression only; no capture/model/GPU',
                checks=checks, checks_total=len(checks),
                checks_passed=sum(x['passed'] for x in checks),
                wall_seconds=time.monotonic()-started,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (HERE/'core.py', Path(__file__))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    target = Path(parser.parse_args().output).resolve()
    if target.parent != HERE:
        raise ValueError('output must stay in preparation directory')
    if target.exists():
        raise FileExistsError(target)
    result = run()
    with target.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}))
    for item in result['checks']:
        if not item['passed']:
            print(json.dumps(item))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
