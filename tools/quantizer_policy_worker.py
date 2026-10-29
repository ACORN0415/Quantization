"""Isolated GPU jobs for the preregistered development factorial screen."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'src'), str(ROOT/'src/scripts')]
import torch
import mechanism_worker as common
from mechanism_core import RawHistory
from kvq.replay import store_cache
from anchor_identity_gate import IdentityQuantizer
from quantizer_policy_core import make_store, storage_bytes

DATA = Path.home()/'sfq_data/quantizer_policy_cross'
PROTOCOL = ROOT/'src/experiments/QUANTIZER_POLICY_CROSS_DEVELOPMENT.md'
BASE_HASH = common.source_hash
PILOT = common.pilot


def save(path, value):
    common.diagnostic.save(path, value)


def read(path):
    return common.diagnostic.read(path)


def source_hash():
    h = hashlib.sha256(BASE_HASH().encode())
    for path in [PROTOCOL] + [ROOT/'tools'/name for name in
            ('quantizer_policy_core.py', 'quantizer_policy_worker.py',
             'quantizer_policy_study.py', 'test_quantizer_policy_cross.py')]:
        h.update(str(path.relative_to(ROOT)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def report(detail, state='running', **extra):
    value = dict(state=state, detail=detail, pid=os.getpid(),
                 updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
    save(DATA/'status.json', value)
    print(time.strftime('%Y-%m-%d %H:%M:%S'), state, detail, flush=True)


def folder(args):
    return DATA/'development'/f'p{args.prompt:02d}_s{args.seed}_{args.method}_{args.arm}'


def expected_bf16(prompt, seed):
    return torch.load(Path.home()/f'sfq_data/policy_study/reference/p{prompt:02d}_s{seed}/bf16.pt',
                      map_location='cpu', weights_only=True)['latent']


def build(prompt, seed):
    others = common.subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True)
    if any(p.strip().isdigit() and int(p.strip()) != os.getpid() for p in others.splitlines()):
        raise RuntimeError('GPU occupied by another compute process')
    started = time.monotonic()
    pipe = PILOT.build(seed)
    load_s = time.monotonic()-started
    text = (ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines()[prompt]
    started = time.monotonic()
    cond = PILOT.encode(pipe, [text])[0]
    pipe.cross_setup_timing = dict(model_load_s=load_s, text_encode_s=time.monotonic()-started)
    return pipe, cond


def gates():
    from test_quantizer_policy_cross import run
    source = source_hash()
    synthetic = run()
    pipe, cond = build(0, 101)
    expected = expected_bf16(0, 101)
    records = []
    for arm in ('L', 'C_repeat'):
        cache = store_cache([make_store('rtn', arm, identity=IdentityQuantizer()) for _ in range(30)], PILOT.gpu)
        xc = PILOT.cross_cache(pipe, PILOT.gpu)
        noise = PILOT.chunk_noise(pipe, 126, 101, PILOT.gpu)
        for chunk in range(42):
            value = PILOT.run_chunk(pipe, cache, xc, cond, noise, chunk, 101)
            if not torch.equal(value.cpu(), expected[:, 3*chunk:3*chunk+3]):
                raise RuntimeError(f'no-quant identity failed {arm}/{chunk}')
            if chunk % 7 == 0 or chunk == 41:
                report(f'IDENTITY {arm}: {chunk+1}/42 chunks')
        records.append(dict(arm=arm, bitexact_chunks=42))
        del cache, xc, noise, value
        gc.collect(); torch.cuda.empty_cache()
    if source_hash() != source:
        raise RuntimeError('source changed during identity gates')
    save(DATA/'gates.json', dict(passed=True, source_sha256=source, synthetic=synthetic, full_model=records))


class ReplayLayer:
    def __init__(self, layer, rows):
        self.layer, self.rows, self.calls = layer, rows, 0
        self.raw = RawHistory(None, 32760)
        self.stores = {(m, a): make_store(m, a) for m in ('rtn', 'kivi') for a in ('L', 'C_repeat')}

    def __call__(self, k, v, start, *, local=True):
        chunk, call = divmod(self.calls, 5); self.calls += 1
        rk, rv = self.raw.write(k, v, start, local=local)
        self.raw.commit(rk, rv)
        n = self.raw.local_end-k.shape[1]
        refs = (rk[:, :n], rv[:, :n])
        norms = [x.float().norm().clamp_min(1e-12) for x in refs]
        for (method, arm), store in self.stores.items():
            def attend(query, ak, av, method=method, arm=arm, store=store):
                errors = []
                for actual, ref, norm in zip((ak[:, :n], av[:, :n]), refs, norms):
                    error = float((actual.float()-ref.float()).norm()/norm) if n else 0.
                    if not math.isfinite(error):
                        raise RuntimeError('nonfinite replay error')
                    errors.append(error)
                self.rows.append(dict(layer=self.layer, chunk=chunk, call=call, method=method, arm=arm,
                    history_tokens=n, k_rel=errors[0], v_rel=errors[1]))
            store.attend(k, k, v, start, attend, local=local)
            if call == 4:
                self.rows[-1]['storage'] = storage_bytes(store)


def replay(args):
    source = source_hash()
    gate = read(DATA/'gates.json')
    if not gate['passed'] or gate['source_sha256'] != source:
        raise RuntimeError('matching full-model gates required')
    pipe, cond = build(args.prompt, args.seed)
    expected = expected_bf16(args.prompt, args.seed)
    rows = []; layers = (0, 5, 10, 15, 20, 29)
    cache = PILOT.bf16_cache(pipe, PILOT.gpu)
    observers = {layer: ReplayLayer(layer, rows) for layer in layers}
    for layer, observer in observers.items():
        cache[layer]['write_observer'] = observer
    xc = PILOT.cross_cache(pipe, PILOT.gpu)
    noise = PILOT.chunk_noise(pipe, 126, args.seed, PILOT.gpu)
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    for chunk in range(42):
        value = PILOT.run_chunk(pipe, cache, xc, cond, noise, chunk, args.seed)
        if not torch.equal(value.cpu(), expected[:, 3*chunk:3*chunk+3]):
            raise RuntimeError('observed teacher changed from archived BF16')
        if torch.cuda.max_memory_allocated() >= 22*2**30 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 >= 48*2**30:
            raise RuntimeError('replay resource budget exceeded')
        report(f'REPLAY p{args.prompt}: {chunk+1}/42 chunks')
    if any(o.calls != 210 for o in observers.values()) or source != source_hash():
        raise RuntimeError('incomplete or changed replay')
    save(DATA/'replay'/f'p{args.prompt:02d}_s{args.seed}.json', dict(source_sha256=source,
        prompt=args.prompt, seed=args.seed, layers=layers, rows=rows, teacher_bitexact_chunks=42,
        seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        query_probe='not measured: raw-write hook does not expose query'))


def generate(args):
    source = source_hash()
    gate = read(DATA/'gates.json')
    if not gate['passed'] or gate['source_sha256'] != source:
        raise RuntimeError('matching full-model gates required')
    out = folder(args); out.mkdir(parents=True, exist_ok=True)
    if (out/'sampling.json').exists():
        old = read(out/'sampling.json')
        if old['source_sha256'] != source or common.sha(out/'latent.pt') != old['latent_file_sha256']:
            raise RuntimeError('generation resume mismatch')
        return
    common.make = lambda arm, group, observe=False: make_store(args.method, arm, group)
    common.state_bytes = storage_bytes
    monitor = common.Telemetry(); monitor.start()
    started = time.monotonic(); pipe, cond = build(args.prompt, args.seed)
    setup_s = time.monotonic()-started
    report(f'GENERATE p{args.prompt} {args.method}/{args.arm}')
    wall_start = time.time()
    latent, meta = common.execute(pipe, cond, args.arm, 16, args.seed, 126)
    wall_end = time.time()
    cpu = latent.cpu()
    if args.method == 'rtn' and args.prompt == 0:
        old = Path.home()/f'sfq_data/mechanism/performance/{args.arm}_g16_r0/latent.pt'
        ref = torch.load(old, map_location='cpu', weights_only=True)['latent']
        if not torch.equal(cpu, ref):
            raise RuntimeError(f'RTN protected actual trajectory mismatch: {args.arm}')
        meta['protected_rtn_latent_bitexact'] = True
    temp = out/'latent.tmp'; torch.save({'latent': cpu}, temp); temp.replace(out/'latent.pt')
    monitor.stop(); save(out/'telemetry.json', monitor.rows)
    meta.update(source_sha256=source, method=args.method, arm=args.arm, group=16, bits=4,
        prompt=args.prompt, seed=args.seed, kind='cross_development', setup_s=setup_s,
        latent_sha256=common.diagnostic.original.tensor_hash(cpu), latent_file_sha256=common.sha(out/'latent.pt'),
        cpu_lifetime_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        generation_started_wall=wall_start, generation_finished_wall=wall_end)
    meta.update(pipe.cross_setup_timing)
    if source_hash() != source:
        raise RuntimeError('source changed during generation')
    save(out/'sampling.json', meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=('gates', 'replay', 'generate', 'quality'), required=True)
    ap.add_argument('--method', choices=('rtn', 'kivi'), default='rtn')
    ap.add_argument('--arm', choices=('L', 'C_repeat'), default='L')
    ap.add_argument('--prompt', type=int, choices=(0, 4), default=0)
    ap.add_argument('--seed', type=int, choices=(101,), default=101)
    args = ap.parse_args(); args.kind = 'cross_development'; args.group = 16
    os.chdir(ROOT/'src'); torch.set_grad_enabled(False); torch.set_num_threads(4)
    if args.task == 'gates': gates()
    elif args.task == 'replay': replay(args)
    elif args.task == 'generate': generate(args)
    else:
        common.paths = folder
        common.source_hash = source_hash
        common.diagnostic.report = report
        started = time.monotonic()
        common.quality(args)
        save(folder(args)/'quality_resource.json', dict(seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cpu_lifetime_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            source_sha256=source_hash()))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        report(f'{type(exc).__name__}: {exc}', state='failed')
        raise
