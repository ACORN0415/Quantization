"""Reproducible, resumable full-model gates. Run with the isolated interpreter.

Requires completed downloads and local pristine/patched reference checkouts.
Outputs and large latents stay in DATA/verify; only docs/VERIFY.md is tracked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--data', type=Path, default=Path.home()/'sfq_data')
ap.add_argument('--history', type=Path, default=Path.home()/'Quantization')
ap.add_argument('--gates', nargs='+', default=['G1','G2','G3','G4','G5'])
a = ap.parse_args()
DATA, HISTORY = a.data.resolve(), a.history.resolve()
OUT = DATA/'verify'
OUT.mkdir(exist_ok=True)
PY = str(DATA/'py')
state_path = OUT/'status.json'
status = json.loads(state_path.read_text()) if state_path.exists() else {}
for gate in ('G1','G1b','G2','G3','G4','G5'):
    status.setdefault(gate, {'state':'PENDING'})



def save():
    state_path.write_text(json.dumps(status, indent=2))
    lines = ['# Verification results', '', 'Live record; PASS requires completed checks. See `docs/HANDOFF_codex.md` for fixed criteria.', '',
             '| Gate | State | Evidence |', '|---|---|---|']
    for gate, rec in status.items():
        lines.append(f"| {gate} | {rec['state']} | {rec.get('detail','')} |")
    lines += ['', f'Logs, exact commands and machine-readable reports: `{OUT}`.',
              '', 'The signed-int8 KIVI 8-bit compatibility test reproduces historical wraparound;',
              '`--unsigned_codes` uses uint8 to avoid that behavior.', '']
    (ROOT/'docs/VERIFY.md').write_text('\n'.join(lines))


def command(label, argv, cwd=ROOT):
    # Cache completed runs only when command and implementation sources match.
    source = hashlib.sha256()
    for folder in ('kvq','pipeline','wan','utils','configs'):
        for f in sorted((ROOT/folder).rglob('*')):
            if f.is_file() and f.suffix in ('.py','.yaml'):
                source.update(f.read_bytes())
    if Path(cwd) != ROOT:
        for f in sorted(Path(cwd).rglob('*.py')):
            source.update(f.read_bytes())
    source.update((ROOT/'inference.py').read_bytes())
    source.update(json.dumps(argv).encode())
    # Inputs and the called checker matter even when their filenames stay fixed.
    for index, arg in enumerate(argv):
        candidate = Path(arg)
        if candidate.is_file() and candidate.suffix == '.py':
            source.update(candidate.read_bytes())
        if arg in ('--data_path', '--config_path'):
            source.update(Path(argv[index + 1]).read_bytes())
    if label.startswith('g1_'):
        for test in sorted((ROOT/'tests').glob('test_*.py')):
            source.update(test.read_bytes())
    environment = dict(os.environ, PYTHONUNBUFFERED='1',
                       KVQ_REF_REPO=str(DATA/'kv-reference'),
                       SF_REF_REPO=str(DATA/'sf-pristine'),
                       LEGACY_REF_REPO=str(DATA/'sf-legacy'))
    if label == 'g1_cuda': environment['TEST_DEVICE'] = 'cuda'
    else: environment['TEST_DEVICE'] = 'cpu'

    digest = source.hexdigest()
    marker = OUT/f'{label}.done.json'
    if marker.exists() and json.loads(marker.read_text()).get('digest') == digest:
        print(f'REUSE {label}', flush=True)
        return
    log = OUT/f'{label}.log'
    print(f'RUN {label}: {json.dumps(argv)}', flush=True)
    started = time.time()
    with log.open('w') as stream:
        result = subprocess.run(argv, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    record = {'argv': argv, 'cwd': str(cwd), 'exit_code': result.returncode,
              'seconds': time.time()-started, 'digest': digest}
    (OUT/f'{label}.command.json').write_text(json.dumps(record, indent=2))
    if result.returncode:
        raise RuntimeError(f'{label} failed (exit {result.returncode}); see {log}')
    marker.write_text(json.dumps(record, indent=2))
    print(f'DONE {label} {record["seconds"]:.1f}s', flush=True)


def own(label, *, frames=63, rng='chunk', method=None, bits=4, prompts='prompts2.txt', drain=False, storage='rolling'):
    args = [PY, str(ROOT/'inference.py'), '--config_path',
            str(ROOT/('configs/self_forcing_dmd.yaml' if frames == 21 else 'configs/a1.yaml')),
            '--checkpoint_path', str(DATA/'ckpt/checkpoints/self_forcing_dmd.pt'),
            '--data_path', str(OUT/prompts), '--output_folder', str(OUT/label/'videos'),
            '--num_output_frames', str(frames), '--use_ema', '--seed', '0', '--save_with_index',
            '--rng', rng, '--latent_dump_root', str(OUT/label/'latents'), '--no_video',
            '--stats_path', str(OUT/label/'stats.json')]
    if method:
        args += ['--kv_method', method, '--kv_bits', str(bits), '--storage', storage,
                 '--kv_error_log', str(OUT/label/'raw_errors.json')]
    if drain:
        args += ['--drain_rng']
    command(label, args)


def legacy(label, method=None, bits=4):
    args = [PY, str(DATA/'sf-legacy/inference.py'), '--config_path', str(ROOT/'configs/a1.yaml'),
            '--checkpoint_path', str(DATA/'ckpt/checkpoints/self_forcing_dmd.pt'),
            '--data_path', str(OUT/'prompts2.txt'), '--output_folder', str(OUT/label/'videos'),
            '--num_output_frames', '63', '--use_ema', '--seed', '0', '--save_with_index',
            '--latent_dump_root', str(OUT/label/'latents'), '--no_decode',
            '--kv_quant_repo', str(DATA/'kv-reference')]
    if method:
        args += ['--kv_quant', method.upper(), '--kv_bits', str(bits)]
    command(label, args, cwd=DATA/'sf-legacy')


def compare(label, ref, got):
    command(label, [PY, str(ROOT/'scripts/compare_latents.py'), str(OUT/ref/'latents'),
                    str(OUT/got/'latents'), '--report', str(OUT/f'{label}.json')])


prompts = (HISTORY/'docs/prompts10.txt').read_text().splitlines()
for n in (2,3,10):
    (OUT/f'prompts{n}.txt').write_text('\n'.join(prompts[:n])+'\n')
# Fail before constructing models if weights are incomplete.
logs = [DATA/'dl.log', DATA/'download_resume.log']
if not any(p.exists() and 'DL_DONE' in p.read_text(errors='replace') for p in logs):
    raise SystemExit('Downloads not confirmed complete: DL_DONE is required. No full-model gate started.')
checkpoint = DATA/'ckpt/checkpoints/self_forcing_dmd.pt'
with checkpoint.open('rb') as f:
    md5 = hashlib.file_digest(f, 'md5').hexdigest() if hasattr(hashlib,'file_digest') else None
if md5 is None:
    h = hashlib.md5()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda: f.read(16*1024*1024), b''): h.update(chunk)
    md5 = h.hexdigest()
if md5 != 'ef41ae42bfb87abe43d5f640fb634d0c':
    raise SystemExit(f'Checkpoint MD5 mismatch: {md5}')
(OUT/'checkpoint.json').write_text(json.dumps({'md5':md5,'bytes':checkpoint.stat().st_size},indent=2))

for gate in a.gates:
    status[gate] = {'state':'RUNNING'}
    save()
    try:
        if gate == 'G1':
            command('g1_cpu', [PY, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
            command('g1_cuda', [PY, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_quant.py', '-v'])
            status['G1b'] = {'state':'PASS', 'detail':'Store and attention integration assertions passed in g1_cpu.log'}
            detail = 'CPU suite and CUDA numerical reference suite passed; g1_cpu.log and g1_cuda.log'
        elif gate == 'G2':
            label = 'g2_pristine'
            command(label, [PY, str(ROOT/'scripts/reference_run.py'), str(DATA/'sf-pristine'),
                            str(OUT/label/'latents'), '--', '--config_path', str(ROOT/'configs/self_forcing_dmd.yaml'),
                            '--checkpoint_path', str(checkpoint), '--data_path', str(OUT/'prompts3.txt'),
                            '--output_folder', str(OUT/label/'videos'), '--num_output_frames','21',
                            '--use_ema','--seed','0','--save_with_index'])
            own('g2_ours', frames=21, rng='release', prompts='prompts3.txt')
            compare('g2_comparison', label, 'g2_ours')
            detail = '3/3 final latents bit-exact; g2_comparison.json'
        elif gate == 'G3':
            own('g3_base')
            own('g3_drain', drain=True)
            compare('g3_comparison','g3_base','g3_drain')
            detail = '2 prompts × 21 chunks; CUDA RNG drained by 7777 draws/chunk; bit-exact'
        elif gate == 'G4':
            for name, method, bits in [('bf16',None,4),('rtn4','rtn',4),('rtn2','rtn',2),('kivi4','kivi',4)]:
                old, new = 'g4_legacy_'+name, 'g4_ours_'+name
                legacy(old, method, bits)
                if method is None:
                    own('g3_base')
                    new = 'g3_base'
                else:
                    own(new, method=method, bits=bits)
                compare('g4_comparison_'+name,old,new)
            detail = '4 settings × 2 prompts × 21 chunks bit-exact; g4_comparison_*.json'
        elif gate == 'G5':
            own('g5_ours', method='kivi', prompts='prompts10.txt')
            command('g5_comparison', [PY, str(ROOT/'scripts/legacy_errors.py'),
                                     str(OUT/'g5_ours/raw_errors.json'),
                                     str(HISTORY/'results/kivi_rw/kivi4_rw0_kv_err.json'),
                                     '--output', str(OUT/'g5_comparison.json')])
            detail = '210 historical rows within relative 1e-6; g5_comparison.json'
        elif gate == 'CHUNK':
            own('chunk_ours', method='rtn', storage='chunk')
            reference = OUT/'g4_ours_rtn4/latents'
            candidate = OUT/'chunk_ours/latents'
            equal = []
            for p in sorted(reference.glob('prompt*/chunk_*.pt')):
                import torch
                q = candidate/p.relative_to(reference)
                equal.append(torch.equal(torch.load(p,weights_only=True)['latent'],torch.load(q,weights_only=True)['latent']))
            if not equal or all(equal): raise RuntimeError('chunk path must differ from rolling on this test')
            detail = f'Chunk-policy full rollout completed; differs from rolling in {sum(not x for x in equal)}/{len(equal)} chunks'
        else:
            raise ValueError(gate)
        status[gate] = {'state':'PASS','detail':detail}
    except Exception as error:
        status[gate] = {'state':'FAIL','detail':str(error)}
        save()
        raise
    save()
