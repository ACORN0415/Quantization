"""GO-only performance gates, registered held-out generation and publication."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/scripts')]
import torch
import mechanism_run as d
import mechanism_worker as worker

DATA=Path.home()/'sfq_data/mechanism'


def performance_record(s,arm,group,rep):
    if (s['arm'],s['group'],str(s['rep']))!=(arm,group,str(rep)):
        raise RuntimeError('performance metadata identifies a different run')
    return dict(s)


def run(*args):
    print('RUN',*map(str,args),flush=True)
    subprocess.run(list(map(str,args)),cwd=ROOT,check=True)


def launch(arm,group=16,rep='0',kind='development',prompt=0,seed=101,warmup=False,observe=False,profile=False,quality=False):
    args=[sys.executable,ROOT/'tools/mechanism_worker.py','--task','quality' if quality else 'generate',
          '--kind',kind,'--arm',arm,'--group',str(group),'--rep',str(rep),'--prompt',str(prompt),'--seed',str(seed)]
    if warmup:args.append('--warmup')
    if observe:args.append('--observe')
    if profile:args.append('--profile')
    folder=DATA/'fr'/f'p{prompt:02d}_s{seed}_{arm}' if kind=='heldout' else DATA/'performance'/f'{arm}_g{group}_r{rep}'
    for attempt in range(3):
        started=time.time()
        try:run(*args);return
        except subprocess.CalledProcessError:
            failure=folder/'last_failure.json'
            if not failure.exists():raise
            record=d.read(failure)
            d.save(folder/f'{"quality" if quality else "generate"}_failure_attempt{attempt+1}.json',record)
            if record['wall']<started or not record['infrastructure'] or attempt==2:raise


def main():
    decision=d.read(DATA/'decision.json')
    if not decision['advance'] or decision['source_sha256']!=d.fingerprint():raise RuntimeError('follow-up requires matching GO')
    source=worker.source_hash()
    # C-repeat has no archived FR counterpart: verify stats do not affect it.
    for observe,rep in [(False,'identity_off'),(True,'identity_on')]:
        launch('C_repeat',rep=rep,observe=observe)
    a=DATA/'performance/C_repeat_g16_ridentity_off';b=DATA/'performance/C_repeat_g16_ridentity_on'
    x=torch.load(a/'latent.pt',map_location='cpu',weights_only=True)['latent'];y=torch.load(b/'latent.pt',map_location='cpu',weights_only=True)['latent']
    if not torch.equal(x,y):raise RuntimeError('diagnostic ON/OFF output mismatch')
    del x,y
    base=[('BF16',16),('L',16),('C_repeat',16),('C_once',16),('L',24)]
    performance=[]
    for rep,rotation in enumerate((0,2,4)):
        order=base[rotation:]+base[:rotation]
        for arm,group in order:
            launch(arm,group,str(rep),warmup=True)
            folder=DATA/'performance'/f'{arm}_g{group}_r{rep}'
            s=d.read(folder/'sampling.json');performance.append(performance_record(s,arm,group,rep))
            if arm in ('BF16','L','C_once') and group==16:
                old=Path.home()/'sfq_data/policy_study'/('reference/p00_s101/bf16.pt' if arm=='BF16' else f'cases/p00_s101_{"rolling" if arm=="L" else "chunk"}_int4/fr.pt')
                latent=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True)['latent']
                ref=torch.load(old,map_location='cpu',weights_only=True)['latent']
                if not torch.equal(latent,ref):raise RuntimeError('performance output differs from archived diagnostics')
                del latent,ref
            if rep==0:launch(arm,group,str(rep),quality=True)
    profiles=[]
    for arm,group in base:
        launch(arm,group,'profile',warmup=True,profile=True)
        folder=DATA/'performance'/f'{arm}_g{group}_rprofile'
        meta=d.read(folder/'sampling.json');ref=d.read(DATA/'performance'/f'{arm}_g{group}_r0'/'sampling.json')
        if meta['latent_sha256']!=ref['latent_sha256']:raise RuntimeError('profiling changed output')
        profiles.append(meta)
    if source!=worker.source_hash():raise RuntimeError('follow-up source changed before registration')
    d.save(DATA/'followup_gates.json',dict(passed=True,source_sha256=source,diagnostic_go_sha256=worker.sha(DATA/'decision.json'),
        on_off_bitexact=True,performance=performance,profiles=profiles,
        note='15 benchmark trajectories, 5 separate profile trajectories, two ON/OFF identity trajectories; all development only'))
    results=ROOT/'src/experiments/mechanism_diagnostic_results'
    d.save(results/'performance.json',d.read(DATA/'followup_gates.json'))
    seal=dict(id='group_rule_fr_v0_3',source_sha256=source,diagnostic_sha256=d.fingerprint(),
              development_gates_passed=True,gate_sha256=worker.sha(DATA/'followup_gates.json'),
              prompt_sha256=worker.sha(ROOT/'src/experiments/mechanism_prompts_v0.txt'),
              planned_trajectories=72,primary='C-repeat16 minus L16; mean raw MUSIQ frames 249..500',
              registered_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),held_out_measurements_started=False)
    path=ROOT/'src/experiments/mechanism_fr_registration.json'
    if path.exists():
        old=d.read(path)
        for key in ('source_sha256','gate_sha256','prompt_sha256'):
            if old[key]!=seal[key]:raise RuntimeError('existing FR registration differs')
    else:d.save(path,seal)
    run('git','add','--',path.relative_to(ROOT))
    if subprocess.run(['git','diff','--cached','--quiet','--',str(path.relative_to(ROOT))],cwd=ROOT).returncode:
        run('git','commit','-m','Seal conditional group-rule FR study after diagnostic GO and development gates','--',path.relative_to(ROOT))
    run('git','push','origin','main')
    completed=0
    for prompt in range(12):
        for seed in (303,404):
            for arm in ('BF16','L','C_repeat'):
                if worker.source_hash()!=source:raise RuntimeError('registered implementation changed')
                d.report(f'HELDOUT {prompt} seed{seed} {arm}',completed=completed,total=72)
                launch(arm,kind='heldout',prompt=prompt,seed=seed)
                launch(arm,kind='heldout',prompt=prompt,seed=seed,quality=True)
                completed+=1
                d.report('HELDOUT trajectory complete',completed=completed,total=72)
    run(sys.executable,ROOT/'tools/analyze_mechanism_fr.py')


if __name__=='__main__':main()
