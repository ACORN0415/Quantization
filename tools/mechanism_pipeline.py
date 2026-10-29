"""Detached coordinator. No held-out access unless all prerequisites pass."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DATA=Path.home()/'sfq_data/mechanism'


def run(*args):
    print('RUN',*map(str,args),flush=True)
    subprocess.run(list(map(str,args)),cwd=ROOT,check=True)


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    for mode in ('gates','diagnose','decision'):
        if mode=='gates' and (DATA/'gates.json').exists():
            sys.path.insert(0,str(ROOT/'tools'));import mechanism_run as d
            gate=d.read(DATA/'gates.json')
            if gate['passed'] and gate['source_sha256']==d.fingerprint():continue
        run(sys.executable,ROOT/'tools/mechanism_run.py','--mode',mode)
    result=json.loads((DATA/'decision.json').read_text())
    if result['advance']:
        run(sys.executable,ROOT/'tools/mechanism_followup.py')
    paths=['src/experiments/mechanism_diagnostic_results']
    if (ROOT/'src/experiments/mechanism_fr_results').exists():paths.append('src/experiments/mechanism_fr_results')
    run('git','add','--',*paths)
    if subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=ROOT).returncode:
        run('git','commit','-m','Publish fixed mechanism decision and all authorized follow-up results','--',*paths)
    run('git','push','origin','main')
    p=DATA/'status.json';v=json.loads(p.read_text())
    v.update(state='pass',detail='고정된 메커니즘 계획 완료: '+('GO 후속 단계 포함' if result['advance'] else 'NO-GO; 후속 생성 없음'),
             updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
    print('MECHANISM PIPELINE COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        DATA.mkdir(parents=True,exist_ok=True);p=DATA/'status.json';v=json.loads(p.read_text()) if p.exists() else {}
        v.update(state='failed',detail=f'Coordinator: {type(exc).__name__}: {exc}',pid=os.getpid())
        p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');raise
