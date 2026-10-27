"""Local coordinator: passed gates -> committed registration -> study -> publish.

No credentials are stored. Use the machine's existing Git authentication helper
when available, otherwise ordinary git. Model code/plan hashes gate every stage.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
SRC=REPO/'src'


def run(command,**kwargs):
    print('RUN',*map(str,command),flush=True)
    subprocess.run(list(map(str,command)),cwd=REPO,check=True,**kwargs)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gate-pid',type=int,required=True)
    ap.add_argument('--data',type=Path,default=Path.home()/'sfq_data/policy_study')
    args=ap.parse_args();data=args.data
    # Coordinator runs under the same isolated interpreter as model execution.
    sys.path.insert(0,str(SRC/'scripts'));study=importlib.import_module('policy_study')
    expected=study.source_hash();gate_path=data/'gates.json'
    while not gate_path.exists():
        proc=Path(f'/proc/{args.gate_pid}/cmdline')
        if not proc.exists() or b'policy_study.py' not in proc.read_bytes():
            raise RuntimeError('gate process stopped without passing prerequisites; inspect policy_study.log')
        time.sleep(5)
    gates=json.loads(gate_path.read_text())
    if not gates.get('passed') or gates['source_sha256']!=expected or study.source_hash()!=expected:
        raise RuntimeError('gate source mismatch; held-out measurements remain disabled')
    if subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()!='main':
        raise RuntimeError('publication coordinator requires the authorized main branch')
    registration=SRC/'experiments/policy_study_registration.json'
    value=dict(id=study.PLAN['id'],source_sha256=expected,plan_sha256=study.file_hash(study.PLAN_PATH),
               prompt_sha256=study.file_hash(study.PROMPTS_PATH),gate_sha256=study.file_hash(gate_path),
               implementation_commit=gates['git_commit'],
               registered_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               planned_policy_cases=80,held_out_measurements_started=False,
               gates=gates)
    if registration.exists():
        old=json.loads(registration.read_text())
        for key in ('source_sha256','plan_sha256','prompt_sha256','gate_sha256'):
            if old[key]!=value[key]:raise RuntimeError('existing registration differs; amendment required')
        value=old
    study.save_json(registration,value)
    relative=str(registration.relative_to(REPO))
    run(['git','add','--',relative])
    if subprocess.run(['git','diff','--cached','--quiet','--',relative],cwd=REPO).returncode:
        run(['git','commit','-m','Seal main-study preregistration after all development gates pass','--',relative])
    helper=data.parent/'git_with_ide.py'
    push=[sys.executable,helper,'push','origin','main'] if helper.exists() else ['git','push','origin','main']
    run(push)
    env=dict(os.environ,TZ='Asia/Seoul',PYTHONUNBUFFERED='1')
    with (data.parent/'policy_study.log').open('a') as log:
        run([sys.executable,SRC/'scripts/policy_study.py','--mode','study','--data',data],
            stdout=log,stderr=subprocess.STDOUT,env=env)
    results=SRC/'experiments/policy_study_results'
    if not (results/'RESULTS.md').exists():raise RuntimeError('final analysis missing')
    readme=REPO/'README.md';s=readme.read_text()
    link='> Completed registered P1/P3 study: [results and limits](src/experiments/policy_study_results/RESULTS.md).\n'
    if link not in s:readme.write_text(link+s)
    paths=['README.md',str(results.relative_to(REPO))]
    run(['git','add','--',*paths])
    run(['git','commit','-m','Publish complete registered P1 P3 study and prompt-paired analysis','--',*paths])
    run(push)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    value=json.loads((data/'status.json').read_text())
    value['publication']=dict(state='pass',detail='all measurements, analysis, final commit and GitHub push complete',
                              commit=head,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    study.save_json(data/'status.json',value)
    (data.parent/'WORK_STATUS.md').write_text(
        f'COMPLETE: all 80 registered cases, final analysis and GitHub publication.\nCommit: {head}\n'+
        'Results: Quantization/src/experiments/policy_study_results/RESULTS.md\n'+
        'All final tensors and raw metrics retained. No queued GPU work remains.\n')
    print('PIPELINE COMPLETE',head,flush=True)


if __name__=='__main__':main()
