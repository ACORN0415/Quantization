"""Local coordinator: passed gates -> committed registration -> study -> publish.

No credentials are stored. Use the machine's existing Git authentication helper
when available, otherwise ordinary git. Model code/plan hashes gate every stage.
"""
import argparse
import csv
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


def finalize_artifacts(data,results):
    # Live status changes during publication and is not research evidence.
    # Exclude it rather than publish a hash that becomes stale immediately.
    path=results/'evidence.json';value=json.loads(path.read_text())
    value['artifacts']=[r for r in value['artifacts'] if r['path']!='status.json']
    value['artifact_note']='Live status.json is mutable and excluded from the research manifest.'
    costs=[]
    for folder in sorted((data/'cases').iterdir()):
        if not (folder/'complete.json').exists():continue
        c=json.loads((folder/'complete.json').read_text())
        stages={a:json.loads((folder/f'{a}_sampling.json').read_text()) for a in ('tf','fr','noise')}
        row=dict(case=c['case'],seconds=c['seconds'],policy=c['policy'],bits=c['bits'],
                          peak_allocated_gib=max(s['peak_allocated_gib'] for s in stages.values()),
                          max_noise_norm_relative_mismatch=c['max_noise_norm_relative_mismatch'],
                          noise_nonfinite=c['noise_nonfinite'])
        for arm in ('tf','fr','noise'):
            q=json.loads((folder/f'{arm}_quality.json').read_text())
            row[arm+'_motion_absdiff_mean']=q.get('motion_absdiff_mean')
            row[arm+'_resident_state_bytes']=stages[arm]['cache_state']['resident_tensor_bytes']
            row[arm+'_hypothetical_packed_bytes']=stages[arm]['cache_state']['hypothetical_packed_bytes']
            row[arm+'_diagnostic_source_bytes']=stages[arm]['cache_state']['diagnostic_source_bytes']
        costs.append(row)
    value['case_costs']=costs
    path.write_text(json.dumps(value,indent=2)+'\n')
    with (results/'case_costs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(costs[0]));writer.writeheader();writer.writerows(costs)
    with (results/'RESULTS.md').open('a') as f:
        f.write('\nExecution and whole-video motion diagnostics: [case_costs.csv](case_costs.csv). '
                'Resident and hypothetical packed state bytes exclude diagnostic originals, which are listed separately. '
                'Sampling peaks include model/reconstruction states and are not packed-inference savings.\n')


def push_with_retry(command,data):
    for attempt in range(60):
        try:
            # Native Git can use the memory credential cache after IDE logout.
            native=subprocess.run(['git','push','origin','main'],cwd=REPO,
                env=dict(os.environ,GIT_TERMINAL_PROMPT='0'))
            if native.returncode==0:return
            run(command);return
        except subprocess.CalledProcessError:
            if attempt==59:raise
            path=data/'status.json';value=json.loads(path.read_text())
            value['publication']=dict(state='retrying',detail=f'GitHub push retry {attempt+1}/60; local work retained')
            path.write_text(json.dumps(value,indent=2)+'\n')
            time.sleep(60)


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
    # The gate writes its result before Python tears down GPU model state.
    # Wait for that process to exit before loading the study model.
    while True:
        proc=Path(f'/proc/{args.gate_pid}/cmdline')
        if not proc.exists() or b'policy_study.py' not in proc.read_bytes():break
        time.sleep(1)
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
    push_with_retry(push,data)
    env=dict(os.environ,TZ='Asia/Seoul',PYTHONUNBUFFERED='1')
    with (data.parent/'policy_study.log').open('a') as log:
        run([sys.executable,SRC/'scripts/policy_study.py','--mode','study','--data',data],
            stdout=log,stderr=subprocess.STDOUT,env=env)
    if study.source_hash()!=expected:raise RuntimeError('source changed during measurement/analysis; publication disabled')
    results=SRC/'experiments/policy_study_results'
    if not (results/'RESULTS.md').exists():raise RuntimeError('final analysis missing')
    finalize_artifacts(data,results)
    readme=REPO/'README.md';s=readme.read_text()
    link='> Completed registered P1/P3 study: [results and limits](src/experiments/policy_study_results/RESULTS.md).\n'
    if link not in s:readme.write_text(link+s)
    paths=['README.md',str(results.relative_to(REPO))]
    run(['git','add','--',*paths])
    run(['git','commit','-m','Publish complete registered P1 P3 study and prompt-paired analysis','--',*paths])
    push_with_retry(push,data)
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


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('PIPELINE FAILED',type(exc).__name__,str(exc),flush=True)
        raise
