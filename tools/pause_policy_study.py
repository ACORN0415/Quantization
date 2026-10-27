"""Administrative pause at a durable case boundary; frozen study code unchanged."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def is_process(pid,script):
    path=Path(f'/proc/{pid}/cmdline')
    try:
        return path.stat().st_uid==os.getuid() and any(
            item.endswith(('/'+script).encode()) or item==script.encode()
            for item in path.read_bytes().split(b'\0'))
    except (FileNotFoundError,ProcessLookupError):return False


def stop_owned(pid,script):
    if is_process(pid,script):os.kill(pid,signal.SIGTERM)


def checkpoint_report(data,target):
    # Descriptive administrative report only. The registered n=10 analysis is
    # deferred until all 80 cases; never perform interim significance testing.
    cases=[read(p) for p in sorted((data/'cases').glob('*/complete.json'))]
    if len(cases)!=target:raise RuntimeError('completed case count differs from pause boundary')
    if any(c['source_sha256']!=read(REPO/'src/experiments/policy_study_registration.json')['source_sha256'] for c in cases):
        raise RuntimeError('checkpoint source mismatch')
    tensors=[];instabilities=0
    for c in cases:
        folder=data/'cases'/c['case'];instabilities+=bool(c.get('noise_nonfinite'))
        for item in read(folder/'latent_manifest.json')['artifacts']:
            path=folder/item['path']
            if not path.exists() or path.stat().st_size!=item['bytes']:
                raise RuntimeError('checkpoint tensor missing or truncated: '+str(path))
            tensors.append(dict(path=str(path.relative_to(data)),**{k:v for k,v in item.items() if k!='path'}))
    retries=len(list((data/'failures').glob('*.json'))) if (data/'failures').exists() else 0
    report=dict(state='paused',updated_utc=utc(),completed=target,planned=80,remaining=80-target,
        noise_instabilities=instabilities,infrastructure_retry_records=retries,
        completed_case_names=[c['case'] for c in cases],retained_case_tensors=tensors,
        source_sha256=cases[0]['source_sha256'],reason='User-requested administrative pause; no sample/endpoint/code changes',
        final_analysis='deferred until all 80 registered cases',resume_command='bash ~/Quantization/tools/resume_policy_study.sh')
    save(data/'pause_checkpoint.json',report)
    out=REPO/'src/experiments/policy_study_midpoint';out.mkdir(exist_ok=True)
    save(out/'checkpoint.json',report)
    lines=['# Administrative midpoint checkpoint','',
        f'{target}/80 policy cases completed; {80-target} remain. User requested a pause at this count before reviewing this checkpoint.',
        'The planned sample, parameters, endpoints and analysis remain unchanged. Frozen experimental source was not edited.',
        f'Completed-case noise instabilities: {instabilities}. Infrastructure retry records: {retries}.',
        f'{len(tensors)} final case tensors retained, with paths/sizes and original hashes in [checkpoint.json](checkpoint.json). BF16 references, videos and raw metrics are also retained outside Git.',
        '','The registered prompt-paired analysis is deferred until all 80 cases. No interim significance test or quality-driven sample change is performed.',
        '','Resume the remaining cases: `bash tools/resume_policy_study.sh`.',
        'Existing completed stages are checked for matching source and tensor hashes before reuse. The original coordinator resumes final analysis and publication after case 80.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


def pause(data,target):
    status=read(data/'status.json')['study'];worker=status['pid']
    supervisor=int((data.parent/'policy_study_pipeline.pid').read_text())
    if not is_process(worker,'policy_study.py'):raise RuntimeError('study worker is not running')
    request=dict(state='armed',target_cases=target,requested_utc=utc(),worker_pid=worker,
                 coordinator_pid=supervisor,reason='User requested a pause and intermediate report at 40 completed cases')
    save(data/'pause_request.json',request)
    libc=ctypes.CDLL(None,use_errno=True)
    libc.inotify_init1.argtypes=[ctypes.c_int];libc.inotify_init1.restype=ctypes.c_int
    libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
    libc.inotify_add_watch.restype=ctypes.c_int
    fd=libc.inotify_init1(os.O_CLOEXEC)
    if fd<0 or libc.inotify_add_watch(fd,os.fsencode(data),0x80)<0:
        raise OSError(ctypes.get_errno(),'cannot watch durable queue boundary')
    print('ARMED',utc(),'target',target,'worker',worker,'coordinator',supervisor,flush=True)
    try:
        while True:
            queue=read(data/'queue.json')
            if queue['complete']>=target:
                os.kill(worker,signal.SIGSTOP)
                break
            if not is_process(worker,'policy_study.py'):raise RuntimeError('worker exited before requested boundary')
            ready,_,_=select.select([fd],[],[],5)
            if ready:
                events=os.read(fd,65536);offset=0;changed=False
                while offset<len(events):
                    _,_,_,length=struct.unpack_from('iIII',events,offset)
                    name=events[offset+16:offset+16+length].split(b'\0')[0]
                    offset+=16+length;changed=changed or name==b'queue.json'
                if not changed:continue
        # Stop the publication coordinator first, so it cannot interpret the
        # intentionally terminated model as a failed full-study publication.
        stop_owned(supervisor,'continue_policy_study.py')
        stop_owned(worker,'policy_study.py');os.kill(worker,signal.SIGCONT)
        deadline=time.monotonic()+30
        while is_process(worker,'policy_study.py'):
            if time.monotonic()>deadline:
                os.kill(worker,signal.SIGKILL);break
            time.sleep(.1)
    finally:os.close(fd)
    queue=read(data/'queue.json')
    if queue['complete']!=target:raise RuntimeError('pause boundary exceeded')
    summary=checkpoint_report(data,target)
    request.update(state='paused',paused_utc=utc(),completed_cases=target,remaining_cases=80-target)
    save(data/'pause_request.json',request)
    status=dict(state='paused',detail=f'User-requested checkpoint: {target}/80 complete; {80-target} remain',
                completed_cases=target,total_cases=80,remaining_cases=80-target,updated_utc=utc(),
                resume_command=summary['resume_command'])
    save(data/'status.json',{'study':status})
    (data.parent/'WORK_STATUS.md').write_text(
        f'PAUSED AT USER REQUEST: {target}/80 completed; {80-target} remain.\n'+
        'All completed tensors, videos, raw metrics and source fingerprints retained.\n'+
        'Final analysis deferred until case 80; experimental source and plan unchanged.\n'+
        'Resume: bash ~/Quantization/tools/resume_policy_study.sh\n'+
        'Report: Quantization/src/experiments/policy_study_midpoint/REPORT.md\n')
    # Commit only the administrative checkpoint; never touch the sealed source.
    paths=['src/experiments/policy_study_midpoint']
    subprocess.run(['git','add','--',*paths],cwd=REPO,check=True)
    subprocess.run(['git','commit','-m',f'Record user-requested pause at {target} of 80 registered cases','--',*paths],cwd=REPO,check=True)
    subprocess.run(['git','push','origin','main'],cwd=REPO,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'),check=True)
    print('PAUSED AND REPORTED',utc(),target,'/80; resume:',summary['resume_command'],flush=True)


def resume(data):
    request=read(data/'pause_request.json')
    if request['state']!='paused':raise RuntimeError('study has not reached its requested pause')
    previous=read(data/'status.json')['study']
    if previous.get('pid') and is_process(previous['pid'],'policy_study.py'):raise RuntimeError('worker already active')
    old=int((data.parent/'policy_study_pipeline.pid').read_text())
    if is_process(old,'continue_policy_study.py'):raise RuntimeError('coordinator already active')
    if read(data/'queue.json')['complete']!=request['completed_cases']:raise RuntimeError('checkpoint queue changed')
    with (data.parent/'policy_study_pipeline.log').open('a') as log:
        process=subprocess.Popen([sys.executable,str(REPO/'tools/continue_policy_study.py'),
            '--gate-pid','0','--data',str(data)],cwd=REPO,start_new_session=True,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,TZ='Asia/Seoul',PYTHONUNBUFFERED='1'))
    (data.parent/'policy_study_pipeline.pid').write_text(str(process.pid)+'\n')
    request.update(state='resumed',resumed_utc=utc(),coordinator_pid=process.pid)
    save(data/'pause_request.json',request)
    print('RESUMED coordinator PID',process.pid,'completed cases will be hash-checked and reused',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['pause-after','resume'])
    ap.add_argument('--cases',type=int,default=40)
    ap.add_argument('--data',type=Path,default=Path.home()/'sfq_data/policy_study')
    args=ap.parse_args()
    if args.action=='pause-after':pause(args.data,args.cases)
    else:resume(args.data)
