"""Read-only CLI status for reconstruction review and experiment jobs."""
import argparse,json,os,subprocess,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path.home()/'sfq_data')
ap.add_argument('--watch',action='store_true');ap.add_argument('--interval',type=int,default=5)
ap.add_argument('--history',action='store_true',help='also show archived reconstruction logs')
a=ap.parse_args()
def show():
    if a.watch:print('\033[2J\033[H',end='')
    print(time.strftime('%Y-%m-%d %H:%M:%S'),'— Quantization progress')
    print('Local data:',a.data)
    current=a.data/'policy_study/status.json'
    if current.exists() and not a.history:
        try:
            for label,value in json.loads(current.read_text()).items():
                print('\n'+label.upper(),value.get('state','?'),value.get('detail',''))
                print('Updated UTC:',value.get('updated_utc','?'))
                if 'completed_cases' in value:
                    print('Cases:',value['completed_cases'],'/',value.get('total_cases',80),
                          'remaining hours (rough):',value.get('eta_hours_rough','not measured yet'))
                if value.get('state')=='running' and value.get('pid'):
                    try:os.kill(value['pid'],0);print('Process:',value['pid'],'alive')
                    except ProcessLookupError:print('ATTENTION: recorded process has stopped; inspect failure log')
        except json.JSONDecodeError:print('Status updating…')
        try:
            gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                                '--format=csv,noheader'],capture_output=True,text=True,timeout=2)
            if gpu.returncode==0:print('GPU utilization / used / total:',gpu.stdout.strip())
        except (OSError,subprocess.TimeoutExpired):pass
        for name in ('WORK_STATUS.md','policy_study.log','policy_study_pipeline.log'):
            path=a.data/name
            if path.exists():
                print('\n'+name)
                print('\n'.join(path.read_text(errors='replace').replace('\r','\n').splitlines()[-10:]))
        return
    for name in ('WORK_STATUS.md','policy_study.log','policy_study_pipeline.log','policy_study_tests.log','feasibility.log','feasibility_tests.log','anchor_validity.log','latent_recovery.log','validity_followup_tests.log','review_tests_final.log','review_identity.log','p3_gate_final.log','p3_queue.log'):
        path=a.data/name
        if not path.exists():continue
        print('\n'+name+' ['+time.strftime('%H:%M:%S',time.localtime(path.stat().st_mtime))+']')
        lines=path.read_text(errors='replace').replace('\r','\n').splitlines()
        print('\n'.join(lines[-8:]))
    for pattern in ('verify/identity_*.log','p3_pilot/*.log'):
        for path in sorted(a.data.glob(pattern)):
            print('\n'+str(path.relative_to(a.data)))
            print('\n'.join(path.read_text(errors='replace').replace('\r','\n').splitlines()[-4:]))
    for relative in ('policy_study/status.json','feasibility/status.json','verify/status.json','latent_recovery/status.json','p3_pilot/queue.json'):
        path=a.data/relative
        if not path.exists():continue
        print('\n'+relative)
        try:
            for key,value in json.loads(path.read_text()).items():
                print(' ',key,':',value.get('state','?'),value.get('detail',''))
        except json.JSONDecodeError:print('  Updating…')
show()
while a.watch:
    time.sleep(max(1,min(a.interval,60)));show()
