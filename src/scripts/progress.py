"""Read-only CLI status for reconstruction review and experiment jobs."""
import argparse,json,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path.home()/'sfq_data')
ap.add_argument('--watch',action='store_true');ap.add_argument('--interval',type=int,default=5)
a=ap.parse_args()
def show():
    if a.watch:print('\033[2J\033[H',end='')
    print(time.strftime('%Y-%m-%d %H:%M:%S'),'— Quantization progress')
    print('Local data:',a.data)
    for name in ('WORK_STATUS.md','review_tests_final.log','review_identity.log','p3_gate_final.log','p3_queue.log'):
        path=a.data/name
        if not path.exists():continue
        print('\n'+name+' ['+time.strftime('%H:%M:%S',time.localtime(path.stat().st_mtime))+']')
        lines=path.read_text(errors='replace').replace('\r','\n').splitlines()
        print('\n'.join(lines[-8:]))
    for pattern in ('verify/identity_*.log','p3_pilot/*.log'):
        for path in sorted(a.data.glob(pattern)):
            print('\n'+str(path.relative_to(a.data)))
            print('\n'.join(path.read_text(errors='replace').replace('\r','\n').splitlines()[-4:]))
    for relative in ('verify/status.json','p3_pilot/queue.json'):
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
