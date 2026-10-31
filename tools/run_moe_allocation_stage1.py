"""Sequential, isolated subprocess runner with a shared absolute deadline."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

REPO=Path(__file__).resolve().parents[1]
RECORD=REPO/'src/experiments/moe_allocation_stage1'
DATA=Path('/home/acorn0415/sfq_data/moe_allocation_stage1')
deadline=json.loads((RECORD/'run_state.json').read_text())['deadline_epoch']
env=os.environ.copy()
env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
           OPENBLAS_NUM_THREADS='4',HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false',
           HF_HUB_OFFLINE='0',HF_XET_CHUNK_CACHE_SIZE_BYTES='0')


def run(name,arguments):
    command=[sys.executable,str(REPO/'tools/moe_allocation_stage1.py'),*arguments]
    start=time.perf_counter()
    with (RECORD/f'{name}.log').open('w') as log:
        process=subprocess.Popen(command,env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        done=threading.Event()
        cap_reasons=[]
        def watch():
            while not done.wait(0.5):
                reason=None
                if time.time()>=deadline:
                    reason='Registered absolute wall-time cap reached'
                disk=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())
                if disk>20*2**30:
                    reason='Registered incremental disk cap reached'
                if reason:
                    cap_reasons.append(reason)
                    process.terminate()
                    if not done.wait(5): process.kill()
                    return
        monitor=threading.Thread(target=watch,daemon=True)
        monitor.start()
        process.wait()
        elapsed=time.perf_counter()-start
        done.set()
        monitor.join(timeout=1)
        if cap_reasons: raise RuntimeError(cap_reasons[0])
    result=dict(name=name,pid=process.pid,exit_code=process.returncode,
                complete_process_seconds=elapsed,arguments=arguments)
    (RECORD/f'{name}_process.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if process.returncode:
        raise RuntimeError(f'{name} failed; see {name}.log')
    return result


if __name__=='__main__':
    phase=sys.argv[1]
    if phase=='prepare':
        run('prepare',['prepare'])
    elif phase=='diagnose':
        run('diagnose',['diagnose'])
    elif phase=='cost':
        for method in ('B','C'):
            run(f'warmup_{method}',['cost','--method',method,'--run-id',f'warmup_{method}'])
        for pair,order in enumerate((('B','C'),('C','B'),('B','C')),1):
            for method in order:
                name=f'pair{pair}_{method}'
                run(name,['cost','--method',method,'--run-id',name])
    else:
        raise ValueError(phase)
