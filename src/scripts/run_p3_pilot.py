"""Serial pilot queue; no concurrent GPU models; per-case durable progress."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from p3_tf_fr import source_hash

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path.home()/'sfq_data/p3_pilot')
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
gates=a.out/'gates.json'
if not gates.exists():raise SystemExit('Run p3_tf_fr.py --mode gate first')
g=json.loads(gates.read_text())
if not g['pass'] or g['source_sha256']!=source_hash():raise SystemExit('Current-source gates required')
cleanup=[]
for folder in sorted(a.out.glob('gate_prompt*')):
    for path in sorted(folder.glob('*.pt')):
        cleanup.append({'path':str(path.relative_to(a.out)),'bytes':path.stat().st_size,
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
if cleanup:
    (a.out/'gate_cleanup.json').write_text(json.dumps(cleanup,indent=2))
    for rec in cleanup:(a.out/rec['path']).unlink()
state_file=a.out/'queue.json'
state=json.loads(state_file.read_text()) if state_file.exists() else {}
for bits in (4,2):
    for prompt in (0,1):
        name=f'p{prompt}_s0_int{bits}';folder=a.out/name
        done=folder/'complete.json'
        if done.exists() and json.loads(done.read_text())['source_sha256']==source_hash():
            if not all((folder/f'{arm}.pt').exists() for arm in ('bf16','tf','fr')):
                raise RuntimeError(f'{name}: completed case lacks retained tensors; restore before reuse')
            state[name]={'state':'PASS','reused':True};continue
        argv=[sys.executable,'-u',str(ROOT/'scripts/p3_tf_fr.py'),'--mode','pilot',
              '--out',str(a.out),'--bits',str(bits),'--prompt_index',str(prompt)]
        state[name]={'state':'RUNNING','argv':argv,'source_sha256':source_hash()}
        state_file.write_text(json.dumps(state,indent=2));print('RUN',name,flush=True)
        start=time.time()
        with (a.out/f'{name}.log').open('w') as log:
            result=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state[name].update(state='PASS' if result.returncode==0 else 'FAIL',exit_code=result.returncode,
                           seconds=time.time()-start)
        state_file.write_text(json.dumps(state,indent=2))
        if result.returncode:raise SystemExit(f'{name} failed; inspect its log')
        print('DONE',name,flush=True)
state_file.write_text(json.dumps(state,indent=2));print('PILOT COMPLETE',flush=True)
