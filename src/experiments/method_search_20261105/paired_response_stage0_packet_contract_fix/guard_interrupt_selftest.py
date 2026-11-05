"""Finite dummy process tests; never invokes research execute(), LOCK or driver."""
import ast,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
H=Path(__file__).resolve().parent;source=H/'run_guarded_once.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
functions=[n for n in fn.body if isinstance(n,ast.FunctionDef) and n.name in ('track','kill_owned')]
assert len(functions)==2
handler=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='terminate_as_interrupt')
module=ast.Module(body=functions+[handler],type_ignores=[]);code=compile(module,str(source),'exec')
assert any(isinstance(n,ast.ExceptHandler) and isinstance(n.type,ast.Name) and n.type.id=='BaseException' for n in ast.walk(fn))
rows=[]
for mode in ('normal','KeyboardInterrupt','SIGTERM'):
 childcode='import time;time.sleep(.03)' if mode=='normal' else 'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],start_new_session=True);print(p.pid,flush=True);time.sleep(30)'
 child=subprocess.Popen([sys.executable,'-c',childcode],stdout=subprocess.PIPE,text=True,start_new_session=True)
 ns={'psutil':psutil,'child':child,'owned':{},'signal':signal};exec(code,ns);ns['track'](psutil.Process(child.pid));desc=None;status='running';previous=signal.signal(signal.SIGTERM,ns['terminate_as_interrupt'])
 try:
  if mode=='normal':child.wait(timeout=3);status='completed'
  else:
   desc=int(child.stdout.readline());assert desc not in [p for p,_ in ns['owned']]
   try:
    if mode=='KeyboardInterrupt':raise KeyboardInterrupt('direct synthetic interrupt')
    os.kill(os.getpid(),signal.SIGTERM)
   except BaseException:
    status='inconclusive_interrupted';ns['kill_owned']()
   child.wait(timeout=3)
  for pid in ([child.pid] if desc is None else [child.pid,desc]):
   try:assert psutil.Process(pid).status() in (psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD)
   except psutil.NoSuchProcess:pass
  assert status==('completed' if mode=='normal' else 'inconclusive_interrupted')
  rows.append({'mode':mode,'status':status,'initially_unobserved_descendant_pid':desc,'owned_after_cleanup':len(ns['owned']),'no_running_or_stopped_dummy_remains':True})
 finally:
  signal.signal(signal.SIGTERM,previous)
  if child.poll() is None:ns['kill_owned']();child.wait(timeout=3)
  if child.stdout:child.stdout.close()
report={'status':'passed','cases':3,'rows':rows,'scope':'exact extracted cleanup functions and signal handler on finite dummy processes; no research driver or authorization flow execution','guard_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'selftest_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'research_driver_starts':0,'LOCK_RUN_ONCE_files_created':False,'actual_tensor_reads':0,'GPU_calls':0}
with (H/'GUARD_INTERRUPT_SELFTEST.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':'passed','cases':3}))
