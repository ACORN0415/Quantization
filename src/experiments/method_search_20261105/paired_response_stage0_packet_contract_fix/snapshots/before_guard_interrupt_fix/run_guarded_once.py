"""Prepared only: explicit one repair run under the ORIGINAL absolute deadline.

No invocation without --execute starts a process or writes execution records.
This wrapper does not grant authority; a reviewed root LOCK must explicitly
permit the additional validity-repair run. No original record is changed.
"""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'paired_response_stage0'
ABSOLUTE_DEADLINE_UTC=1793889540.4246237
RSS_CAP=8*1024**3
COMBINED_DISK_CAP=2*1024**3
SHA_ORIGINAL_RUN='6d667837f032b0b3cea211084ab73af66e49ffe29d27f01f1cdde91315892c40'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def combined_disk():return sum(p.stat().st_size for base in (ORIGINAL,HERE) for p in base.rglob('*') if p.is_file())
def validate_original():
 if sha(ORIGINAL/'RUN.json')!=SHA_ORIGINAL_RUN:raise PermissionError('Original invalid RUN changed')
 run=json.loads((ORIGINAL/'RUN.json').read_text())
 if run['status']!='invalid' or run['started_utc']!=1793886840.4246237:raise PermissionError('Original run identity/status mismatch')
 if run['stages']!=[{'label':'prepare_18_state','exit_code':0},{'label':'arm_18_R_bf16','exit_code':1}]:raise PermissionError('Original invalid exit chain mismatch')
 if abs(run['started_utc']+2700-ABSOLUTE_DEADLINE_UTC)>1e-6:raise PermissionError('Original deadline mismatch')
 binding=json.loads((HERE/'ORIGINAL_RUN_BINDING.json').read_text())
 if sha(ORIGINAL/'LOCK.json')!=binding['original_lock_sha256'] or sha(ORIGINAL/'diagnose.py')!=binding['original_driver_sha256']:raise PermissionError('Original source/LOCK changed')
 return run

def execute():
 # This clock never resets the old 45-minute allowance. UTC and monotonic both
 # enforce the remaining original time even if the wall clock moves backwards.
 began=time.time();remaining=ABSOLUTE_DEADLINE_UTC-began
 if remaining<=0:raise TimeoutError('Original 45-minute deadline already elapsed')
 mono_deadline=time.monotonic()+remaining
 original=validate_original()
 if (HERE/'RUN.json').exists() or (HERE/'ONCE_CLAIM.json').exists():raise FileExistsError('Refuse any second repaired invocation')
 lock=json.loads((HERE/'LOCK.json').read_text())
 required={'root_execution_authorized':True,'actual_opus_review_complete':True,'validity_policy_accepted':True,'validity_repair_additional_run_authorized':True,'maximum_additional_runs':1,'original_run_sha256':SHA_ORIGINAL_RUN,'absolute_deadline_utc':ABSOLUTE_DEADLINE_UTC}
 if any(lock.get(k)!=v for k,v in required.items()):raise PermissionError('Missing explicit reviewed validity-repair authority')
 if lock.get('pending'):raise PermissionError('Unresolved review contract')
 os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
 spec=importlib.util.spec_from_file_location('guarded_frozen_driver',HERE/'diagnose.py');d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d);d.setup_cpu();_,lock_sha=d.verify_lock()
 # Recheck hash of the wrapper itself via the frozen declared source set.
 if str(Path(__file__).resolve()) not in lock['sources'] or sha(__file__)!=lock['sources'][str(Path(__file__).resolve())]:raise PermissionError('Unbound external supervisor')
 if combined_disk()>COMBINED_DISK_CAP:raise OSError('Combined original+new folder disk cap before launch')
 if time.time()>=ABSOLUTE_DEADLINE_UTC or time.monotonic()>=mono_deadline:raise TimeoutError('Deadline reached during verification')
 claim={'started_utc':time.time(),'original_run_sha256':SHA_ORIGINAL_RUN,'original_deadline_utc':ABSOLUTE_DEADLINE_UTC,'lock_sha256':lock_sha,'maximum_additional_runs':1,'meaning':'one authorized validity-repair invocation; never an automatic retry loop'}
 with (HERE/'ONCE_CLAIM.json').open('x') as f:json.dump(claim,f,indent=2);f.write('\n')
 state={**claim,'status':'running','wrapper_pid':os.getpid(),'max_RSS_by_identity':{},'combined_disk_cap_bytes':COMBINED_DISK_CAP,'sources_verified':True};owned={};child=None;log=None
 def track(process):
  try:owned[(process.pid,process.create_time())]=process
  except psutil.NoSuchProcess:pass
 def kill_owned():
  # Only observed descendants of the subprocess created by THIS invocation.
  targets=[]
  for (pid,created),process in list(owned.items()):
   try:
    if process.is_running() and process.create_time()==created:targets.append(process);process.terminate()
   except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
  _,live=psutil.wait_procs(targets,timeout=2)
  for process in live:
   try:
    if (process.pid,process.create_time()) in owned:process.kill()
   except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
 try:
  log=(HERE/'guarded_supervisor.log').open('x');child=subprocess.Popen([sys.executable,str(HERE/'diagnose.py'),'--supervisor'],cwd=d.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ));track(psutil.Process(child.pid));state['supervisor_pid']=child.pid
  write(HERE/'GUARDED_STATUS.json',state)
  while child.poll() is None:
   if time.time()>=ABSOLUTE_DEADLINE_UTC or time.monotonic()>=mono_deadline:raise TimeoutError('Original absolute deadline reached')
   used=combined_disk();state['combined_disk_bytes']=used
   if used>COMBINED_DISK_CAP:raise OSError('Combined original+new folder disk cap')
   try:processes=[psutil.Process(child.pid)]+psutil.Process(child.pid).children(recursive=True)
   except psutil.NoSuchProcess:processes=[]
   for process in processes:
    try:
     track(process);identity=f'{process.pid}:{process.create_time()}';rss=process.memory_info().rss;state['max_RSS_by_identity'][identity]=max(state['max_RSS_by_identity'].get(identity,0),rss)
     if rss>RSS_CAP:raise MemoryError('Child process RSS exceeds8GiB')
    except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
   if psutil.Process().memory_info().rss>RSS_CAP:raise MemoryError('Wrapper RSS cap')
   write(HERE/'GUARDED_STATUS.json',state);time.sleep(.25)
  state['supervisor_exit_code']=child.returncode
  if child.returncode:raise RuntimeError('Repaired execution failed; no further retry permitted')
  if time.time()>=ABSOLUTE_DEADLINE_UTC or time.monotonic()>=mono_deadline:raise TimeoutError('Completion exceeded original deadline')
  if combined_disk()>COMBINED_DISK_CAP:raise OSError('Completion disk cap')
  validate_original();d.verify_lock()
  if time.time()>=ABSOLUTE_DEADLINE_UTC or time.monotonic()>=mono_deadline:raise TimeoutError('Final verification exceeded original deadline')
  state['status']='completed';state['original_record_unchanged']=True
 except Exception as exc:
  state.update(status='incomplete_budget' if isinstance(exc,(TimeoutError,MemoryError,OSError)) else 'invalid_repaired_execution',error=repr(exc))
  if child is not None:kill_owned()
  raise
 finally:
  if log is not None:log.close()
  state['finished_utc']=time.time();state['wrapper_elapsed_seconds']=time.time()-began;state['elapsed_from_original_run_seconds']=time.time()-original['started_utc'];write(HERE/'GUARDED_STATUS.json',state)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 if args.execute:execute()
 else:print(json.dumps({'status':'preparation_only','execution_started':False,'root_reviewed_LOCK_required':True,'additional_runs_max':1,'original_deadline_utc':ABSOLUTE_DEADLINE_UTC,'combined_disk_cap_bytes':COMBINED_DISK_CAP}))
