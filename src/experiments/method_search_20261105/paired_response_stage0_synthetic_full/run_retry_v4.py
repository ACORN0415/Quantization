"""Outer 20-minute fixture watchdog; unchanged copied driver keeps its own guard."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent
reg=json.loads((HERE/'FIXTURE_REGISTRATION.json').read_text());reg['copied_source_sha256']=json.loads((HERE/'FIXTURE_RETRY_V4.json').read_text())['copied_source_sha256'];deadline=reg['started_utc']+1200
status={'status':'running','synthetic_only':True,'started_utc':reg['started_utc'],'deadline_utc':deadline,'stages':[],'max_RSS_by_pid':{},'actual_method_effect_measurements':0,'original_capture_reads':0,'GPU_calls':0}
def save():
 (HERE/'FULL_FIXTURE_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
def identity(p):return (p.pid,p.create_time())
def stop(process):
 try:children=psutil.Process(process.pid).children(recursive=True)+[psutil.Process(process.pid)]
 except psutil.NoSuchProcess:return
 known=[]
 for p in children:
  try:known.append((p,identity(p)));p.terminate()
  except psutil.NoSuchProcess:pass
 _,live=psutil.wait_procs([p for p,_ in known],timeout=2)
 for p in live:
  try:
   if identity(p) in [x for _,x in known]:p.kill()
  except psutil.NoSuchProcess:pass
child=None;log=None;save()
try:
 for name,cmd in [('full_driver',[sys.executable,str(HERE/'diagnose.py'),'--supervisor'])]:
  if time.time()>=deadline:raise TimeoutError('20min fixture before stage')
  log=(HERE/(name+'.log')).open('x');child=subprocess.Popen(cmd,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONPATH':str(HERE),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
  status['active_stage']=name;status['active_pid']=child.pid;save()
  while child.poll() is None:
   if time.time()>=deadline:raise TimeoutError('20min fixture total including preparation')
   size=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
   if size>2*1024**3:raise OSError('2GiB fixture generated files')
   status['disk_bytes']=size
   try:processes=[psutil.Process(child.pid)]+psutil.Process(child.pid).children(recursive=True)
   except psutil.NoSuchProcess:processes=[]
   for p in processes:
    try:
     rss=p.memory_info().rss;pid=str(p.pid);status['max_RSS_by_pid'][pid]=max(status['max_RSS_by_pid'].get(pid,0),rss)
     if rss>8*1024**3:raise MemoryError('fixture child 8GiB RSS')
    except psutil.NoSuchProcess:pass
   save();time.sleep(.5)
  log.close();log=None;status['stages'].append({'stage':name,'returncode':child.returncode});save()
  if child.returncode:raise RuntimeError('synthetic stage exception: '+name)
  child=None
 result=json.loads((HERE/'RESULT.json').read_text());assert result['status']=='completed' and len(result['rows'])==68
 for source,digest in reg['copied_source_sha256'].items():
  assert hashlib.sha256((HERE/source).read_bytes()).hexdigest()==digest
  assert hashlib.sha256((HERE.parent/'paired_response_stage0'/source).read_bytes()).hexdigest()==digest
 assert not (HERE.parent/'paired_response_stage0'/'LOCK.json').exists() and not (HERE.parent/'paired_response_stage0'/'RUN.json').exists()
 audits=[json.loads(p.read_text()) for p in HERE.glob('IO_AUDIT_*.json')]
 assert all(not x['denied_external_tensor_paths'] for x in audits)
 status.update(status='completed',result_cells=68,synthetic_decision=result['decision'],result_sha256=hashlib.sha256((HERE/'RESULT.json').read_bytes()).hexdigest(),io_audit_processes=len(audits),original_stage0_LOCK_RUN_absent=True,source_bytes_unchanged=True)
except Exception as e:
 status.update(status='incomplete_or_invalid',error=repr(e))
 if child is not None and child.poll() is None:stop(child)
 raise
finally:
 if log is not None:log.close()
 status['elapsed_seconds']=time.time()-reg['started_utc'];save()
