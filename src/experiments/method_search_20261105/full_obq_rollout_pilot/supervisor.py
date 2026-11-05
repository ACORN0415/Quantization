"""Single-use sequential pilot supervisor. A ready binding and reviewed LOCK are required."""
import json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import pilot_common as p

def idle():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
 return not out.strip()
def stop(child):
 if child is None or child.poll() is not None:return
 os.killpg(child.pid,signal.SIGTERM)
 try:child.wait(timeout=3)
 except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
def main():
 # Pending configuration is rejected before allocating a single-use run marker.
 b=json.loads((p.HERE/'STAGE2_BINDING.json').read_text())
 assert b.get('status')=='ready','Stage2 still pending; no pilot started'
 assert (p.HERE/'LOCK.json').exists(),'Root/Fable-reviewed LOCK is required'
 start=time.monotonic();run={'status':'starting','started_monotonic':start,'started_utc':time.time(),'stages':[]}
 target=p.HERE/'RUN.json'
 with target.open('x') as f:json.dump(run,f,indent=2)
 child=None
 try:
  def interrupted(signum,frame):raise KeyboardInterrupt(f'Supervisor received signal {signum}')
  signal.signal(signal.SIGTERM,interrupted)
  p.verify_lock()
  assert idle(),'GPU is busy; no overlapping pilot worker started'
  time.sleep(3);assert idle(),'GPU became busy during idle confirmation'
  counter=p.HERE/'COUNTERS.json'
  with counter.open('x') as f:json.dump({'generations':0,'vae_decodes':0},f)
  stages=[('prepare',['worker.py','prepare'],False),('integration',['worker.py','integration'],True),
   ('generate_0',['worker.py','generate','--input','0'],True),('generate_1',['worker.py','generate','--input','1'],True),
   ('quality',['quality.py'],True)]
  for name,args,gpu in stages:
   p.guard()
   if gpu:assert idle(),f'GPU busy before {name}'
   left=3595-(time.monotonic()-start)
   if left<=0:raise TimeoutError('60-minute budget; cleanup reserve reached')
   row={'name':name,'status':'running','started_utc':time.time()};run['stages'].append(row);run['status']='running';p.save(target,run)
   env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
   if not gpu:env['CUDA_VISIBLE_DEVICES']=''
   with (p.HERE/f'{name}.log').open('x') as log:
    child=subprocess.Popen([sys.executable,str(p.HERE/args[0])]+args[1:],cwd=p.ROOT/'src',env=env,
     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=child.wait(timeout=left)
    except BaseException:
     stop(child);row['status']='stopped';raise
   row.update(return_code=code,finished_utc=time.time(),status='completed' if code==0 else 'failed');p.save(target,run)
   child=None
   if code:raise RuntimeError(f'{name} failed; no retries or new candidates')
  q=json.loads((p.HERE/'QUALITY.json').read_text());assert q['status']=='completed'
  counts=json.loads(counter.read_text());assert counts=={'generations':8,'vae_decodes':8}
  p.verify_lock();run.update(status='completed',counters=counts,quality=p.record(p.HERE/'QUALITY.json'))
 except BaseException as exc:
  stop(child);run.update(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  run.update(finished_utc=time.time(),elapsed_seconds=time.monotonic()-start);p.save(target,run)
if __name__=='__main__':main()
