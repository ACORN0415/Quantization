"""Process-local toolchain repair; same source and original absolute deadline."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time,traceback
P=Path(__file__).resolve().parent;OLD=P.with_name('obq_gpu_encoder_stage2')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def save(name,data):
    path=P/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def environment():
    env=dict(os.environ);env.update(PATH='/usr/bin:/bin:'+env.get('PATH',''),CUDA_HOME='/usr',CC='/usr/bin/gcc-13',CXX='/usr/bin/g++-13')
    return env

def preflight():
    assert not (P/'PREFLIGHT.json').exists();env=environment()
    assert sys.executable=='/home/acorn0415/sfq_data/env/bin/python'
    tools={}
    for name in ('/usr/bin/ninja','/usr/bin/nvcc','/usr/bin/c++','/usr/bin/gcc-13','/usr/bin/g++-13'):
        out=subprocess.check_output([name,'--version'],env=env,text=True,stderr=subprocess.STDOUT)
        tools[name]={'version_output':out,'sha256':sha(name)}
    assert shutil.which('ninja',path=env['PATH'])=='/usr/bin/ninja'
    assert shutil.which('nvcc',path=env['PATH'])=='/usr/bin/nvcc'
    probe='import json,sys,torch; from torch.utils.cpp_extension import CUDA_HOME,get_cxx_compiler; print(json.dumps(dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,CUDA_HOME=CUDA_HOME,CXX=get_cxx_compiler(),gpu_initialized=torch.cuda.is_initialized())))'
    state=json.loads(subprocess.check_output([sys.executable,'-c',probe],env=env,text=True))
    assert state['CUDA_HOME']=='/usr' and state['CXX']=='/usr/bin/g++-13' and not state['gpu_initialized']
    for name in ('experiment.py','obq_pack.cu','selftest.py','SELFTEST.json','REGISTRATION_DRAFT_KO.txt'):
        assert sha(P/name)==sha(OLD/name)
    started=read(OLD/'SUPERVISOR.json')['started_utc']
    save('PREFLIGHT.json',{'created_utc':time.time(),'tools':tools,'state':state,'environment_overrides':{k:env[k] for k in ('PATH','CUDA_HOME','CC','CXX')},
        'source_copy_sha256':{name:sha(P/name) for name in ('experiment.py','obq_pack.cu','selftest.py','SELFTEST.json','REGISTRATION_DRAFT_KO.txt')},
        'original_started_utc':started,'absolute_deadline_utc':started+2700,'original_supervisor_sha256':sha(OLD/'SUPERVISOR.json')})

def lock():
    assert not (P/'LOCK.json').exists();pf=read(P/'PREFLIGHT.json')
    spec=importlib.util.spec_from_file_location('unchanged_stage2',P/'experiment.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    files=m.source_files()+[P/'env_supervisor.py',P/'ENV_RECOVERY_REGISTRATION_KO.txt',P/'PREFLIGHT.json',P/'FABLE_REVIEW_AT_LOCK_KO.txt',P/'ENV_REVIEW_AT_LOCK_KO.txt',OLD/'SUPERVISOR.json',OLD/'RESULT.json',OLD/'BUILD.json',OLD/'LOCK.json']
    save('LOCK.json',{'registered_utc':time.time(),'absolute_deadline_utc':pf['absolute_deadline_utc'],'original_started_utc':pf['original_started_utc'],
        'sha256':{str(p):sha(p) for p in files}})

def run():
    assert not (P/'SUPERVISOR.json').exists();lockdata=read(P/'LOCK.json');deadline=lockdata['absolute_deadline_utc'];start=time.monotonic();process=None
    r={'started_utc':time.time(),'absolute_deadline_utc':deadline,'original_started_utc':lockdata['original_started_utc'],'status':'running'}
    try:
        assert all(sha(p)==h for p,h in lockdata['sha256'].items())
        remaining=deadline-time.time();assert remaining>5,'original 45-minute budget exhausted'
        with (P/'RUN_LOG.txt').open('x') as out:
            process=subprocess.Popen([sys.executable,str(P/'experiment.py'),'worker'],env=environment(),stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=max(.01,deadline-time.time()-5))
            except subprocess.TimeoutExpired:
                r['timeout']=True;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=max(.01,deadline-time.time()))
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise TimeoutError('shared original 45-minute absolute deadline')
        r.update(return_code=code,status='finished' if code==0 else 'worker_failed')
    except BaseException as exc:
        if process is not None and process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
        r.update(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        ninja=P/'build/build.ninja'
        if ninja.exists():
            txt=ninja.read_text();ok=('cxx = /usr/bin/g++-13' in txt and 'nvcc = /usr/bin/nvcc' in txt and '-ccbin /usr/bin/gcc-13' in txt)
            r['build_ninja']={'sha256':sha(ninja),'expected_compilers_selected':ok}
        r.update(finished_utc=time.time(),elapsed_seconds=time.monotonic()-start,total_since_original_seconds=time.time()-r['original_started_utc'],deadline_exceeded=time.time()>deadline)
        save('SUPERVISOR.json',r)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('preflight','lock','run'));args=ap.parse_args()
    {'preflight':preflight,'lock':lock,'run':run}[args.action]()
