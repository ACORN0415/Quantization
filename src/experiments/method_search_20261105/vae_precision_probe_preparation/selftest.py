"""CPU synthetic/statics only. Does not import model or load any original tensor."""
import ast,hashlib,importlib.util,json,math,tempfile,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('vae_precision_worker',HERE/'worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
checks=[]
def ok(name,condition):
 assert condition,name
 checks.append(name)
def rejects(name,fn):
 try:fn()
 except (AssertionError,KeyError,FileNotFoundError):checks.append(name)
 else:raise AssertionError(name)
a=np.zeros((501,3,2,2),dtype=np.float32);b=a+np.float32(1/32)
a_before=a.tobytes();b_before=b.tobytes()
r=w.statistics(a,b)
ok('501 exact frame rows',len(r)==501)
ok('MSE matches independent constant difference',all(x['MSE']==1/1024 for x in r))
ok('MAE and linear q99 exact',all(x['MAE']==1/32 and x['q99_abs']==1/32 for x in r))
ok('input arrays unchanged',a.tobytes()==a_before and b.tobytes()==b_before)
s=w.summarize(r)
ok('late fixed252frame RMSE',s['late_RGB_RMSE']==1/32)
ok('sensitivity gate positive without RGB drift criterion',s['decision']=='review_precision_sensitivity' and s['mean_RGB_change_reduction_descriptive'] is None)
z=w.statistics(a,a);zs=w.summarize(z)
ok('identical path zero stop',zs['late_RGB_RMSE']==0 and zs['decision']=='stop_precision_expansion')
rejects('shape mismatch rejects',lambda:w.statistics(a,b[:2]))
rejects('float64 input rejects',lambda:w.statistics(a.astype(np.float64),b.astype(np.float64)))
rejects('partial frame summary rejects',lambda:w.summarize(r[:500]))
rejects('missing LOCK rejects',lambda:w.read_lock(HERE/'DOES_NOT_EXIST_LOCK.json'))
ok('no RUN created',not (HERE/'RUN.json').exists())
source=(w.SOURCE/'utils/wan_wrapper.py').read_text();tree=ast.parse(source)
c=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WanVAEWrapper'][0]
methods={n.name:n for n in c.body if isinstance(n,ast.FunctionDef)}
ok('original unchanged decode and constructor present',set(['__init__','decode_to_pixel'])<=set(methods))
ok('wrapper false-cache branch is native decode','decode_function = self.model.decode' in source)
compile((HERE/'worker.py').read_text(),str(HERE/'worker.py'),'exec')
ok('worker compiles',True)
# A fake path reproduces the atomic-rename stat race without deleting any file.
class Gone:
 def is_file(self):return True
 def stat(self):raise FileNotFoundError('atomic rename fixture')
class Stable:
 def is_file(self):return True
 def stat(self):return type('S',(),{'st_size':7})()
class FakeRoot:
 def rglob(self,pattern):return iter([Gone(),Stable()])
original_path=w.Path
try:
 w.Path=lambda x:x
 ok('disappeared atomic temporary excluded',w.output_bytes(FakeRoot())==7)
 class Denied(Stable):
  def stat(self):raise PermissionError('must not silently ignore monitor failure')
 class BadRoot:
  def rglob(self,pattern):return iter([Denied()])
 try:w.output_bytes(BadRoot())
 except PermissionError:checks.append('other disk errors propagate to fail-closed watchdog')
 else:raise AssertionError('disk failure silently swallowed')
finally:w.Path=original_path
record={'status':'passed_synthetic_and_static_only','checks':checks,'count':len(checks),
 'actual_model_loads':0,'actual_latent_loads':0,'GPU_calls':0,'VAE_decodes':0,
 'numpy_version':np.__version__,'python_version':__import__('sys').version,
 'source_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['worker.py','selftest.py']},
 'native_WanVAEWrapper_AST_sha256':hashlib.sha256(ast.dump(c,include_attributes=False).encode()).hexdigest(),
 'finished_epoch':time.time(),
 'not_verified':['native VAE execution','CUDA memory peak','FP32 checkpoint stored dtype','501frame decoder walltime','watchdog live GPU safety','RGB numerical difference']}
with (HERE/'SELFTEST_V2.json').open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':record['status'],'count':len(checks),'worker_sha256':record['source_sha256']['worker.py']}))
