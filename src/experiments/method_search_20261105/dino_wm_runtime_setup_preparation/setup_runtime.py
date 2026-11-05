#!/usr/bin/env python3
"""Bounded, explicitly approved setup stages. No work runs on import.
Preparation is not execution authorization. Each stage is one attempt; no retry.
Offline children require bubblewrap read-only host + writable new destination.
"""
import argparse, email, hashlib, json, os, pathlib, re, resource, signal
import stat, subprocess, sys, time, urllib.parse, urllib.request, zipfile

HERE=pathlib.Path(__file__).resolve().parent
SPEC=HERE/'ROOT_SPEC.json'
BASE=pathlib.Path('/home/acorn0415/sfq_data/dino_wm_planning')
INTERPRETER='/home/acorn0415/sfq_data/env/bin/python'
ENVROOT=pathlib.Path('/home/acorn0415/sfq_data/env').resolve()
PROTECTED={'torch','torchvision','torchaudio','triton','nvidia','cuda','cudnn'}
HEX=lambda value,n:isinstance(value,str) and re.fullmatch('[0-9a-f]{'+str(n)+'}',value) is not None

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def require(value,message):
 if not value:raise ValueError(message)

def rel(value):
 require(isinstance(value,str) and value and '\\' not in value,'invalid relative path')
 p=pathlib.PurePosixPath(value)
 require(not p.is_absolute() and all(x not in ('','..','.') for x in value.split('/')),'path traversal')
 require(all(re.fullmatch('[A-Za-z0-9_.+-]+',x) for x in p.parts),'unsupported path component')
 return p

def norm(value):return re.sub('[-_.]+','-',value).lower()
def protected(value):return norm(value).split('-')[0] in PROTECTED

def check_manifest(m,stage):
 require(m.get('schema_version')==1 and m.get('status')=='ROOT_APPROVED_ISOLATED_SETUP','unapproved schema/status')
 require(m.get('root_execution_authorized') is True and m.get('actual_claude_source_review_completed') is True,'root/review authorization missing')
 require(m.get('complete') is True and m.get('source_closure_reviewed') is True,'incomplete source/dependency contract')
 require(m.get('runner_sha256')==sha(__file__) and m.get('root_spec_sha256')==sha(SPEC),'runner/spec hash changed')
 require(m.get('interpreter')==INTERPRETER,'wrong interpreter')
 require(stage in m.get('approved_stages',[]),'stage not approved')
 dest=pathlib.Path(m['destination']);require(dest.is_absolute() and dest.parent==BASE,'destination must be new direct task child')
 require(re.fullmatch('[a-zA-Z0-9_-]+',dest.name) is not None,'destination name')
 require(not dest.is_symlink() and BASE.resolve()==BASE,'destination parent symlink')
 if dest.exists():require(dest.resolve().parent==BASE,'resolved destination escaped')
 require(sha(INTERPRETER)==m['interpreter_sha256'],'interpreter changed')
 require(pathlib.Path(m['bwrap_path']).is_absolute() and sha(m['bwrap_path'])==m['bwrap_sha256'],'bwrap binding')
 require(isinstance(m['environment_guard_files'],dict) and m['environment_guard_files'],'environment guards empty')
 for p,h in m['environment_guard_files'].items():
  require(pathlib.Path(p).resolve().is_relative_to(ENVROOT) and HEX(h,64),'invalid environment guard');require(sha(p)==h,'existing environment guard changed')
 lim=m['limits'][stage]
 for key in ('seconds','disk_bytes','RSS_bytes'):require(isinstance(lim[key],int) and lim[key]>0,'missing positive stage budget '+key)
 require(m.get('CPU_threads')==2,'CPU contract')
 arts=m['artifacts'];require(isinstance(arts,list) and 0<len(arts)<=200,'artifact count')
 ids=set();paths=set()
 for a in arts:
  require(re.fullmatch('[A-Za-z0-9_-]+',a['id']) is not None and a['id'] not in ids,'duplicate/bad artifact id');ids.add(a['id'])
  p=rel(a['path']);require(str(p) not in paths,'duplicate path');paths.add(str(p))
  require(a['kind'] in ('source','wheel','sdist','weight'),'artifact kind')
  expected_root={'source':'sources','wheel':'artifacts','sdist':'artifacts','weight':'weights'}[a['kind']]
  require(p.parts[0]==expected_root and len(p.parts)>1,'wrong artifact subtree')
  require(all(not protected(part) for part in p.parts),'protected Torch/CUDA path')
  require(isinstance(a['bytes'],int) and a['bytes']>0,'exact byte size required')
  u=urllib.parse.urlsplit(a['url']);require(u.scheme=='https' and u.hostname and not u.username and not u.password and not u.fragment,'HTTPS URL contract')
  require(isinstance(a['allowed_hosts'],list) and u.hostname in a['allowed_hosts'],'host allowlist')
  if not HEX(a.get('sha256'),64):
   require(a['kind']=='weight' and a.get('sha256') is None and a.get('root_accepts_first_acquired_sha') is True,'unknown digest not allowed')
   ev=a['official_provenance'];require(pathlib.Path(ev['path']).is_absolute() and sha(ev['path'])==ev['sha256'],'official weight evidence missing')
  if a['kind']=='source':require(HEX(a.get('git_blob_sha1'),40) and HEX(a.get('revision'),40),'source git pin missing')
  if a['kind'] in ('wheel','sdist'):
   require(not protected(a['distribution']) and isinstance(a['version'],str) and a['version'],'protected/unversioned distribution')
   require(p.name==a['filename'],'package filename contract')
   if a['kind']=='sdist':require(a.get('root_approved_offline_build') is True and a.get('build_tool_versions'),'sdist offline build not approved')
 if stage=='acquire':require(isinstance(lim['HTTP_body_bytes'],int) and sum(a['bytes'] for a in arts)<=lim['HTTP_body_bytes'],'HTTP cap below exact artifacts')
 if stage=='imports':
  require(m.get('import_modules') and m.get('source_python_paths'),'import contract missing')
  for p in m['source_python_paths']:require(rel(p).parts[0]=='sources','source sys.path')
  for item in m['import_modules']:
   require(re.fullmatch(r'[A-Za-z_]\w*(\.[A-Za-z_]\w*)*',item['name']) is not None,'module name')
   require(item['scope'] in ('environment','overlay','source'),'module provenance scope')
  require(m.get('expected_torch_version') and m.get('expected_torch_dir'),'Torch provenance missing')
 return dest

def exclusive_json(path,data):
 with open(path,'x') as f:json.dump(data,f,indent=2,ensure_ascii=False)

def disk_bytes(root):
 total=0
 for p in root.rglob('*'):
  require(not p.is_symlink(),'unexpected symlink in task directory')
  if p.is_file():total+=p.stat().st_size
 return total

def validate_local(a,path):
 require(path.is_file() and not path.is_symlink() and path.stat().st_size==a['bytes'],'artifact file/size mismatch')
 digest=sha(path)
 if a.get('sha256'):require(digest==a['sha256'],'SHA256 mismatch')
 if a['kind']=='source':
  h=hashlib.sha1(b'blob '+str(a['bytes']).encode()+b'\0')
  with open(path,'rb') as f:
   for b in iter(lambda:f.read(65536),b''):h.update(b)
  require(h.hexdigest()==a['git_blob_sha1'],'git blob mismatch')
 return digest

class StrictRedirect(urllib.request.HTTPRedirectHandler):
 def __init__(self,hosts):super().__init__();self.hosts=hosts
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  u=urllib.parse.urlsplit(newurl)
  require(u.scheme=='https' and u.hostname in self.hosts and not u.username and not u.password,'redirect not approved')
  return super().redirect_request(req,fp,code,msg,headers,newurl)

def acquire(m,dest,guard):
 rows=[];read_bytes=0;cap=m['limits']['acquire']['HTTP_body_bytes']
 for a in m['artifacts']:
  guard();target=dest/str(rel(a['path']));target.parent.mkdir(parents=True,exist_ok=True);part=target.with_name(target.name+'.part')
  require(not target.exists() and not part.exists(),'no-overwrite acquisition')
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),StrictRedirect(a['allowed_hosts']))
  req=urllib.request.Request(a['url'],headers={'Accept-Encoding':'identity','User-Agent':'bounded-runtime-setup/1'})
  received=0;h=hashlib.sha256();t=time.time()
  with opener.open(req,timeout=min(20,max(1,guard()))) as response,open(part,'xb') as out:
   require(response.status==200,'unexpected HTTP status')
   length=response.headers.get('Content-Length')
   if length is not None:require(int(length)==a['bytes'],'server size mismatch')
   final=response.geturl();require(urllib.parse.urlsplit(final).hostname in a['allowed_hosts'],'final URL mismatch')
   response_identity={k:response.headers.get(k) for k in ('Content-Length','ETag','Last-Modified','x-amz-version-id')}
   while received<a['bytes']:
    guard();require(read_bytes<cap,'HTTP body cap exhausted');chunk=response.read(min(65536,a['bytes']-received,cap-read_bytes))
    if not chunk:break
    received+=len(chunk);read_bytes+=len(chunk);require(received<=a['bytes'] and read_bytes<=cap,'HTTP body cap')
    out.write(chunk);h.update(chunk)
  require(received==a['bytes'],'incomplete download');validate_local(a,part)
  os.link(part,target);part.unlink()
  rows.append({'id':a['id'],'path':str(target),'bytes':received,'sha256':h.hexdigest(),'digest_verified_against_prior':bool(a.get('sha256')),'first_acquired_sha_requires_independent_audit':a.get('sha256') is None,'requested_url':a['url'],'final_url':final,'response_identity':response_identity,'started_epoch':t,'completed_epoch':time.time()})
 return {'artifacts':rows,'HTTP_body_bytes':read_bytes,'model_load_authorized':False}

def inspect_wheel(path,distribution,version):
 with zipfile.ZipFile(path) as z:
  require(len(z.infolist())<=100000,'wheel entries cap');metadata=[]
  for i in z.infolist():
   parts=rel(i.filename.rstrip('/')).parts
   require(not stat.S_ISLNK(i.external_attr>>16),'wheel symlink')
   require(all(not protected(x) for x in parts),'shadow Torch/CUDA wheel content')
   require(not i.filename.endswith('.pth'),'executable .pth not accepted')
   if i.filename.endswith('.dist-info/METADATA'):metadata.append(i)
  require(len(metadata)==1 and metadata[0].file_size<=1024**2,'wheel METADATA contract')
  msg=email.message_from_bytes(z.read(metadata[0]))
  require(norm(msg['Name'])==norm(distribution) and msg['Version']==version,'wheel name/version mismatch')
 return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}

def offline_env(dest):
 env=dict(os.environ)
 for k in list(env):
  if k.startswith(('PYTHON','PIP_')) or k in ('LD_PRELOAD','LD_LIBRARY_PATH'):env.pop(k,None)
 env.update({'CUDA_VISIBLE_DEVICES':'','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','XFORMERS_DISABLED':'1','PYTHONBREAKPOINT':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PIP_NO_INDEX':'1','PIP_NO_INPUT':'1','PIP_DISABLE_PIP_VERSION_CHECK':'1','TMPDIR':str(dest/'tmp')})
 return env

def sandbox_command(m,dest,cmd):
 return [m['bwrap_path'],'--die-with-parent','--new-session','--unshare-user','--unshare-pid','--unshare-net','--ro-bind','/','/','--bind',str(dest),str(dest),'--tmpfs','/tmp','--proc','/proc','--dev','/dev','--chdir',str(dest),'--']+cmd

def process_tree_rss(pid):
 # The direct Popen child is still alive during observation, preventing root PID reuse.
 todo=[pid];seen=set();rss=0
 while todo:
  current=todo.pop()
  if current in seen:continue
  seen.add(current)
  try:
   for line in pathlib.Path(f'/proc/{current}/status').read_text().splitlines():
    if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
   children=pathlib.Path(f'/proc/{current}/task/{current}/children').read_text().split()
   todo.extend(map(int,children))
  except (FileNotFoundError,ProcessLookupError):pass
 return rss

def run_offline(m,dest,cmd,label,guard):
 log=dest/(label+'.log');require(not log.exists(),'no-overwrite log')
 limit=m['limits'][m['_stage']]['RSS_bytes'];proc=None;peak=0;pending=[]
 old={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
 def defer(sig,frame):pending.append(sig)
 def affinity():os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
 with open(log,'xb') as f:
  try:
   for sig in old:signal.signal(sig,defer)
   try:proc=subprocess.Popen(sandbox_command(m,dest,cmd),stdout=f,stderr=subprocess.STDOUT,env=offline_env(dest),start_new_session=True,preexec_fn=affinity)
   finally:
    for sig,handler in old.items():signal.signal(sig,handler)
   if pending:raise KeyboardInterrupt('interrupt during child start '+str(pending))
   while proc.poll() is None:
    guard();peak=max(peak,process_tree_rss(proc.pid));require(peak<=limit,'observed child-tree RSS cap');time.sleep(0.1)
   code=proc.returncode
  finally:
   if proc is not None and proc.poll() is None:proc.kill();proc.wait(timeout=10)
 require(code==0,'offline subprocess failed: '+label)
 guard();return {'command':cmd,'exit_code':code,'log':str(log),'log_sha256':sha(log),'observed_peak_child_tree_RSS_bytes':peak,'RSS_sampling_seconds':0.1,'read_only_host_network_namespace':'bubblewrap; no unsandboxed fallback'}

def installed_build_versions(m,dest,a,guard):
 script='import importlib.metadata as m,json;print(json.dumps({x:m.version(x) for x in '+repr(list(a['build_tool_versions']))+'},sort_keys=True))'
 result=run_offline(m,dest,[INTERPRETER,'-c',script],a['id']+'_build_tools',guard)
 actual=json.loads(pathlib.Path(result['log']).read_text().strip());require(actual==a['build_tool_versions'],'build tool versions changed')
 return actual

def acquire_record(m,dest,manifest_hash):
 record=json.loads((dest/'ACQUIRE_RESULT.json').read_text());require(record['status']=='completed' and record['manifest_sha256']==manifest_hash,'acquisition not complete for this manifest')
 stored={x['id']:x for x in record['detail']['artifacts']}
 for a in m['artifacts']:
  p=dest/a['path'];require(validate_local(a,p)==stored[a['id']]['sha256'],'acquired artifact modified')
 return stored

def install(m,dest,guard,manifest_hash):
 acquire_record(m,dest,manifest_hash);overlay=dest/'overlay';require(not overlay.exists(),'overlay already exists');overlay.mkdir();commands=[];built=[]
 for a in m['artifacts']:
  if a['kind'] not in ('wheel','sdist'):continue
  guard();wheel=dest/a['path']
  if a['kind']=='sdist':
   tools=installed_build_versions(m,dest,a,guard);builddir=dest/'build'/a['id'];builddir.mkdir(parents=True)
   commands.append(run_offline(m,dest,[INTERPRETER,'-m','pip','--isolated','wheel','--no-deps','--no-index','--no-build-isolation','--no-cache-dir','--wheel-dir',str(builddir),str(wheel)],a['id']+'_build',guard))
   found=list(builddir.glob('*.whl'));require(len(found)==1,'build must yield exactly one wheel');wheel=found[0]
   built.append({'id':a['id'],'tools':tools,**inspect_wheel(wheel,a['distribution'],a['version'])})
  else:inspect_wheel(wheel,a['distribution'],a['version'])
  commands.append(run_offline(m,dest,[INTERPRETER,'-m','pip','--isolated','install','--no-deps','--no-index','--no-compile','--no-cache-dir','--target',str(overlay),str(wheel)],a['id']+'_install',guard))
 return {'install_package_count':sum(a['kind'] in ('wheel','sdist') for a in m['artifacts']),'offline_build_count':len(built),'mode':'empty_overlay_no_package_install' if not commands else 'offline_explicit_packages','commands':commands,'built_wheels':built,'overlay_files':{str(p.relative_to(overlay)):sha(p) for p in overlay.rglob('*') if p.is_file()},'import_success':False,'model_load_authorized':False}

IMPORT_SCRIPT=r'''
import importlib,importlib.metadata,json,pathlib,pickle,sys
m=json.loads(pathlib.Path(sys.argv[1]).read_text());dest=pathlib.Path(m['destination'])
sys.path[:0]=[str(dest/p) for p in m['source_python_paths']]+[str(dest/'overlay')]
def deny(*a,**k):raise RuntimeError('model/tensor/hub/CUDA execution forbidden in import-only stage')
pickle.load=pickle.loads=deny
import torch
assert not torch.cuda.is_initialized()
assert torch.__version__==m['expected_torch_version']
assert pathlib.Path(torch.__file__).resolve().is_relative_to(pathlib.Path(m['expected_torch_dir']).resolve())
torch.load=deny;torch.serialization.load=deny;torch.hub.load=deny;torch.hub.load_state_dict_from_url=deny;torch.cuda._lazy_init=deny;torch.nn.Module.__init__=deny
rows=[]
for x in m['import_modules']:
 mod=importlib.import_module(x['name']);file=getattr(mod,'__file__',None)
 assert file is not None,('module lacks file provenance',x['name'])
 p=pathlib.Path(file).resolve();base={'environment':pathlib.Path('/home/acorn0415/sfq_data/env'),'overlay':dest/'overlay','source':dest/'sources'}[x['scope']]
 assert p.is_relative_to(base.resolve()),('module provenance mismatch',x['name'],str(p))
 rows.append({'module':x['name'],'file':str(p),'scope':x['scope'],'distribution_version':importlib.metadata.version(x['distribution']) if x.get('distribution') else None})
assert not torch.cuda.is_initialized()
print(json.dumps({'status':'imports_only_no_model_smoke','torch_file':torch.__file__,'torch_version':torch.__version__,'CUDA_initialized':torch.cuda.is_initialized(),'modules':rows,'model_construct_load_authorized':False},sort_keys=True))
'''

def imports(m,dest,guard,manifest_hash,manifest_path):
 acquire_record(m,dest,manifest_hash);r=json.loads((dest/'INSTALL_RESULT.json').read_text());require(r['status']=='completed' and r['manifest_sha256']==manifest_hash,'install not complete')
 actual={str(p.relative_to(dest/'overlay')):sha(p) for p in (dest/'overlay').rglob('*') if p.is_file()};require(actual==r['detail']['overlay_files'],'overlay changed')
 command=run_offline(m,dest,[INTERPRETER,'-c',IMPORT_SCRIPT,str(manifest_path)],'CPU_IMPORT',guard)
 # Other packages may print; require a unique final JSON result line.
 final=pathlib.Path(command['log']).read_text().splitlines()[-1];detail=json.loads(final);require(detail['status']=='imports_only_no_model_smoke' and detail['CUDA_initialized'] is False,'CPU import result')
 return {'command':command,'import_result':detail,'model_smoke_completed':False,'unknown_digest_model_load_authorized':False}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--stage',choices=['validate','acquire','install','imports'],required=True);a=parser.parse_args()
 started=time.time();path=pathlib.Path(a.manifest).resolve();m=json.loads(path.read_text());stage='acquire' if a.stage=='validate' else a.stage
 dest=check_manifest(m,stage);manifest_hash=sha(path)
 if a.stage=='validate':print(json.dumps({'status':'valid_contract_no_execution','manifest_sha256':manifest_hash}));return
 m['_stage']=stage
 if stage=='acquire':dest.mkdir(mode=0o700,exist_ok=False);(dest/'tmp').mkdir()
 else:require(dest.is_dir(),'missing acquired destination')
 deadline=started+m['limits'][stage]['seconds']
 def guard():
  remaining=deadline-time.time();require(remaining>0,'stage deadline exceeded');require(disk_bytes(dest)<=m['limits'][stage]['disk_bytes'],'disk cap');return remaining
 attempt=dest/(stage.upper()+'_ATTEMPT.json');exclusive_json(attempt,{'attempt':1,'started_epoch':started,'deadline_epoch':deadline,'manifest_sha256':manifest_hash,'runner_sha256':sha(__file__)})
 def interrupted(sig,frame):raise KeyboardInterrupt('signal '+str(sig))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 result={'stage':stage,'started_epoch':started,'deadline_epoch':deadline,'manifest_sha256':manifest_hash,'status':'failed'}
 try:
  guard()
  if stage=='acquire':detail=acquire(m,dest,guard)
  elif stage=='install':detail=install(m,dest,guard,manifest_hash)
  else:detail=imports(m,dest,guard,manifest_hash,path)
  guard();check_manifest(m,stage);require(sha(path)==manifest_hash,'manifest changed during stage');result.update(status='completed',detail=detail)
 except BaseException as exc:result.update(error_type=type(exc).__name__,error=str(exc));raise
 finally:
  result['completed_epoch']=time.time();result['elapsed_seconds']=result['completed_epoch']-started;result['disk_bytes_before_result']=disk_bytes(dest)
  exclusive_json(dest/(stage.upper()+'_RESULT.json'),result)
if __name__=='__main__':main()
