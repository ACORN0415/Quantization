"""Tiny local contracts only; does not run acquisition, pip, builds or Torch imports."""
import copy, hashlib, importlib.util, json, pathlib, tempfile, time, zipfile, stat, sys
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('setup_runtime_local',HERE/'setup_runtime.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
checks=[]
def ok(name,fn):fn();checks.append({'check':name,'passed':True})
def bad(name,fn):
 try:fn()
 except (ValueError,FileExistsError):checks.append({'check':name,'passed':True});return
 raise AssertionError('expected rejection: '+name)
with tempfile.TemporaryDirectory(prefix='local_fixture_',dir=HERE) as temp:
 root=pathlib.Path(temp);p=root/'payload';p.write_bytes(b'abc')
 a={'id':'tiny','kind':'source','path':'sources/tiny.py','url':'https://example.invalid/tiny.py','allowed_hosts':['example.invalid'],'bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob 3\0abc').hexdigest(),'revision':'1'*40}
 guardfile=pathlib.Path('/home/acorn0415/sfq_data/env/lib/python3.10/site-packages/torch/__init__.py')
 m={'schema_version':1,'status':'ROOT_APPROVED_ISOLATED_SETUP','root_execution_authorized':True,'actual_claude_source_review_completed':True,'complete':True,'source_closure_reviewed':True,'runner_sha256':r.sha(HERE/'setup_runtime.py'),'root_spec_sha256':r.sha(r.SPEC),'interpreter':r.INTERPRETER,'interpreter_sha256':r.sha(r.INTERPRETER),'bwrap_path':'/usr/bin/bwrap','bwrap_sha256':r.sha('/usr/bin/bwrap'),'destination':str(r.BASE/'synthetic_contract_only_never_created'),'environment_guard_files':{str(guardfile):r.sha(guardfile)},'approved_stages':['acquire','install','imports'],'CPU_threads':2,'limits':{s:{'seconds':10,'disk_bytes':1024**2,'RSS_bytes':256*1024**2,'HTTP_body_bytes':100} for s in ['acquire','install','imports']},'artifacts':[a]}
 ok('fully_bound_synthetic_contract_valid_no_execution',lambda:r.check_manifest(m,'acquire'))
 for key,val in [('root_execution_authorized',False),('actual_claude_source_review_completed',False),('complete',False),('source_closure_reviewed',False),('runner_sha256','0'*64),('destination',str(r.ENVROOT/'bad'))]:
  mm=copy.deepcopy(m);mm[key]=val;bad('reject_'+key,lambda mm=mm:r.check_manifest(mm,'acquire'))
 for path in ['../escape','/absolute','a//b','a/./b','a/../../b','a\\b']:
  bad('reject_path_'+path,lambda path=path:r.rel(path))
 mm=copy.deepcopy(m);mm['artifacts'].append(copy.deepcopy(a));bad('duplicate_artifact',lambda:r.check_manifest(mm,'acquire'))
 mm=copy.deepcopy(m);mm['artifacts'][0]['path']='sources/torch/__init__.py';bad('source_shadow_torch',lambda:r.check_manifest(mm,'acquire'))
 mm=copy.deepcopy(m);mm['artifacts'][0]['sha256']=None;bad('source_unknown_sha',lambda:r.check_manifest(mm,'acquire'))
 ok('local_size_sha_git_blob',lambda:r.validate_local(a,p))
 for field,val in [('bytes',4),('sha256','0'*64),('git_blob_sha1','0'*40)]:
  aa=copy.deepcopy(a);aa[field]=val;bad('bad_'+field,lambda aa=aa:r.validate_local(aa,p))
 j=root/'record.json';r.exclusive_json(j,{'first':1});bad('exclusive_record_no_overwrite',lambda:r.exclusive_json(j,{}));assert json.loads(j.read_text())=={'first':1}
 def wheel(name,extra=None,symlink=False):
  p=root/(name+'.whl')
  with zipfile.ZipFile(p,'w') as z:
   z.writestr('smallpkg-1.0.dist-info/METADATA','Metadata-Version: 2.1\nName: smallpkg\nVersion: 1.0\n')
   if extra:
    info=zipfile.ZipInfo(extra)
    if symlink:info.external_attr=(stat.S_IFLNK|0o777)<<16
    z.writestr(info,'dummy')
  return p
 good=wheel('good','smallpkg/__init__.py');ok('tiny_wheel_metadata',lambda:r.inspect_wheel(good,'smallpkg','1.0'))
 for label,path,symlink in [('traversal','../escape',False),('torch','torch/__init__.py',False),('cuda','nvidia/cuda/lib.so',False),('pth','smallpkg.pth',False),('symlink','smallpkg/link',True)]:
  pp=wheel(label,path,symlink);bad('wheel_'+label,lambda pp=pp:r.inspect_wheel(pp,'smallpkg','1.0'))
 bad('wheel_distribution_mismatch',lambda:r.inspect_wheel(good,'other','1.0'))
 bad('wheel_version_mismatch',lambda:r.inspect_wheel(good,'smallpkg','2.0'))
 # Zero-package path creates only an empty target and verifies local tiny source.
 empty=root/'empty_install';(empty/'sources').mkdir(parents=True);(empty/'sources/tiny.py').write_bytes(b'abc')
 r.exclusive_json(empty/'ACQUIRE_RESULT.json',{'status':'completed','manifest_sha256':'synthetic_fixture_only','detail':{'artifacts':[{'id':'tiny','sha256':a['sha256']}]}})
 zero=r.install(m,empty,lambda:5,'synthetic_fixture_only')
 assert zero['install_package_count']==0 and zero['offline_build_count']==0 and zero['commands']==[] and zero['overlay_files']=={}
 checks.append({'check':'zero_package_empty_overlay_no_pip_no_build','passed':True})
 # Execute only stdlib dummy processes. No network requests, package installation/import, build, or assets.
 dest=root/'sandbox';dest.mkdir();(dest/'tmp').mkdir();outside=root/'readonly.txt';outside.write_text('unchanged');m['_stage']='install'
 def budget():return 5.0
 script="from pathlib import Path; import errno; p=Path("+repr(str(outside))+");\ntry:p.write_text('bad');raise AssertionError('host writable')\nexcept OSError as e:assert e.errno in (errno.EACCES,errno.EPERM,errno.EROFS)\nPath('allowed.txt').write_text('ok');print('readonly-host-dummy-ok')"
 ok('sandbox_readonly_host_writable_destination',lambda:r.run_offline(m,dest,[sys.executable,'-c',script],'DUMMY_READONLY',budget))
 assert outside.read_text()=='unchanged' and (dest/'allowed.txt').read_text()=='ok'
 until=time.time()+0.3
 def deadline():r.require(time.time()<until,'test deadline');return max(0.001,until-time.time())
 bad('dummy_deadline_child_cleanup',lambda:r.run_offline(m,dest,[sys.executable,'-c','import time;time.sleep(10)'],'DUMMY_TIMEOUT',deadline))
 summary={'status':'passed','checks':checks,'count':len(checks),'runner_sha256':r.sha(HERE/'setup_runtime.py'),'selftest_sha256':r.sha(__file__),'network_requests':0,'actual_acquisition_stages':0,'package_payloads':0,'pip_installs_builds':0,'Torch_or_model_imports':0,'GPU':0,'synthetic_approved_contract_persisted':False,'note':'Validation dictionary only; no approved runtime manifest or setup destination created. Two stdlib-only sandbox child processes; not a real setup test.'}
(HERE/'SELFTEST_LOCAL.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'status':summary['status'],'count':summary['count'],'runner_sha256':summary['runner_sha256']}))
