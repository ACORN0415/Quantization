"""Read-only source/result audit of the preregistered build failure; no CUDA."""
from pathlib import Path
import hashlib,json,time
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
checks=[]
def check(name,ok):
    assert ok,name
    checks.append(name)
t0=time.monotonic();lock=read(P/'LOCK.json');r=read(P/'RESULT.json');b=read(P/'BUILD.json');s=read(P/'SUPERVISOR.json');review=read(P/'REVIEW_LINK.json');st=read(P/'SELFTEST.json')
for path,h in lock['sha256'].items():check('source '+path,sha(path)==h)
check('result lock hash',r['lock_sha256']==sha(P/'LOCK.json'))
check('review snapshot',sha(P/review['snapshot'])==review['review_sha256'])
check('reviewed source hashes',review['python_sha256']==sha(P/'experiment.py') and review['cuda_sha256']==sha(P/'obq_pack.cu') and review['registration_sha256']==sha(P/'REGISTRATION_DRAFT_KO.txt'))
check('selftest code hashes',st['passed'] and st['python_sha256']==sha(P/'experiment.py') and st['cuda_sha256']==sha(P/'obq_pack.cu') and st['selftest_sha256']==sha(P/'selftest.py'))
check('two same-source load attempts',len(b['attempts'])==2 and b['source_sha256']==sha(P/'obq_pack.cu') and [x['status'] for x in b['attempts']]==['failed','failed'])
check('first failure is Ninja prerequisite',b['attempts'][0]['error']=="RuntimeError('Ninja is required to load C++ extensions')")
check('second is missing extension',b['attempts'][1]['error'].startswith('ImportError(') and '.so: cannot open shared object file' in b['attempts'][1]['error'])
check('run explicitly failed',r['status']=='failed_or_incomplete' and s['status']=='worker_failed' and s['return_code']==1)
check('no code/quality/timing measurement',r['timings']==[] and 'code_lifetime_validation' not in r and not (P/'QUALITY.json').exists() and not list((P/'build').glob('*.so')))
check('no model or generation',r['models_loaded']==r['generations']==0)
check('elapsed/observed resources',s['elapsed_seconds']<2700 and r['max_RSS_bytes']<8*1024**3 and r['peak_cuda_allocated_bytes']<4*1024**3 and r['disk_bytes']<2*1024**3)
log=(P/'RUN_LOG.txt').read_text()
check('actual failure log',"/usr/local/bin/ninja" in log and "ModuleNotFoundError: No module named 'ninja'" in log and 'skipping build step' in log)
result={'passed':True,'checks_passed':len(checks),'locked_source_files':len(lock['sha256']),'elapsed_seconds':time.monotonic()-t0,
        'artifacts':{p.name:sha(p) for p in (P/'RESULT.json',P/'BUILD.json',P/'SUPERVISOR.json',P/'RUN_LOG.txt',Path(__file__))},
        'CUDA_compiler_invocations_observed':0,'custom_kernel_executions':0,'quality_endpoints':0,'timing_samples':0,'GPU_reexecutions':0,
        'scope':'Audit verifies preserved failure records and file hashes. Worker initialized CUDA and copied 64KiB Hadamard; no custom kernel/math/quality result exists.',
        'unmeasured':'Compiler subprocess peak RSS; actual compiler not reached.'}
with (P/'AUDIT_FAILURE.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
