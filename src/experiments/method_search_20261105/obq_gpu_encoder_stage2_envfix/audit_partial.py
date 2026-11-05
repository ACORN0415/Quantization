"""Read-only audit of the preserved partial execution; no GPU or model."""
from pathlib import Path
import hashlib,json,re,time
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
checks=[]
def ck(name,ok):assert ok,name;checks.append(name)
t0=time.monotonic();l=read(P/'LOCK.json');r=read(P/'RESULT.json');b=read(P/'BUILD.json');s=read(P/'SUPERVISOR.json');pf=read(P/'PREFLIGHT.json')
for name,h in l['sha256'].items():ck(name,sha(name)==h)
ck('result lock',r['lock_sha256']==sha(P/'LOCK.json'))
ck('original deadline',s['absolute_deadline_utc']==pf['absolute_deadline_utc']==l['absolute_deadline_utc'] and not s['deadline_exceeded'] and s['total_since_original_seconds']<2700)
ck('source copies',all(sha(P/name)==h for name,h in pf['source_copy_sha256'].items()))
ck('build one success',len(b['attempts'])==1 and b['attempts'][0]['status']=='success' and b['source_sha256']==sha(P/'obq_pack.cu') and Path(b['module']).exists())
ninja=(P/'build/build.ninja').read_text()
ck('actual toolchain',s['build_ninja']['expected_compilers_selected'] and s['build_ninja']['sha256']==sha(P/'build/build.ninja') and all(x in ninja for x in ('cxx = /usr/bin/g++-13','nvcc = /usr/bin/nvcc','-ccbin /usr/bin/gcc-13','--fmad=false','--prec-div=true','--ftz=false')))
ck('failure retained',r['status']=='failed_or_incomplete' and s['status']=='worker_failed' and s['return_code']==1)
rows=r['code_lifetime_validation']
ck('two completed T_code cases',[(x['layer'],x['chunk'],x['stage']) for x in rows]==[(18,1,'T_code'),(24,6,'T_code')])
for row in rows:
 ck('all four implementations recorded '+str(row['layer']),set(row['lifetime'])=={'RTN_eager','RTN_graph','OBQ_graph','OBQ_fused'} and all(all(v.values()) for v in row['lifetime'].values()))
 ck('U copy recorded '+str(row['layer']),row['setup']['OBQ_graph']['input_copy_bytes']==16140288 and row['setup']['OBQ_graph']['factor_copied_each_call'] and not row['setup']['RTN_graph']['factor_copied_each_call'])
ck('no quality/timing',not (P/'QUALITY.json').exists() and r['timings']==[])
ck('no generation/model',r['models_loaded']==r['generations']==0)
ck('observed limits',r['max_RSS_bytes']<8*1024**3 and r['peak_cuda_allocated_bytes']<4*1024**3 and r['disk_bytes']<2*1024**3)
log=(P/'RUN_LOG.txt').read_text();ck('lifetime failure log','packed/scale/sentinel=[False, False, True]' in log and 'two=f(second);pcheck(two,ref2);pcheck(one,snapshot)' in log)
reg=re.search(r'Used (\d+) registers',log);spill=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
ck('ptxas resources recorded',reg is not None and spill is not None)
out={'passed':True,'checks_passed':len(checks),'locked_source_files':len(l['sha256']),'elapsed_seconds':time.monotonic()-t0,
 'complete_T_code_cases':len(rows),'complete_T_encoder_cases':0,'quality_endpoints':0,'timing_samples':0,'additional_GPU_execution':0,
 'ptxas':{'registers':int(reg[1]),'stack_bytes':int(spill[1]),'spill_store_bytes':int(spill[2]),'spill_load_bytes':int(spill[3])},
 'artifacts':{str(p.relative_to(P)):sha(p) for p in (P/'RESULT.json',P/'BUILD.json',P/'SUPERVISOR.json',P/'RUN_LOG.txt',Path(b['module']),Path(__file__))},
 'limits':['Recorded GPU assertions were not independently rerun.','Log does not identify the failed implementation or distinguish second-output reference mismatch from first-output mutation.','Compiler child peak RSS was not measured.']}
with (P/'AUDIT_PARTIAL.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
