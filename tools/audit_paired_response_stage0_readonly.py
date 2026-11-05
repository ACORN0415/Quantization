"""One-shot post-experiment scalar/file audit. No torch, tensor decoding, or scoring.

Invoke only after the root coordinator reports terminal execution. Existing files
are read-only; the sole output is a new AUDIT_INDEPENDENT_CODEX.json in --folder.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

ARMS=('R_bf16','R_fp32','FULL','FIXED17','BIT8','RMU','RP3','RP3Q','FULLP1','RQ','FULLQ','FULLP1Q','FM3','FM8','PAIR17','PAIR29','PAIR43')
PRIMARY=ARMS[:12]
DENSE=dict(zip(ARMS,(0,0,786432,786444,1806361,6144,1694328,1694328,1800600,0,786432,1800600,3267048,4967208,1403904,1403904,1403904)))
CAP=1806361
DEADLINE=1793889540.4246237
LOCK_SHA='5fc4f125866b7630aacf1d2d5394cdd8579e5e6652647e03c870f604e109261d'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def audit(folder):
    began=time.monotonic();started=time.time();folder=Path(folder).resolve()
    output=folder/'AUDIT_INDEPENDENT_CODEX.json'
    if output.exists():raise FileExistsError(output)
    guard=read(folder/'GUARDED_STATUS.json');run=read(folder/'RUN.json')
    if guard.get('status')=='running' or run.get('status')=='running':
        raise RuntimeError('Terminal root completion signal required; no polling or active-run audit')
    checks=[];expected={};sizes={};origins=collections.defaultdict(set)
    def check(name,ok,**detail):
        checks.append({'name':name,'passed':bool(ok),**detail})
    def bind(path,digest,origin,size=None):
        path=str(Path(path).resolve())
        if path in expected and expected[path]!=digest:
            check('conflicting_expected_hash',False,path=path,origins=sorted(origins[path]|{origin}))
        expected[path]=digest;origins[path].add(origin)
        if size is not None:sizes[path]=size
    def records(value,origin):
        if isinstance(value,dict):
            if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
                bind(value['path'],value['sha256'],origin,value.get('bytes'))
            for v in value.values():records(v,origin)
        elif isinstance(value,list):
            for v in value:records(v,origin)
    lock=read(folder/'LOCK.json');inputs=read(folder/'INPUT_BINDINGS_DRAFT.json')
    check('locked_version',sha(folder/'LOCK.json')==LOCK_SHA and run.get('lock_sha256')==LOCK_SHA and guard.get('lock_sha256')==LOCK_SHA)
    check('registered_arms_baselines_cap',lock['arms']==list(ARMS) and lock['primary_baselines']==list(PRIMARY) and lock['dense_extra_bytes']==DENSE and lock['cap_extra_bytes']==CAP)
    check('wrapper_completed_exit0',guard.get('status')=='completed' and guard.get('supervisor_exit_code')==0)
    check('supervisor_completed',run.get('status')=='completed')
    check('original_deadline',lock.get('absolute_deadline_utc')==DEADLINE and guard.get('original_deadline_utc')==DEADLINE and guard.get('finished_utc',float('inf'))<=DEADLINE and guard.get('elapsed_from_original_run_seconds',float('inf'))<=2700)
    check('one_repair_only',lock.get('maximum_additional_runs')==guard.get('maximum_additional_runs')==1)
    check('sampled_RSS_cap',all(v<=8*1024**3 for v in guard.get('max_RSS_by_identity',{}).values()))
    check('recorded_disk_cap',guard.get('combined_disk_bytes',float('inf'))<=2*1024**3)
    original=folder.parent/'paired_response_stage0'
    disk=sum(p.stat().st_size for base in (original,folder) for p in base.rglob('*') if p.is_file())
    check('postrun_combined_disk_cap',disk<=2*1024**3,bytes=disk)
    stages=run.get('stages',[])
    stage_names=[f'{kind}_{layer}_{arm}' for layer in (18,24) for kind,arm in [('prepare','state')]+[('arm',a) for a in ARMS]]
    check('36_unique_expected_stages',len(stages)==36 and [s['label'] for s in stages]==stage_names and all(s['exit_code']==0 for s in stages))
    for category in ('sources','review_evidence_sha256'):
        for path,digest in lock[category].items():bind(path,digest,category)
    check('76_sources_38_review_records',len(lock['sources'])==76 and len(lock['review_evidence_sha256'])==38,source_count=len(lock['sources']),review_count=len(lock['review_evidence_sha256']))
    bind(folder/'INPUT_BINDINGS_DRAFT.json',lock['input_bindings_sha256'],'input_binding')
    preserved=read(folder/'ROOT_ORIGINAL_PRESERVATION_BASELINE.json')['sha256']
    check('8_original_preserved_records',len(preserved)==8)
    for path,digest in preserved.items():bind(path,digest,'original_preservation')
    records(inputs,'input_artifact')
    documents=[];child_rows=[]
    for layer in (18,24):
        prep=folder/f'PREPARE_{layer}.json'
        if prep.exists():documents.append((prep.name,read(prep)))
        for arm in ARMS:
            p=folder/f'RESULT_{layer}_{arm}.json'
            if p.exists():
                d=read(p);documents.append((p.name,d));child_rows.extend(d.get('rows',[]))
                check('worker_identity_'+p.stem,d.get('status')=='completed' and d.get('arm')==arm and d.get('layer')==layer)
    for name,d in documents:
        records(d,name)
        check('worker_RSS_'+name,d.get('peak_RSS_bytes',float('inf'))<=8*1024**3)
    result=read(folder/'RESULT.json') if (folder/'RESULT.json').exists() else {}
    rows=result.get('rows',[]);grid={(f'causal_layer{l}_c7_call{c}',a) for l in (18,24) for c in (0,3) for a in ARMS}
    keys=[(r.get('endpoint'),r.get('arm')) for r in rows]
    check('68_unique_grid',len(rows)==68 and len(set(keys))==68 and set(keys)==grid)
    check('aggregate_equals_child_rows',rows==child_rows)
    anchors={('causal_layer%d_c7_call%s'%(l['layer'],call),arm):s for l in inputs['layers'] for call,arms in l['anchors'].items() for arm,s in arms.items()}
    anchor_count=0
    for i,row in enumerate(rows):
        arm=row['arm'];extra=row.get('extra_persistent_bytes',-1)
        check('cell_validity_%d'%i,row.get('validity')=='valid' and 0<=extra<=DENSE.get(arm,-1) and row.get('dense_extra_bound_bytes')==DENSE.get(arm))
        check('cell_cap_%d'%i,row.get('cap_eligible')==(arm not in ('FM3','FM8')) and (arm in ('FM3','FM8') or extra<=CAP))
        for field in ('primary_Konly','full_BF16_reference','CPU_vs_native_floor'):
            metric=row.get(field,{})
            ok=all(isinstance(metric.get(x),(int,float)) and math.isfinite(metric[x]) and metric[x]>=0 for x in ('sse','reference_energy'))
            check('metric_%d_%s'%(i,field),ok)
            if ok:
                energy=metric['reference_energy'];nse=metric.get('nse')
                check('nse_scalar_%d_%s'%(i,field),nse is None if energy==0 else isinstance(nse,(int,float)) and math.isclose(nse,metric['sse']/energy,rel_tol=1e-12,abs_tol=0))
        if (row['endpoint'],arm) in anchors:
            anchor_count+=1
            check('legacy_Konly_anchor_%d'%i,math.isclose(row['primary_Konly']['sse'],anchors[row['endpoint'],arm],rel_tol=1e-9,abs_tol=1e-9))
        if arm.startswith('PAIR'):
            count=row.get('fallback_count');rate=row.get('fallback_rate')
            check('fallback_units_%d'%i,isinstance(count,int) and 0<=count<=1536 and row.get('fallback_pairs_total')==1536 and isinstance(rate,(int,float)) and math.isclose(rate,count/1536,rel_tol=0,abs_tol=1e-15))
            check('denominator_diagnostics_%d'%i,row.get('log_denominator_ratio_summary',{}).get('total_pairs')==1536)
    check('16_legacy_anchors',anchor_count==16)
    recomputed=[]
    if len(rows)==68 and set(keys)==grid and len(set(keys))==68:
        for endpoint in sorted({e for e,a in grid}):
            rs={r['arm']:r for r in rows if r['endpoint']==endpoint}
            best=min(PRIMARY,key=lambda a:rs[a]['primary_Konly']['sse']);base=rs[best]['primary_Konly']['sse']
            for seed in (17,29,43):
                rr=rs['PAIR'+str(seed)];gain=1-rr['primary_Konly']['sse']/base if base>0 else None
                passed=base>0 and gain>=.1 and rr['fallback_count']/1536<=.001 and rr['cap_eligible']
                recomputed.append({'endpoint':endpoint,'seed':seed,'best_baseline':best,'gain':gain,'passed':passed})
        check('12_seed_endpoint_decisions',recomputed==result.get('comparisons'))
    independent_decision=('continue_local_feasibility' if all(x['passed'] for x in recomputed) else 'stop_fixed_Gaussian_rank32') if len(recomputed)==12 else 'inconclusive_incomplete_or_invalid'
    check('decision_from_stored_scalars',result.get('decision')==independent_decision)
    hashes=[];hashed_bytes=0
    for path,digest in sorted(expected.items()):
        if time.monotonic()-began>175:
            check('audit_time_cap',False);break
        p=Path(path)
        if not p.is_file():check('file_missing',False,path=path);continue
        actual=sha(p);size=p.stat().st_size;hashed_bytes+=size
        ok=actual==digest and (path not in sizes or sizes[path]==size)
        check('file_integrity',ok,path=path)
        hashes.append({'path':path,'sha256':actual,'expected_sha256':digest,'bytes':size,'match':ok,'origins':sorted(origins[path])})
    check('all_unique_paths_hashed',len(hashes)==len(expected))
    passed=all(c['passed'] for c in checks)
    report={'status':'passed' if passed else 'failed_or_inconclusive','research_decision':independent_decision if passed else 'inconclusive_audit_or_execution_invalid','created_utc':time.time(),'audit_started_utc':started,'audit_after_experiment_deadline':started>DEADLINE,'audit_wall_seconds':time.monotonic()-began,'checks':checks,'checks_total':len(checks),'all_checks_passed':passed,'independent_scalar_comparisons':recomputed,'unique_file_count':len(hashes),'hashed_bytes':hashed_bytes,'files':hashes,'source_sha256':sha(__file__),'limits':{'time_seconds':180,'CPU_threads_max':4,'GPU_calls':0,'tensor_loads':0,'attention_SSE_recomputations':0},'scope':'Stored scalar arithmetic and byte integrity only; no independent output tensor/SSE reconstruction. Sampled RSS records are not a continuous peak proof. Post-experiment audit time is separate from the original experiment budget.','max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k] for k in ('status','research_decision','checks_total','unique_file_count','hashed_bytes','audit_wall_seconds')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True);args=parser.parse_args()
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    audit(args.folder)
