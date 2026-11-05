"""Read-only JSON/hash audit after synthetic pipeline; no attention recomputation."""
import hashlib,json,math,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
status=json.loads((HERE/'FULL_FIXTURE_STATUS.json').read_text());assert status['status']=='completed',status
result=json.loads((HERE/'RESULT.json').read_text());run=json.loads((HERE/'RUN.json').read_text());lock=json.loads((HERE/'LOCK.json').read_text());bindings=json.loads((HERE/'INPUT_BINDINGS_DRAFT.json').read_text());checks=[]
def ok(name,condition):
 if not condition:raise AssertionError(name)
 checks.append(name)
ok('68 cells',len(result['rows'])==68);ok('completed supervisor',run['status']=='completed');ok('36 completed sequential stages',len(run['stages'])==36 and all(x['exit_code']==0 for x in run['stages']));ok('20min original deadline',status['elapsed_seconds']<=1200)
expected={(f'causal_layer{l}_c7_call{c}',a) for l in (18,24) for c in (0,3) for a in lock['arms']}
ok('unique complete grid',{(r['endpoint'],r['arm']) for r in result['rows']}==expected)
for p,digest in lock['sources'].items():ok('source hash '+p,sha(p)==digest)
for layer in (18,24):
 prep=json.loads((HERE/f'PREPARE_{layer}.json').read_text());ok(f'prepare {layer}',prep['status']=='completed');ok(f'prepare state hash {layer}',sha(prep['state']['path'])==prep['state']['sha256'])
 for arm in lock['arms']:
  x=json.loads((HERE/f'RESULT_{layer}_{arm}.json').read_text());ok(f'worker {layer} {arm}',x['status']=='completed' and len(x['rows'])==2);ok(f'worker state hash {layer} {arm}',sha(x['state_artifact']['path'])==x['state_artifact']['sha256']);ok(f'RSS {layer} {arm}',x['peak_RSS_bytes']<=8*1024**3)
  for entry in x['source_loads']:ok('synthetic input origin '+entry['path'],Path(entry['path']).resolve().is_relative_to(HERE))
  for row in x['rows']:
   ok('valid cell '+row['endpoint']+arm,row['validity']=='valid' and math.isfinite(row['primary_Konly']['sse']) and math.isfinite(row['full_BF16_reference']['sse']))
   ok('fixed bytes '+row['endpoint']+arm,0<=row['extra_persistent_bytes']<=lock['dense_extra_bytes'][arm] and row['cap_eligible']==(arm not in ('FM3','FM8')))
   if arm.startswith('PAIR'):
    ok('fallback unit '+row['endpoint']+arm,row['fallback_pairs_total']==1536 and row['fallback_rate']==row['fallback_count']/1536)
    ok('denominator recorded '+row['endpoint']+arm,row['log_denominator_ratio_summary']['total_pairs']==1536)
    pm=x['paired_manifest'];ok('Omega binding '+row['endpoint']+arm,pm['layer']==layer and pm['seed']==int(arm[4:]) and len(pm['chunks'])==6 and all(z['omega_sha256']==pm['omega_sha256'] and z['layer']==layer for z in pm['chunks']))
comparisons=[]
for endpoint in sorted({r['endpoint'] for r in result['rows']}):
 rows={r['arm']:r for r in result['rows'] if r['endpoint']==endpoint};best=min((rows[a] for a in lock['primary_baselines']),key=lambda r:r['primary_Konly']['sse']);bs=best['primary_Konly']['sse']
 for seed in (17,29,43):
  r=rows['PAIR'+str(seed)];gain=1-r['primary_Konly']['sse']/bs if bs>0 else None;passed=bs>0 and gain>=.1 and r['fallback_count']/1536<=.001 and r['cap_eligible'];comparisons.append({'endpoint':endpoint,'seed':seed,'best_baseline':best['arm'],'gain':gain,'passed':passed})
ok('stored decision arithmetic',comparisons==result['comparisons']);decision='continue_local_feasibility' if all(x['passed'] for x in comparisons) else 'stop_fixed_Gaussian_rank32';ok('synthetic branch matches',decision==result['decision'])
auditfiles=list(HERE.glob('IO_AUDIT_*.json'));io=[json.loads(x.read_text()) for x in auditfiles];ok('no forbidden original .pt reads',all(not x['denied_external_tensor_paths'] for x in io));ok('original stage0 untouched',not (HERE.parent/'paired_response_stage0/LOCK.json').exists() and not (HERE.parent/'paired_response_stage0/RUN.json').exists())
report={'status':'passed','checks':len(checks),'check_names':checks,'scope':'read-only stored JSON, source/state artifact hashes, budget and decision arithmetic; no attention/data recomputation','real_capture_reads':0,'actual_method_effect_measurements':0,'GPU_calls':0,'synthetic_result_sha256':sha(HERE/'RESULT.json'),'status_sha256':sha(HERE/'FULL_FIXTURE_STATUS.json'),'maximum_worker_RSS_bytes':max(status['max_RSS_by_pid'].values()),'io_audit_files':len(io),'elapsed_from_fixture_start':time.time()-json.loads((HERE/'FIXTURE_REGISTRATION.json').read_text())['started_utc']}
with (HERE/'AUDIT_SYNTHETIC_FINAL.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:report[k] for k in ['status','checks','maximum_worker_RSS_bytes','real_capture_reads','actual_method_effect_measurements']}))
