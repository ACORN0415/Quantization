"""Read-only saved-record audit. No GPU, torch import, capture replay, or writes."""
import hashlib,json,math
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'RESULT.json').read_text())
l=json.loads((P/'REGISTRATION_LOCK.json').read_text())
s=json.loads((P/'SELFTEST.json').read_text())
assert r['status']=='completed' and r['completed_layers']==2
assert l['registered_utc'] < r['started_utc'] < r['finished_utc']
assert s['time_utc'] < l['registered_utc'] and s['passed']
assert r['registration_sha256']==sha(P/'REGISTRATION_LOCK.json')
for p,h in l['sha256'].items(): assert sha(Path(p))==h,p
assert s['code_sha256']==sha(P/'diagnose_obq.py')
assert l['q_indices']==[i*4680//128 for i in range(128)]
assert r['cpu_threads']==4 and r['cuda_visible']==''
assert r['elapsed_seconds']<1200 and r['max_rss_bytes']<8*1024**3
assert abs((r['finished_utc']-r['started_utc'])-r['elapsed_seconds']) < .01
checks=[]
for row in r['rows']:
    a=row['arms']; b=a['B0_FP32_RTN']; d=a['OBQ_DIAG']; f=a['OBQ_FULL']
    assert row['source_tensors_preserved']
    assert len({x['packed_payload_bytes'] for x in a.values()})==1
    assert len({x['scale_sha256'] for x in a.values()})==1
    for k in ['code_sha256','sse','nse','logit_mse','past_key_sse']: assert b[k]==d[k],k
    for x in a.values():
        assert x['key_payload_bytes']+x['value_payload_bytes']==x['packed_payload_bytes']
        assert math.isclose(x['nse'],x['sse']/row['reference_energy'],rel_tol=1e-14)
    gain={k:1-f['sse']/a[k]['sse'] for k in ['B0_BF16_RTN','B0_FP32_RTN']}
    recorded=next(x for x in r['gains'] if x['layer']==row['layer'])
    for k,v in gain.items(): assert math.isclose(v,recorded[k],abs_tol=1e-15)
    checks.append({'layer':row['layer'],'gains':gain,'payload_bytes':f['packed_payload_bytes'],
      'logit_mse_reduction_vs_native':1-f['logit_mse']/a['B0_BF16_RTN']['logit_mse'],
      'key_sse_increase_vs_native':f['past_key_sse']/a['B0_BF16_RTN']['past_key_sse']-1,
      'diagonal_equals_fp32_rtn':True})
passed=all(v>=.2 for x in checks for v in x['gains'].values())
failed=any(v<=-.1 for x in checks for v in x['gains'].values())
expected='proceed_causal_capture' if passed else ('stop_this_instance' if failed else 'undecided')
assert r['decision']==expected
files=sorted(p for p in P.iterdir() if p.is_file())
assert sum(p.stat().st_size for p in files)<2*1024**3
print(json.dumps({'passed':True,'scope':'saved-record consistency and current code/registration hashes; no attention/capture recomputation',
'checks':checks,'decision':expected,'elapsed_seconds':r['elapsed_seconds'],'max_rss_bytes':r['max_rss_bytes'],
'result_sha256':sha(P/'RESULT.json'),'registration_lock_sha256':sha(P/'REGISTRATION_LOCK.json'),
'folder_bytes_at_audit':sum(p.stat().st_size for p in files)},ensure_ascii=False,indent=2))
