"""Independent scalar + byte/hash audit, no torch/GPU and no result mutation."""
from pathlib import Path
import hashlib
import json
import math
import time

HERE=Path(__file__).resolve().parent

def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()

def main():
    begin=time.monotonic()
    result=json.loads((HERE/'RESULT.json').read_text())
    lock=json.loads((HERE/'LOCK.json').read_text())
    prior_path=HERE.parent/'obq_capacity_codex/RESULT.json'
    previous=json.loads(prior_path.read_text())
    assert result['status']=='completed' and result['new_layers_completed']==28
    assert len(result['rows'])==30 and sorted(x['layer'] for x in result['rows'])==list(range(30))
    assert all(sha(p)==h for p,h in lock['sha256'].items())
    assert sha(HERE/'SELFTEST.json')==lock['selftest_sha256']
    assert sha(HERE/'LOCK.json')==result['lock_sha256']
    assert sha(prior_path)==result['prior_result_sha256']
    assert result['started_utc']>=lock['registered_utc']
    assert result['elapsed_seconds']<=600 and result['max_rss_bytes']<=8*1024**3
    assert result['cpu_threads']==4 and result['cuda_visible']==''
    bases=('B0_BF16_RTN','B0_FP32_RTN')
    sums={a:math.fsum(r['arms'][a]['sse'] for r in result['rows']) for a in bases+('OBQ_FULL',)}
    expected_k=18*1568*12*128//2+18*98*12*128*2+6
    expected_v=6*4688*12*128//2+6*293*12*128*2
    independent=[];bad=[];improved=0;raw_hashes=[]
    for r in sorted(result['rows'],key=lambda x:x['layer']):
        assert r['source_tensors_preserved']
        assert r['arms']['OBQ_DIAG']['code_sha256']==r['arms']['B0_FP32_RTN']['code_sha256']
        assert r['arms']['OBQ_DIAG']['scale_sha256']==r['arms']['B0_FP32_RTN']['scale_sha256']
        assert r['arms']['OBQ_DIAG']['sse']==r['arms']['B0_FP32_RTN']['sse']
        for a,d in r['arms'].items():
            assert d['key_payload_bytes']==expected_k and d['value_payload_bytes']==expected_v
            assert d['packed_payload_bytes']==expected_k+expected_v
            assert math.isfinite(d['sse']) and d['sse']>=0
            assert math.isclose(d['nse'],d['sse']/r['reference_energy'],rel_tol=1e-12,abs_tol=1e-15)
            assert d['scale_sha256']==r['arms']['B0_FP32_RTN']['scale_sha256']
        if r['layer'] in (18,24):
            assert r['measurement_origin']=='reused_original_capacity'
            stripped=dict(r);stripped.pop('measurement_origin')
            assert stripped==next(x for x in previous['rows'] if x['layer']==r['layer'])
        else:assert r['measurement_origin']=='new_cpu_extension'
        for key in ('calibration','evaluation'):
            item=r[key];path=Path(item['path'])
            digest=sha(path)
            assert item['manifest_verified'] and digest==item['sha256']
            manifest=json.loads((path.parent/'manifest.json').read_text())
            entry=next(x for x in manifest['artifacts'] if Path(x['path']).name==path.name)
            assert digest==entry['sha256'] and path.stat().st_size==entry['bytes']
            raw_hashes.append({'layer':r['layer'],'role':key,'sha256':digest,'bytes':path.stat().st_size})
        o=r['arms']['OBQ_FULL']['sse']
        both=all(o<r['arms'][b]['sse'] for b in bases);improved+=both
        row={'layer':r['layer'],'origin':r['measurement_origin'],'OBQ_FULL_sse':o,
            'OBQ_FULL_nse':r['arms']['OBQ_FULL']['nse'],'improves_both':both,
            'OBQ_SSE_share_percent':100*o/sums['OBQ_FULL'] if sums['OBQ_FULL'] else None,'comparisons':{}}
        for b in bases:
            bs=r['arms'][b]['sse'];worse=o>bs*1.1
            if worse:bad.append({'layer':r['layer'],'baseline':b})
            row['comparisons'][b]={'sse':bs,'nse':r['arms'][b]['nse'],
                'decrease_percent':100*(1-o/bs) if bs else None,'absolute_sse_decrease':bs-o,
                'SSE_share_percent':100*bs/sums[b] if sums[b] else None,'worsens_over_10_percent':worse}
        independent.append(row)
    pooled={b:{'baseline_sse_sum':sums[b],'OBQ_FULL_sse_sum':sums['OBQ_FULL'],
        'decrease_percent':100*(1-sums['OBQ_FULL']/sums[b]) if sums[b] else None,
        'passes_20_percent':sums['OBQ_FULL']<=.8*sums[b]} for b in bases}
    passes=not bad and all(x['passes_20_percent'] for x in pooled.values()) and improved>=24
    decision='continue_all_layer_risk_gate' if passes else 'not_safe_for_uniform_all_layer_use'
    recorded=result['judgement']
    assert decision==recorded['decision'] and improved==recorded['improved_both_count']
    assert bad==[{'layer':x['layer'],'baseline':x['baseline']} for x in recorded['worsens_over_10_percent']]
    for b in bases:
        for field in ('baseline_sse_sum','OBQ_FULL_sse_sum'):
            assert math.isclose(pooled[b][field],recorded['pooled'][b][field],rel_tol=1e-12,abs_tol=1e-9)
    print(json.dumps({'passed':True,'scope':'independent scalar decision + 60 source-file hashes; no attention rerun',
        'locked_file_hash_count':len(lock['sha256']),'raw_file_hash_count':len(raw_hashes),
        'raw_hashes':raw_hashes,'result_sha256':sha(HERE/'RESULT.json'),'lock_sha256':sha(HERE/'LOCK.json'),
        'code_sha256':sha(HERE/'run_all_layers.py'),'decision':decision,'improved_both_count':improved,
        'worsens_over_10_percent':bad,'pooled':pooled,'per_layer':independent,
        'packed_payload_bytes':expected_k+expected_v,'execution_seconds':result['elapsed_seconds'],
        'max_rss_GiB':result['max_rss_bytes']/1024**3,'audit_seconds':time.monotonic()-begin},indent=2))

if __name__=='__main__':main()
