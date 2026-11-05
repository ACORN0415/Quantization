"""Read-only audit of registered execution and stored scalar results; no torch/GPU."""
from pathlib import Path
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    lock=json.loads((HERE/'REGISTRATION_LOCK.json').read_text())
    result=json.loads((HERE/'RESULT.json').read_text())
    selftest=json.loads((HERE/'SELFTEST.json').read_text())
    previous=json.loads((HERE.parent/'obq_capacity_codex/RESULT.json').read_text())
    assert all(sha(p)==h for p,h in lock['sha256'].items())
    assert sha(HERE/'SELFTEST.json')==lock['selftest_sha256']
    assert result['registration_sha256']==sha(HERE/'REGISTRATION_LOCK.json')
    assert result['prior_result_sha256']==sha(HERE.parent/'obq_capacity_codex/RESULT.json')
    assert selftest['passed'] and selftest['code_sha256']==sha(HERE/'diagnose_rotation.py')
    assert result['status']=='completed' and result['completed_layers']==2
    assert result['elapsed_seconds']<=1200 and result['max_rss_bytes']<=8*1024**3
    assert result['cpu_threads']==4 and result['cuda_visible']==''
    assert result['started_utc']>=lock['registered_utc']
    assert [r['layer'] for r in result['rows']]==[18,24]
    key_bytes=18*1568*12*128//2+18*98*12*128*2+6
    # V quantizes six 4680-token chunks independently, padding each to 4688.
    value_bytes=6*4688*12*128//2+6*293*12*128*2
    summary=[]
    for r in result['rows']:
        p=next(x for x in previous['rows'] if x['layer']==r['layer'])
        assert r['input_preserved'] and r['anchor_reproduction_passed']
        assert r['calibration']==p['calibration'] and r['evaluation']==p['evaluation']
        assert set(r['arms'])=={'U_FP32_RTN','UHP_FP32_RTN'}
        for name,item in r['prior_arms_reused'].items():assert item==p['arms'][name]
        obq=p['arms']['OBQ_FULL']['sse'];b0=p['arms']['B0_FP32_RTN']['sse']
        entry={'layer':r['layer'],'B0_FP32_RTN_sse':b0,'OBQ_FULL_sse':obq,'rotations':{}}
        for name,a in r['arms'].items():
            assert a['key_payload_bytes']==key_bytes and a['value_payload_bytes']==value_bytes
            assert a['packed_payload_bytes']==key_bytes+value_bytes==p['arms']['OBQ_FULL']['packed_payload_bytes']
            gain=1-obq/a['sse']
            assert gain==r['obq_gains_vs_rotations'][name]
            assert math.isclose(a['nse'],a['sse']/r['reference_energy'],rel_tol=1e-15)
            nq=a['noquant_rotation_error_against_k_reference']['sse']
            assert nq/b0==a['noquant_fraction_of_B0_FP32_RTN']<=.01
            entry['rotations'][name]={'sse':a['sse'],'OBQ_gain_percent':100*gain,
                'noquant_sse':nq,'noquant_percent_of_B0':100*nq/b0,
                'noquant_vs_cpu_native_floor':a['noquant_ratio_to_cpu_native_floor'],
                'rotation_persistent_bytes':a['rotation_persistent_bytes']}
        summary.append(entry)
    expected='continue_causal_feasibility' if all(a['OBQ_gain_percent']>=10 for x in summary for a in x['rotations'].values()) else 'not_continue'
    assert expected==result['decision']
    print(json.dumps({'passed':True,'audit_scope':'stored scalar recomputation + hashes; no raw tensor recalculation',
        'source_hash_checks':len(lock['sha256']),'code_sha256':sha(HERE/'diagnose_rotation.py'),
        'result_sha256':sha(HERE/'RESULT.json'),'registration_sha256':sha(HERE/'REGISTRATION_LOCK.json'),
        'summary':summary,'decision_recalculated':expected,'payload_bytes':key_bytes+value_bytes,
        'elapsed_seconds':result['elapsed_seconds'],'max_rss_GiB':result['max_rss_bytes']/1024**3,
        'new_directory_bytes':sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())},indent=2))

if __name__=='__main__':main()
