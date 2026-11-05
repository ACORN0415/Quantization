"""Immutable run_layer reuse for 28 not-yet-measured development layers; CPU only."""
import argparse
import copy
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'obq_capacity_codex'
spec=importlib.util.spec_from_file_location('frozen_capacity_all_layers',OLD/'diagnose_obq.py')
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
NEW_LAYERS=tuple(l for l in range(30) if l not in (18,24))
BASES=('B0_BF16_RTN','B0_FP32_RTN')
BEGIN=None


def decide(rows):
    assert len(rows)==30 and sorted(r['layer'] for r in rows)==list(range(30))
    sums={name:sum(r['arms'][name]['sse'] for r in rows) for name in BASES+('OBQ_FULL',)}
    assert all(math.isfinite(v) and v>=0 for v in sums.values())
    worsening=[];table=[];improved=0
    for r in sorted(rows,key=lambda x:x['layer']):
        o=r['arms']['OBQ_FULL']['sse']
        assert math.isfinite(o) and o>=0
        both=True;item={'layer':r['layer'],'measurement_origin':r.get('measurement_origin','synthetic'),
            'OBQ_FULL_sse':o,'OBQ_FULL_nse':r['arms']['OBQ_FULL']['nse'],'comparisons':{}}
        for base in BASES:
            b=r['arms'][base]['sse']
            assert math.isfinite(b) and b>=0
            bad=o>1.1*b
            both=both and o<b
            if bad:worsening.append({'layer':r['layer'],'baseline':base,'obq_sse':o,'baseline_sse':b})
            item['comparisons'][base]={'sse':b,'nse':r['arms'][base]['nse'],'absolute_sse_decrease':b-o,
                'fraction_decrease':1-o/b if b else None,'worsens_over_10_percent':bad}
        improved+=int(both);item['improves_both']=both;table.append(item)
    pooled={base:{'baseline_sse_sum':sums[base],'OBQ_FULL_sse_sum':sums['OBQ_FULL'],
        'fraction_decrease':1-sums['OBQ_FULL']/sums[base] if sums[base] else None,
        'passes_20_percent':sums['OBQ_FULL']<=.8*sums[base]} for base in BASES}
    passes=not worsening and all(p['passes_20_percent'] for p in pooled.values()) and improved>=24
    return {'decision':'continue_all_layer_risk_gate' if passes else 'not_safe_for_uniform_all_layer_use',
        'improved_both_count':improved,'worsens_over_10_percent':worsening,'pooled':pooled,'per_layer':table}


def source_refs():
    prior=json.loads((OLD/'REGISTRATION_LOCK.json').read_text())
    return list(map(Path,prior['sha256']))+[OLD/'REGISTRATION_LOCK.json',OLD/'SELFTEST.json',OLD/'RESULT.json',
        HERE/'REGISTRATION_DRAFT_KO.txt',Path(__file__)]+[
        old.DATA/f'p{p:02d}_s101/manifest.json' for p in (0,4)]


def verify_old():
    lock=json.loads((OLD/'REGISTRATION_LOCK.json').read_text())
    assert all(old.sha(p)==h for p,h in lock['sha256'].items())
    test=json.loads((OLD/'SELFTEST.json').read_text())
    assert test['passed'] and test['code_sha256']==old.sha(OLD/'diagnose_obq.py')
    previous=json.loads((OLD/'RESULT.json').read_text())
    assert previous['status']=='completed' and [r['layer'] for r in previous['rows']]==[18,24]
    return previous


def selftest():
    assert not old.torch.cuda.is_available()
    verify_old()
    rows=[{'layer':l,'arms':{name:{'sse':70. if name=='OBQ_FULL' else 100.,'nse':.01} for name in BASES+('OBQ_FULL',)}} for l in range(30)]
    assert decide(rows)['decision']=='continue_all_layer_risk_gate'
    bad=copy.deepcopy(rows);bad[0]['arms']['OBQ_FULL']['sse']=111.
    assert decide(bad)['decision']=='not_safe_for_uniform_all_layer_use'
    boundary=copy.deepcopy(rows);boundary[0]['arms']['OBQ_FULL']['sse']=110.
    assert not decide(boundary)['worsens_over_10_percent']
    few=copy.deepcopy(rows)
    for r in few[23:]:r['arms']['OBQ_FULL']['sse']=100.
    assert decide(few)['improved_both_count']==23 and decide(few)['decision']=='not_safe_for_uniform_all_layer_use'
    incomplete_rejected=False
    try:decide(rows[:-1])
    except AssertionError:incomplete_rejected=True
    assert incomplete_rejected
    zero=copy.deepcopy(rows)
    for name in BASES+('OBQ_FULL',):zero[0]['arms'][name]['sse']=0.
    assert decide(zero)['per_layer'][0]['comparisons'][BASES[0]]['fraction_decrease'] is None
    result={'passed':True,'code_sha256':old.sha(__file__),'new_layers':NEW_LAYERS,
        'old_registration_verified':True,'actual_capture_loads':0,
        'checks':['positive','over10percent_worsening','exact10percent_boundary','23_improved_insufficient','incomplete_rejected','zero_denominator_explicit']}
    old.save(HERE/'SELFTEST.json',result);print(json.dumps(result,indent=2))


def lock():
    p=HERE/'LOCK.json'
    if p.exists():raise RuntimeError('refuse overwrite lock')
    verify_old();st=json.loads((HERE/'SELFTEST.json').read_text())
    assert st['passed'] and st['code_sha256']==old.sha(__file__)
    old.save(p,{'registered_utc':time.time(),'sha256':{str(p):old.sha(p) for p in source_refs()},
        'selftest_sha256':old.sha(HERE/'SELFTEST.json'),'new_layers':NEW_LAYERS,'reused_layers':[18,24],
        'seconds_cap':600,'cpu_threads':4,'rss_bytes_cap':8*1024**3,'disk_bytes_cap':2*1024**3})


def alarm_handler(signum,frame):
    raise TimeoutError('registered 600-second total execution limit')


def guard():
    if time.monotonic()-BEGIN>600:raise TimeoutError('registered 600-second total execution limit')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:raise MemoryError('8GiB RSS cap')
    if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())>2*1024**3:raise OSError('2GiB new disk cap')


def run():
    global BEGIN
    out=HERE/'RESULT.json'
    if out.exists():raise RuntimeError('refuse overwrite result')
    lock=json.loads((HERE/'LOCK.json').read_text())
    assert all(old.sha(p)==h for p,h in lock['sha256'].items())
    assert old.sha(HERE/'SELFTEST.json')==lock['selftest_sha256']
    prior=verify_old();BEGIN=time.monotonic();old.RUN_START=BEGIN
    signal.signal(signal.SIGALRM,alarm_handler);signal.alarm(600)
    result={'status':'running','started_utc':time.time(),'lock_sha256':old.sha(HERE/'LOCK.json'),
        'prior_result_sha256':old.sha(OLD/'RESULT.json'),'cpu_threads':old.torch.get_num_threads(),
        'cuda_visible':os.environ['CUDA_VISIBLE_DEVICES'],'rows':[],'new_layers_completed':0}
    for row in prior['rows']:
        copied=copy.deepcopy(row);copied['measurement_origin']='reused_original_capacity';result['rows'].append(copied)
    old.save(out,result)
    try:
        for layer in NEW_LAYERS:
            guard();row=old.run_layer(layer)
            row['measurement_origin']='new_cpu_extension'
            result['rows'].append(row);result['new_layers_completed']+=1
            result['elapsed_so_far_seconds']=time.monotonic()-BEGIN
            old.save(out,result);gc.collect();guard()
            print('LAYER_DONE',layer,'count',result['new_layers_completed'],'seconds',row['seconds'],flush=True)
        assert result['new_layers_completed']==28
        result['judgement']=decide(result['rows']);result['status']='completed'
    except BaseException as exc:
        result.update(status='incomplete_or_invalid',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0)
        result.update(finished_utc=time.time(),elapsed_seconds=time.monotonic()-BEGIN,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            new_directory_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()))
        old.save(out,result)
    print(json.dumps({'status':result['status'],'new_layers_completed':result['new_layers_completed'],
        'decision':result['judgement']['decision'],'elapsed_seconds':result['elapsed_seconds']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['selftest','lock','run']);args=p.parse_args()
    {'selftest':selftest,'lock':lock,'run':run}[args.action]()
