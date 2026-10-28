"""Separate artifact/aggregation audit; not independent validation of the metrics."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=Path.home()/'sfq_data/policy_supplement'
BASE=DATA.parent/'policy_study'
OUT=ROOT/'src/experiments/policy_supplement_results'


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def strict_mean(values):
    return None if any(v is None for v in values) else statistics.mean(values)


def equal(a,b):
    return a is None and b is None or a is not None and b is not None and math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-10)


def check_interval(diffs, contrast):
    # Reuse the specified PCG64 draws, but independently sum resampled prompts
    # with Python arithmetic and interpolate percentile endpoints explicitly.
    draws=np.random.default_rng(872341).integers(0,10,(20000,10))
    boot=sorted(sum(diffs[int(i)] for i in row)/10 for row in draws)
    def percentile(q):
        position=q*(len(boot)-1);lo=math.floor(position);hi=math.ceil(position)
        return boot[lo]+(position-lo)*(boot[hi]-boot[lo])
    assert equal(percentile(.025),contrast['ci_low'])
    assert equal(percentile(.975),contrast['ci_high'])


def interval_mutation_test():
    diffs=list(range(10))
    draws=np.random.default_rng(872341).integers(0,10,(20000,10))
    lo,hi=np.quantile(np.asarray(diffs)[draws].mean(1),[.025,.975])
    check_interval(diffs,dict(ci_low=float(lo),ci_high=float(hi)))
    try:check_interval(diffs,dict(ci_low=float(lo)+.1,ci_high=float(hi)))
    except AssertionError:return
    raise RuntimeError('audit failed to detect altered confidence interval')


def main():
    if not __debug__:raise RuntimeError('Audit requires assertions; do not run Python with -O')
    interval_mutation_test()
    summary=read(OUT/'summary.json');gate=read(DATA/'gates.json')
    assert summary['provenance']==gate['provenance']
    assert sha(ROOT/'tools/evaluate_policy_supplement.py')==gate['provenance']['evaluator_sha256']
    assert sha(ROOT/'src/experiments/POLICY_STUDY_SUPPLEMENT.md')==gate['provenance']['protocol_sha256']
    assert len(summary['artifacts'])==260
    records=[];keys=set();empty=[]
    windows={'whole':(0,42),'early':(1,7),'middle':(7,21),'late':(21,42)}
    for a in summary['artifacts']:
        p=DATA/a['path'];assert sha(p)==a['sha256'];r=read(p)
        assert r['key'] not in keys;keys.add(r['key'])
        assert r['provenance']==gate['provenance']
        assert sha(BASE/r['path'])==r['video_sha256']==a['video_sha256']
        assert r['frames']==501 and len(r['pairs'])==84 and len(r['chunks'])==42
        assert sum(c['subject_transitions'] for c in r['chunks'])==500
        for pair in r['pairs']:
            assert 0<=pair['coverage']<=1
            assert (pair['warp_error'] is None)==(pair['coverage']==0)
            if pair['coverage']==0:empty.append(dict(video=r['key'],chunk=pair['chunk'],first_frame=pair['first_frame'],last_frame=pair['last_frame']))
        for name,(lo,hi) in windows.items():
            pairs=[v for v in r['pairs'] if lo<=v['chunk']<hi]
            chunks=[v for v in r['chunks'] if lo<=v['chunk']<hi]
            for metric in ('warp_error','coverage','flow_pixels','raw_difference'):
                assert equal(strict_mean([v[metric] for v in pairs]),r['windows'][name][metric])
            expected=sum(v['subject']*v['subject_transitions'] for v in chunks)/sum(v['subject_transitions'] for v in chunks)
            assert equal(expected,r['windows'][name]['subject'])
        records.append(r)
    counts=collections.Counter((r['bits'],r['policy'],r['arm']) for r in records)
    assert counts[(0,'reference','bf16')]==20 and len(counts)==13
    for bits in (4,2):
        for policy in ('rolling','chunk'):
            for arm in ('fr','tf','noise'):
                assert counts[(bits,policy,arm)]==20
                assert {(r['prompt'],r['seed']) for r in records if (r['bits'],r['policy'],r['arm'])==(bits,policy,arm)}=={(p,s) for p in range(10) for s in (101,202)}
    assert {(r['prompt'],r['seed']) for r in records if r['arm']=='bf16'}=={(p,s) for p in range(10) for s in (101,202)}
    for c in summary['contrasts']:
        selected=[r for r in records if r['bits']==c['bits'] and r['arm']==c['arm']]
        metric=c['metric'];window=c['window']
        if any(r['windows'][window][metric] is None for r in selected):
            assert c['mean'] is None and c['ci_low'] is None and c['ci_high'] is None
        else:
            diffs=[]
            for prompt in range(10):
                seeds=[]
                for seed in (101,202):
                    cell={r['policy']:r['windows'][window][metric] for r in selected if r['prompt']==prompt and r['seed']==seed}
                    seeds.append(cell['chunk']-cell['rolling'])
                diffs.append(statistics.mean(seeds))
            assert equal(statistics.mean(diffs),c['mean'])
            assert all(equal(a,b) for a,b in zip(diffs,c['prompt_differences']))
            check_interval(diffs,c)
    result=dict(passed=True,verified_video_records=260,verified_input_videos=260,
                pair_records=260*84,chunk_records=260*42,contrast_records=len(summary['contrasts']),
                empty_mask_pairs=empty,source_and_protocol_hashes_match=True,
                no_finite_only_contrasts=True,balanced_cells=True)
    result.update(bootstrap_intervals_recomputed=True,altered_interval_detected=True,
                  scope='Separate implementation of artifact and aggregation checks; not an external review or official metric differential test')
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('AUDIT PASS',len(records),'videos,',len(empty),'empty-mask pairs retained')


if __name__=='__main__':main()
