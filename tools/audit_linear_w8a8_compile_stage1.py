"""Read-only CPU audit of A-1 saved outputs, integers, timing and generated code."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/experiments/linear_w8a8_compile_stage1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def eq(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def main():
    torch.set_num_threads(4)
    r=json.loads((OUT/'results.json').read_text())
    assert r['status']=='completed' and r['model_forwards']==0
    assert r['finished_utc']<r['deadline_utc']
    assert sha(ROOT/'tools/linear_w8a8_compile_stage1.py')==r['code_sha256']
    assert sha(OUT/'REGISTRATION_KO.txt')==r['registration_sha256']
    rows=[]
    for s in r['shapes']:
        assert s['status']=='completed' and s['all_valid']
        assert s['finished_utc']-s['started_utc']<300
        cap=s['capture'];raw=s['validation_artifact']
        assert sha(cap['path'])==cap['sha256'] and sha(raw['path'])==raw['sha256']
        inp=torch.load(cap['path'],map_location='cpu',weights_only=True)
        bank=torch.load(raw['path'],map_location='cpu',weights_only=True)
        assert eq(bank['outputs']['BF16_eager'],inp['y'])
        assert eq(bank['outputs']['BF16_compiled'],inp['y'])
        exact=bank['qx'].double()@bank['qw'].double().t()
        assert torch.equal(exact,bank['int32_eager'].double())
        assert eq(bank['int32_eager'],bank['int32_compiled'])
        qdiff=int((bank['qx']!=bank['isolated_compiled_qx']).sum())
        assert qdiff==s['isolated_compiled_quant']['int8_different_elements']
        ref=exact.float()*bank['sx']*bank['sw'].t()
        if inp['bias'] is not None:ref=ref+inp['bias'].float()
        metrics={}
        for arm in ('W8A8_eager','W8A8_compiled'):
            y=bank['outputs'][arm];yf=y.reshape_as(ref).float()
            qrel=float((yf-ref).double().norm()/ref.double().norm())
            brel=float((y.float()-inp['y'].float()).double().norm()/inp['y'].double().norm())
            assert qrel<=.002 and brel<=.05 and torch.isfinite(y).all()
            assert bool(((yf-ref).abs()<=.02+.02*ref.abs()).all())
            assert abs(brel-s['metrics'][arm]['vs_bf16']['rel_l2'])<1e-6
            metrics[arm]={'cpu_fp64_vs_quantized_rel_l2':qrel,'cpu_fp64_vs_bf16_rel_l2':brel}
        assert len(s['timing'])==12
        arms=['BF16_eager','BF16_compiled','W8A8_eager','W8A8_compiled']
        for i,t in enumerate(s['timing']):assert t['order']==arms[i%4:]+arms[:i%4]
        for base in ('BF16_eager','BF16_compiled'):
            d=np.array([100*(1-t['W8A8_compiled']['wall_ms']/t[base]['wall_ms']) for t in s['timing']])
            rng=np.random.default_rng(20261104)
            ci=np.quantile(np.median(rng.choice(d,(10000,12)),axis=1),[.025,.975])
            assert np.array_equal(d,s['comparisons'][base]['reductions_percent'])
            assert np.array_equal(ci,s['comparisons'][base]['bootstrap_median_95ci_percent'])
            assert float(np.median(d))==s['comparisons'][base]['median_percent']
            assert not s['comparisons'][base]['passes']
        assert not s['promising']
        assert sha(s['trace']['path'])==s['trace']['sha256']
        sources=[]
        for c in s['generated_code']:
            assert sha(c['path'])==c['sha256']
            text=Path(c['path']).read_text()
            if c['aten_int_mm'] and c['triton_definitions']==2:
                assert 'extern_kernels._int_mm' in text
                assert 'torch.int32' in text
                sources.append(c['path'])
        assert len(sources)==1
        rows.append({'shape_MNK':s['shape_MNK'],'full_int32_cpu_reference_equal':True,
                     'native_bf16_bytes_equal':True,'quant_diff_count':qdiff,
                     'output_metrics':metrics,'timing_and_decision_recomputed':True,
                     'whole_candidate_generated_source':sources[0],
                     'implementation':'two generated Triton pointwise/reduction kernels around external _int_mm; INT32 buffer exists'})
    print(json.dumps({'status':'passed','gpu_calls':0,'shapes':rows},indent=2))


if __name__=='__main__':main()
