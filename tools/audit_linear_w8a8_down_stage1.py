"""Read-only CPU audit of A-3 and its prospective continuation calculation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import numpy as np
import torch
from audit_linear_w8a8_fused_stage1 import sha,eq,metrics
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/experiments/linear_w8a8_continuation'


def main():
    torch.set_num_threads(4)
    r=json.loads((OUT/'results.json').read_text())
    assert r['status']=='completed' and all(r['validity'].values())
    for f,k in [('tools/linear_w8a8_down_stage1.py','code_sha256'),('tools/linear_w8a8_fused_stage1.py','helper_sha256'),('src/experiments/linear_w8a8_continuation/REGISTRATION_KO.txt','registration_sha256')]:assert sha(ROOT/f)==r[k]
    assert r['finished_utc']<r['deadline_utc'] and r['finished_utc']-r['started_utc']<300
    tensors={}
    for k in ('capture','previous_validation','outputs'):
        e=r[k];assert sha(e['path'])==e['sha256'];tensors[k]=torch.load(e['path'],map_location='cpu',weights_only=True)
    y=tensors['outputs'];assert eq(y['BF16'],tensors['capture']['y'])
    assert eq(y['W8A8'],tensors['previous_validation']['outputs']['W8A8_fused'])
    m=metrics(y['W8A8'],y['BF16']);assert m['rel_l2']<=.05
    assert abs(m['rel_l2']-r['bf16_metric']['rel_l2'])<1e-6
    d=np.array([100*(1-t['W8A8']['wall_ms']/t['BF16']['wall_ms']) for t in r['timing']]);rng=np.random.default_rng(20261104)
    ci=np.quantile(np.median(rng.choice(d,(10000,12)),axis=1),[.025,.975])
    assert np.array_equal(ci,r['comparison']['bootstrap_95ci_percent'])
    assert np.array_equal(d,r['comparison']['reductions_percent'])
    assert float(np.median(d))==r['comparison']['median_percent']
    assert (np.median(d)>=5 and ci[0]>0)==r['comparison']['passes']
    for i,t in enumerate(r['timing']):assert t['order']==(['BF16','W8A8'] if i%2==0 else ['W8A8','BF16'])
    assert sha(r['trace']['path'])==r['trace']['sha256']
    events=json.loads(Path(r['trace']['path']).read_text())['traceEvents']
    kernels=[e['name'] for e in events if e.get('cat')=='kernel']
    assert len(kernels)==4 and any(n.startswith('triton_tem_') for n in kernels)
    assert not any('ampere_igemm' in n for n in kernels)
    for e in r['generated_code']:assert sha(e['path'])==e['sha256']
    ev=json.loads((ROOT/'src/experiments/linear_w8a8_stage1/linear_events.json').read_text())
    q=json.loads((ROOT/'src/experiments/linear_w8a8_fused_stage1/results.json').read_text())['shapes'][0]['comparisons']['BF16_eager']['median_percent']/100
    total=qs=downs=0.
    for e in ev:
        if e['forward'] not in range(1,5):continue
        if e['kind']=='forward':total+=e['gpu_ms']
        elif '.self_attn.q' in e['name']:qs+=e['gpu_ms']
        elif '.ffn.2' in e['name']:downs+=e['gpu_ms']
    projection=100*(qs*q+downs*np.median(d)/100)/total
    assert abs(projection-json.loads((OUT/'projection.json').read_text())['aggregate_projected_percent'])<1e-12
    print(json.dumps({'status':'passed','gpu_calls':0,'outputs_equal_to_captured_BF16_and_A2_W8A8':True,'vs_bf16_cpu_metrics':m,'timing_recomputed':True,'actual_kernel_names':kernels,'projection_recomputed_percent':projection,'scope':'projection is conditional estimate, not measured rollout speed'},indent=2))


if __name__=='__main__':main()
