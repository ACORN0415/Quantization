"""Read-only CPU audit and analysis of the completed calibration-choice record.

Print JSON to stdout. Never rerun GPU candidates or overwrite preserved evidence.
"""
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch

from moe_calibration_choice_core import ROOT, file_hash, tensor_hash

RECORD = ROOT/'src/experiments/moe_calibration_choice_stage1'
DATA = Path('/home/acorn0415/sfq_data/moe_calibration_choice_stage1')
PRIOR = Path('/home/acorn0415/sfq_data/moe_allocation_stage1')


def read(name):
    return json.loads((RECORD/name).read_text())


def choose(scores,reg):
    values=np.asarray(scores,dtype=np.float64)
    low=values.min()
    tolerance=reg['selection']['tie_atol']+reg['selection']['tie_rtol']*abs(low)
    return int(np.flatnonzero(values<=low+tolerance)[0])


def run():
    started=time.perf_counter()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg,result,selection=read('registration.json'),read('result.json'),read('selection.json')
    candidates,choice,evaluation=read('candidates.json'),read('choice_scores.json'),read('evaluation_scores.json')
    assert result['status']=='PASS' and len(candidates)==20
    assert result['failed_candidates']==0 and result['full_model_evaluations']==0
    assert not (RECORD/'failure.json').exists()
    assert len(choice)==80 and len(evaluation)==320
    assert selection['heldout_scores_evaluated_before_freeze']==0
    assert datetime.fromisoformat(selection['frozen_at_utc'])<datetime.fromisoformat(result['completed_at_utc'])
    assert datetime.fromisoformat(result['completed_at_utc'])<datetime.fromisoformat(reg['deadline_utc'])
    assert result['gpu_stage_wall_seconds']<590
    for key,limit in [('cuda_allocated_bytes',reg['caps']['cuda_allocated_bytes']),
                      ('process_peak_rss_bytes',reg['caps']['cpu_rss_bytes'])]:
        assert result['overall_process_peak'][key]<limit
    hashes={}
    for name,info in reg['source_files'].items():
        sha=file_hash(ROOT/name)
        assert sha==info['sha256'],name
        hashes[str(ROOT/name)]=sha
    for name,sha in reg['input_files'].items():
        assert file_hash(PRIOR/name)==sha,name
        hashes[str(PRIOR/name)]=sha
    for name,info in result['artifact_files'].items():
        p=DATA/name
        assert p.stat().st_size==info['bytes'] and file_hash(p)==info['sha256'],name
        hashes[str(p)]=info['sha256']
    block=torch.load(PRIOR/'block.pt',weights_only=True,map_location='cpu',mmap=True)
    docs={d['id']:d for d in torch.load(PRIOR/'captures.pt',weights_only=True,map_location='cpu',mmap=True)}
    refs=torch.load(DATA/'native_references.pt',weights_only=True,map_location='cpu')
    outputs=torch.load(DATA/'native_outputs.pt',weights_only=True,map_location='cpu')
    identity=read('identity.json')
    assert identity['status']=='PASS' and len(identity['rows'])==96
    for row in identity['rows']:
        assert row['bitwise_equal']
        assert tensor_hash(refs[row['expert'],row['document']])==row['reference_sha256']
    # Independent unpacking and FP32 reconstruction, without the vendor quantizer.
    scores={(r['expert'],r['index'],r['document']):r for r in choice+evaluation}
    max_abs=max_rel=max_native=0.
    payloads=[]
    count=0
    for candidate in candidates:
        e,j,label=candidate['expert'],candidate['index'],candidate['label']
        assert candidate['actual_path']==('RTN' if j==0 else 'GPTQ')
        assert candidate['automatic_fallback'] is False
        p=DATA/f'expert{e}_{label}.pt'
        assert file_hash(p)==candidate['archive_sha256']
        a=torch.load(p,weights_only=True,map_location='cpu')
        raw=a['packed'].numpy()
        codes=np.empty(a['shape'],dtype=np.float32)
        codes[:,0::2]=raw&15
        codes[:,1::2]=raw>>4
        scale=np.repeat(a['scale'].numpy(),128,axis=1)
        zero=np.repeat(a['zero'].numpy(),128,axis=1).astype(np.float32)
        q=torch.from_numpy((codes-zero)*scale)
        assert tensor_hash(q)==candidate['decoded_fp32_sha256']
        payload=sum(a[k].numel()*a[k].element_size() for k in ('packed','scale','zero'))
        assert payload==candidate['payload_bytes']==reg['format']['candidate_payload_bytes']
        payloads.append(payload)
        w=block['experts.gate_up_proj'][e,1024:].float()
        assert tensor_hash(w)==candidate['original_weight_sha256']
        if j:
            h=torch.load(DATA/f'hessian_{e}.pt',weights_only=True,map_location='cpu')
            assert tensor_hash(h)==candidate['original_hessian_sha256']
        difference=(q.double()-w.double()).T.contiguous()
        for name in reg['splits']['choice']+reg['splits']['evaluation']:
            d=docs[name]
            x=d['x'][(d['indices']==e).any(dim=1)].double()
            err=x@difference
            exact=float(err.square().sum())
            r=scores[e,j,name]
            delta=abs(exact-r['fp64_sse'])
            rel=delta/max(abs(exact),1e-30)
            max_abs=max(max_abs,delta);max_rel=max(max_rel,rel)
            # CPU/GPU FP64 sums use different accumulation orders.
            assert np.isclose(exact,r['fp64_sse'],rtol=1e-10,atol=1e-8),(e,j,name,exact,r)
            native=float((outputs[e,j,name].double()-refs[e,name].double()).square().sum())
            max_native=max(max_native,abs(native-r['native_bf16_sse']))
            assert np.isclose(native,r['native_bf16_sse'],rtol=1e-13,atol=1e-10)
            assert outputs[e,j,name].numel()==r['elements']
            count+=1
    checks=read('original_input_checks.json')
    assert len(checks)==20 and all(r['unchanged'] for r in checks)
    for r in checks:
        c=next(c for c in candidates if (c['expert'],c['label'])==(r['expert'],r['label']))
        assert r['weight_sha256']==c['original_weight_sha256']
        assert r['hessian_sha256']==c['original_hessian_sha256']
    experts=reg['experts']; labels=reg['candidates']; names=reg['splits']['evaluation']
    table=np.asarray([[sum(r['fp64_sse'] for r in choice if r['expert']==e and r['index']==j)
                       for j in range(5)] for e in experts])
    assert choose(table.sum(0),reg)==selection['common_index']
    assert [choose(r,reg) for r in table]==selection['expert_indices']
    denominator=np.asarray([sum(scores[e,0,n]['elements'] for e in experts) for n in names])
    policies={}
    arrays={}
    for name,indices in selection['policies'].items():
        fp64=np.asarray([sum(scores[e,j,n]['fp64_sse'] for e,j in zip(experts,indices)) for n in names])
        native=np.asarray([sum(scores[e,j,n]['native_bf16_sse'] for e,j in zip(experts,indices)) for n in names])
        arrays[name]=(fp64,native)
        policies[name]=dict(indices=indices,labels=[labels[j] for j in indices],
                            fp64_sse=float(fp64.sum()),fp64_mse=float(fp64.sum()/denominator.sum()),
                            native_bf16_sse=float(native.sum()),native_bf16_mse=float(native.sum()/denominator.sum()),
                            per_document_fp64_sse=fp64.tolist(),per_document_native_sse=native.tolist())
    rng=np.random.default_rng(reg['decision']['bootstrap_seed'])
    samples=rng.integers(0,len(names),(reg['decision']['bootstrap_document_resamples'],len(names)))
    comparisons={}
    target,target_native=arrays['expert_direct']
    for name in ['common_direct','fixed_rtn','fixed_gptq_0.01']:
        base,base_native=arrays[name]
        gain=float(1-target.sum()/base.sum())
        boot=1-target[samples].sum(1)/base[samples].sum(1)
        ci=np.quantile(boot,reg['decision']['confidence_interval']).tolist()
        native_gain=float(1-target_native.sum()/base_native.sum())
        native_diff=float(base_native.sum()-target_native.sum())
        tolerance=reg['validation']['native_tie_absolute_sse']+reg['validation']['native_tie_relative_sse']*float(base_native.sum())
        comparisons[name]=dict(relative_reduction=gain,document_bootstrap_95_interval=ci,
                              native_relative_reduction=native_gain,native_exceeds_tie=native_diff>tolerance,
                              registered_comparison_passes=gain>=.05 and ci[0]>0 and native_diff>tolerance,
                              identical_selected_indices=selection['policies'][name]==selection['policies']['expert_direct'])
    # Measured component accounting, not separately executed policy latency.
    ec=result['expert_costs']
    weight_access=result['weight_archive_open_seconds']+sum(r['weight_load_seconds'] for r in ec)
    fit_access=result['capture_archive_open_seconds']+sum(r['fit_input_transfer_seconds'] for r in ec)
    hessian=sum(r['hessian_seconds'] for r in ec)
    costs={}
    for name,indices in selection['policies'].items():
        direct=name.endswith('_direct')
        need_h=direct or any(indices)
        selected=candidates if direct else [r for r in candidates if r['index']==indices[experts.index(r['expert'])]]
        row=dict(weight_access_seconds=weight_access,
                 fit_access_seconds=fit_access if need_h else 0.,
                 hessian_seconds=hessian if need_h else 0.,
                 quantization_generation_seconds=sum(r['generate_seconds'] for r in selected),
                 packing_export_with_check_seconds=sum(r['pack_seconds'] for r in selected),
                 choice_input_transfer_seconds=sum(r['choice_input_transfer_seconds'] for r in candidates) if direct else 0.,
                 choice_fp64_evaluation_seconds=sum(r['choice_primary_seconds'] for r in candidates) if direct else 0.,
                 selection_reduction_seconds=selection['common_selection_seconds' if name=='common_direct' else 'expert_selection_seconds'] if direct else 0.)
        row['measured_component_sum_seconds']=sum(row.values())
        costs[name]=row
    events=[json.loads(s) for s in (RECORD/'events.jsonl').read_text().splitlines()]
    assert len([e for e in events if e['event']=='candidate_complete'])==20
    assert next(i for i,e in enumerate(events) if e['event']=='identity_passed')<next(i for i,e in enumerate(events) if e['event']=='candidate_complete')
    assert events[-1]['event']=='gpu_stage_complete'
    assert all(datetime.fromisoformat(e['utc'])<datetime.fromisoformat(reg['deadline_utc']) for e in events)
    disk=sum(p.stat().st_size for d in (DATA,RECORD) for p in d.rglob('*') if p.is_file())
    assert disk<reg['caps']['new_disk_bytes']
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',audit_scope='CPU replay of all FP64 projection scores; native SSE replay from saved GPU outputs; no GPU re-execution.',
                audit_seconds=time.perf_counter()-started,independent_decoded_candidates=len(candidates),
                checked_projection_scores=count,max_fp64_sse_absolute_difference=max_abs,
                max_fp64_sse_relative_difference=max_rel,max_saved_native_sse_difference=max_native,
                checked_identity_reference_hashes=len(identity['rows']),unchanged_original_input_records=len(checks),
                same_payload_bytes=payloads[0],total_policy_payload_bytes=4*payloads[0],
                heldout_documents=names,heldout_routed_elements=int(denominator.sum()),
                choice_table_fp64_sse=table.tolist(),policies=policies,comparisons=comparisons,
                registered_selection_benefit=all(c['registered_comparison_passes'] for c in comparisons.values()),
                costs=costs,cost_scope='Single-run synchronized component sums. Packing includes roundtrip check and CPU export. Native diagnostic/evaluation/archive overhead excluded. No isolated policy end-to-end latency or memory claim.',
                artifact_and_input_hashes=hashes,disk_bytes_at_audit=disk,
                audit_source_sha256=file_hash(Path(__file__)))


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
