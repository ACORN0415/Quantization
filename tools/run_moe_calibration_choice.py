"""Bounded projection-level GPTQ/RTN diagnostic; no full-model forward."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import threading
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from moe_calibration_choice_core import (ROOT, VENDOR, cpu_validation, file_hash,
                                         tensor_hash, same_bits, load_primitive)

RECORD = ROOT/'src/experiments/moe_calibration_choice_stage1'
DATA = Path('/home/acorn0415/sfq_data/moe_calibration_choice_stage1')
PRIOR = Path('/home/acorn0415/sfq_data/moe_allocation_stage1')
LABELS = ['rtn','gptq_0.01','gptq_0.05','gptq_0.1','gptq_1.0']
EXPERTS = [43,62,51,17]


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def event(name,**kwargs):
    row=dict(utc=datetime.now(timezone.utc).isoformat(),event=name,**kwargs)
    with (RECORD/'events.jsonl').open('a') as out:
        out.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def peak():
    return dict(cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


class Guard:
    def __init__(self,reg):
        self.reg=reg
        self.gpu_start=None
        self.done=threading.Event()
        threading.Thread(target=self.watch,daemon=True).start()

    def watch(self):
        deadline=datetime.fromisoformat(self.reg['deadline_utc']).timestamp()
        while not self.done.wait(.5):
            reason=None
            if time.time()>=deadline: reason='total wall cap'
            if self.gpu_start is not None and time.perf_counter()-self.gpu_start>=600: reason='GPU stage cap'
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30: reason='CPU RSS cap'
            if torch.cuda.is_initialized() and torch.cuda.memory_allocated()>8*2**30: reason='CUDA allocated cap'
            size=sum(p.stat().st_size for d in (DATA,RECORD) for p in d.rglob('*') if p.is_file())
            if size>512*2**20: reason='new disk cap'
            if reason:
                write(RECORD/'failure.json',dict(status='INCOMPLETE',reason=reason))
                os._exit(90)


def choose(scores):
    scores=np.asarray(scores,dtype=np.float64)
    low=float(scores.min())
    threshold=1e-16+1e-12*abs(low)
    return int(np.flatnonzero(scores<=low+threshold)[0])


def active_x(doc,expert):
    ids=doc['indices']
    mask=(ids==expert).any(dim=1)
    return doc['x'][mask].contiguous()


@torch.inference_mode()
def evaluate(X,Q,W,reference):
    if not len(X):
        return dict(elements=0,fp64_sse=0.,native_bf16_sse=0.,primary_seconds=0.,native_audit_seconds=0.),torch.empty((0,1024),dtype=torch.bfloat16)
    t=sync_time()
    diff=F.linear(X.double(),(Q.double()-W.double()))
    precise=float(diff.square().sum().item())
    primary_seconds=sync_time()-t
    t=sync_time()
    native=F.linear(X,Q.to(torch.bfloat16))
    actual=float((native.double()-reference.double()).square().sum().item())
    assert torch.isfinite(diff).all() and torch.isfinite(native).all()
    saved=native.cpu()
    native_seconds=sync_time()-t
    return dict(elements=int(native.numel()),fp64_sse=precise,native_bf16_sse=actual,
                primary_seconds=primary_seconds,native_audit_seconds=native_seconds),saved


def pack_candidate(utils,quant,Q,scales,zeros):
    assert scales.dtype==torch.float32 and zeros.dtype==torch.int32
    codes=quant.quantize(Q,scales,zeros)
    packed=utils.pack_to_uint8(codes,4)
    recovered=utils.unpack_from_uint8(packed,4,tuple(Q.shape))
    assert torch.equal(codes,recovered)
    decoded=quant.dequantize(recovered,scales,zeros)
    assert same_bits(Q,decoded)
    return dict(packed=packed.cpu(),scale=scales.cpu(),zero=zeros.cpu(),shape=tuple(Q.shape))


@torch.inference_mode()
def run(reg):
    guard=Guard(reg)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    utils,primitive=load_primitive()
    quant=utils.WeightQuantizer(bits=4,symmetric=True,group_size=128,signed=False)
    for name,info in reg['source_files'].items():
        assert file_hash(ROOT/name)==info['sha256']
    cpu=cpu_validation()
    write(RECORD/'cpu_validation.json',cpu)
    event('cpu_validation_passed',max_abs=cpu['max_abs_difference'])
    load_start=time.perf_counter()
    for name,sha in reg['input_files'].items():
        assert file_hash(PRIOR/name)==sha
    source_file_hash_seconds=time.perf_counter()-load_start
    t=time.perf_counter()
    block=torch.load(PRIOR/'block.pt',map_location='cpu',weights_only=True,mmap=True)
    weight_archive_open_seconds=time.perf_counter()-t
    t=time.perf_counter()
    captures=torch.load(PRIOR/'captures.pt',map_location='cpu',weights_only=True,mmap=True)
    capture_archive_open_seconds=time.perf_counter()-t
    docs={d['id']:d for d in captures}
    capture_hashes=json.loads((ROOT/'src/experiments/moe_allocation_stage1/capture_hashes.json').read_text())
    for row in capture_hashes:
        for field in ['x','indices']:
            assert tensor_hash(docs[row['id']][field])==row[field]
    input_hash_and_load_seconds=time.perf_counter()-load_start
    split_counts={}
    for e in EXPERTS:
        split_counts[e]={s:sum(len(active_x(docs[n],e)) for n in names) for s,names in reg['splits'].items()}
        assert split_counts[e]['fit']==reg['expected_fit_counts'][str(e)]
        assert split_counts[e]['choice']==reg['expected_choice_counts'][str(e)]
        assert all(n>0 for n in split_counts[e].values())
    write(RECORD/'input_check.json',dict(status='PASS',split_counts=split_counts,
          input_hash_and_load_seconds=input_hash_and_load_seconds))
    guard.gpu_start=time.perf_counter()
    torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    identity=[]
    refs={}
    identity_start=sync_time()
    # All no-intervention checks precede every real quantized candidate.
    for e in EXPERTS:
        fused=block['experts.gate_up_proj'][e].cuda()
        W=fused[1024:].contiguous()
        for doc in captures:
            x=active_x(doc,e).cuda()
            if len(x):
                ref=F.linear(x,fused).chunk(2,dim=-1)[1].contiguous()
                restored=F.linear(x,W)
                assert same_bits(ref,restored),(e,doc['id'],'identity failed')
            else:
                ref=torch.empty((0,1024),dtype=torch.bfloat16,device='cuda')
            refs[e,doc['id']]=ref.cpu()
            identity.append(dict(expert=e,document=doc['id'],tokens=len(x),bitwise_equal=True,
                                 reference_sha256=tensor_hash(ref)))
        del fused,W,x,ref
        if 'restored' in locals(): del restored
    identity_seconds=sync_time()-identity_start
    torch.save(refs,DATA/'native_references.pt')
    write(RECORD/'identity.json',dict(status='PASS',rows=identity,elapsed_seconds=identity_seconds,
                                    contract='Exact BF16 bytes: fused native gate/up upper slice vs separate up projection'))
    event('identity_passed',comparisons=len(identity),seconds=identity_seconds)
    candidates=[]
    choice_rows=[]
    expert_costs=[]
    native_outputs={}
    original_hash_checks=[]
    generation_wall_start=sync_time()
    for ei,e in enumerate(EXPERTS):
        t=sync_time()
        W=block['experts.gate_up_proj'][e,1024:].contiguous().cuda().float()
        weight_load_seconds=sync_time()-t
        whash=tensor_hash(W)
        H=None
        xs={}
        hseconds=xseconds=0.
        # RTN generation is measured before input/Hessian tensors reside on the GPU.
        order=[0]+([1,2,3,4] if ei%2==0 else [4,3,2,1])
        for index in order:
            label=LABELS[index]
            if index and H is None:
                t=sync_time()
                Xfit=torch.cat([active_x(docs[n],e) for n in reg['splits']['fit']]).cuda().float()
                xseconds=sync_time()-t
                t=sync_time()
                # Fixed normalization; scalar scaling does not change relative damping.
                H=(2./len(Xfit))*(Xfit.T@Xfit)
                hseconds=sync_time()-t
                assert torch.isfinite(H).all()
                hhash=tensor_hash(H)
                torch.save(H.cpu(),DATA/f'hessian_{e}.pt')
                del Xfit
            t=sync_time()
            try:
                if index==0:
                    work_w=W.clone()
                    assert work_w.data_ptr()!=W.data_ptr()
                    scales,zeros=quant.find_qparams(work_w)
                    Q=quant.fake_quantize(work_w,scales,zeros)
                    actual='RTN'
                else:
                    work_w,work_h=W.clone(),H.clone()
                    assert work_w.data_ptr()!=W.data_ptr() and work_h.data_ptr()!=H.data_ptr()
                    Q,scales,zeros=primitive(work_w,work_h,quant,128,float(label.split('_')[1]),False)
                    actual='GPTQ'
                torch.cuda.synchronize()
            except Exception as exc:
                event('candidate_failed',expert=e,label=label,actual_path='failed; no automatic RTN',error=str(exc))
                raise
            generate_seconds=time.perf_counter()-t
            t=sync_time()
            archive=pack_candidate(utils,quant,Q,scales,zeros)
            pack_seconds=sync_time()-t
            t=time.perf_counter()
            assert tensor_hash(W)==whash
            if H is not None: assert tensor_hash(H)==hhash
            original_hash_checks.append(dict(expert=e,label=label,weight_sha256=whash,
                                             hessian_sha256=hhash if H is not None else None,unchanged=True))
            audit_seconds=time.perf_counter()-t
            payload=sum(archive[k].numel()*archive[k].element_size() for k in ['packed','scale','zero'])
            assert payload==1179648
            artifact=DATA/f'expert{e}_{label}.pt'
            torch.save(archive,artifact)
            row=dict(expert=e,index=index,label=label,actual_path=actual,automatic_fallback=False,
                     generate_seconds=generate_seconds,pack_seconds=pack_seconds,audit_seconds=audit_seconds,
                     payload_bytes=payload,archive_bytes=artifact.stat().st_size,
                     archive_sha256=file_hash(artifact),decoded_fp32_sha256=tensor_hash(Q),
                     original_weight_sha256=whash,original_hessian_sha256=hhash if index else None)
            candidates.append(row)
            assert len(candidates)<=20
            t=sync_time()
            primary_evaluation_seconds=0.
            choice_transfer_seconds=0.
            for n in reg['splits']['choice']:
                if n not in xs:
                    transfer_start=sync_time()
                    xs[n]=active_x(docs[n],e).cuda()
                    choice_transfer_seconds+=sync_time()-transfer_start
                scores,native=evaluate(xs[n],Q,W,refs[e,n].cuda())
                primary_evaluation_seconds+=scores['primary_seconds']
                choice_rows.append(dict(expert=e,index=index,label=label,document=n,**scores))
                native_outputs[e,index,n]=native
            row['choice_evaluation_seconds']=sync_time()-t
            row['choice_primary_seconds']=primary_evaluation_seconds
            row['choice_input_transfer_seconds']=choice_transfer_seconds
            row['resident_state']='Current expert W; choice inputs; H only after first GPTQ. Overall process peaks only, not standalone policy peaks.'
            del work_w,Q,scales,zeros,archive
            if index: del work_h
            event('candidate_complete',expert=e,label=label,completed=len(candidates),total=20,
                  generate_seconds=generate_seconds,pack_seconds=pack_seconds)
            write(RECORD/'candidates.json',candidates)
        expert_costs.append(dict(expert=e,weight_load_seconds=weight_load_seconds,
                                 fit_input_transfer_seconds=xseconds,hessian_seconds=hseconds))
        del W,H,xs
    generation_seconds=sync_time()-generation_wall_start
    write(RECORD/'choice_scores.json',choice_rows)
    table=np.asarray([[sum(r['fp64_sse'] for r in choice_rows if r['expert']==e and r['index']==j)
                       for j in range(5)] for e in EXPERTS])
    t=time.perf_counter()
    common=choose(table.sum(axis=0))
    common_selection_seconds=time.perf_counter()-t
    t=time.perf_counter()
    individual=[choose(row) for row in table]
    individual_selection_seconds=time.perf_counter()-t
    policies={'fixed_rtn':[0]*4,**{f'fixed_{LABELS[i]}':[i]*4 for i in range(1,5)},
              'common_direct':[common]*4,'expert_direct':individual}
    frozen=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),criterion='choice FP64 SSE',
                policies=policies,common_index=common,expert_indices=individual,
                aliases={'coverage_0.5_percent':'fixed_gptq_0.01','coverage_plus_2K':'fixed_rtn'},
                common_selection_seconds=common_selection_seconds,
                expert_selection_seconds=individual_selection_seconds,
                heldout_scores_evaluated_before_freeze=0)
    write(RECORD/'selection.json',frozen)
    event('selection_frozen',common=LABELS[common],individual=[LABELS[i] for i in individual])
    eval_rows=[]
    evaluation_start=sync_time()
    for e in EXPERTS:
        W=block['experts.gate_up_proj'][e,1024:].contiguous().cuda().float()
        for index,label in enumerate(LABELS):
            a=torch.load(DATA/f'expert{e}_{label}.pt',map_location='cpu',weights_only=True)
            codes=utils.unpack_from_uint8(a['packed'].cuda(),4,a['shape'])
            Q=quant.dequantize(codes,a['scale'].cuda(),a['zero'].cuda())
            for n in reg['splits']['evaluation']:
                x=active_x(docs[n],e).cuda()
                scores,native=evaluate(x,Q,W,refs[e,n].cuda())
                eval_rows.append(dict(expert=e,index=index,label=label,document=n,**scores))
                native_outputs[e,index,n]=native
            del Q,codes,a,x
        del W
    evaluation_seconds=sync_time()-evaluation_start
    write(RECORD/'evaluation_scores.json',eval_rows)
    torch.save(native_outputs,DATA/'native_outputs.pt')
    write(RECORD/'original_input_checks.json',original_hash_checks)
    gpu_seconds=sync_time()-guard.gpu_start
    gpu_peaks=peak()
    guard.gpu_start=None
    assert gpu_seconds<600 and len(candidates)==20
    result=dict(status='PASS',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                gpu_stage_wall_seconds=gpu_seconds,valid_model_candidates=len(candidates),failed_candidates=0,
                model_forwards=0,full_model_evaluations=0,source_scope='Pinned primitive and quantizer only; not full Olive/GPTQModel',
                environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()),
                expert_costs=expert_costs,input_hash_and_load_seconds=input_hash_and_load_seconds,
                source_file_hash_seconds=source_file_hash_seconds,
                weight_archive_open_seconds=weight_archive_open_seconds,
                capture_archive_open_seconds=capture_archive_open_seconds,
                identity_seconds=identity_seconds,generation_and_choice_phase_seconds=generation_seconds,
                heldout_evaluation_seconds=evaluation_seconds,overall_process_peak=gpu_peaks,
                baseline_memory_note='No independently executed full-policy peak comparisons. Process peak covers all diagnostic phases.',
                artifact_files={p.name:dict(bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(DATA.glob('*.pt'))})
    write(RECORD/'result.json',result)
    guard.done.set()
    event('gpu_stage_complete',seconds=gpu_seconds,candidates=len(candidates),peak=gpu_peaks)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    if args.cpu_only:
        print(json.dumps(cpu_validation(),indent=2));return
    reg=json.loads((RECORD/'registration.json').read_text())
    assert datetime.now(timezone.utc)<datetime.fromisoformat(reg['deadline_utc'])
    assert not (RECORD/'result.json').exists(), 'Do not overwrite a completed run'
    DATA.mkdir(parents=True,exist_ok=True)
    assert not list(DATA.iterdir()),'Do not mix with old artifacts'
    try:
        run(reg)
    except Exception:
        write(RECORD/'failure.json',dict(status='FAILED',traceback=traceback.format_exc(),
                                       completed_at_utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__=='__main__':
    main()
