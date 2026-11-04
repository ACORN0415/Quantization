"""A-3: close the unmeasured down-projection comparison against native BF16."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
import linear_w8a8_fused_stage1 as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'src/experiments/linear_w8a8_continuation'
DATA = Path('/home/acorn0415/sfq_data/linear_w8a8_continuation')
START = 1793802990.
DEADLINE = START + 1800
core.torch = torch
core.DEADLINE = DEADLINE


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    import torch._inductor.config as cfg
    for name in ('max_autotune','max_autotune_gemm','search_autotune_cache','force_fuse_int_mm_with_mul'):
        setattr(cfg, name, False)
    cfg.triton.cudagraphs = False
    path = OUT/'results.json'
    if path.exists(): raise RuntimeError('Refusing overwrite')
    r = {'status':'running','started_utc':time.time(),'registered_start_utc':START,
         'deadline_utc':DEADLINE,'model_forwards':0,'timing':[],
         'code_sha256':core.sha(__file__),'helper_sha256':core.sha(core.__file__),
         'registration_sha256':core.sha(OUT/'REGISTRATION_KO.txt')}
    try:
        core.check()
        pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        assert all(int(p)==os.getpid() for p in pids.split())
        previous=json.loads((ROOT/'src/experiments/linear_w8a8_fused_stage1/results.json').read_text())['shapes'][2]
        cap=previous['capture'];assert core.sha(cap['path'])==cap['sha256']
        old=previous['validation_artifact'];assert core.sha(old['path'])==old['sha256']
        sample=torch.load(cap['path'],map_location='cpu',weights_only=True)
        old_bank=torch.load(old['path'],map_location='cpu',weights_only=True)
        r.update(capture=cap,previous_validation=old,shape_MNK=sample['shape_MNK'])
        x,w,y=(sample[k].cuda() for k in ('x','w','y'))
        bias=sample['bias'].cuda()
        torch.cuda.synchronize();t=time.perf_counter();qw,sw=core.quant(w);wt=qw.t().contiguous();torch.cuda.synchronize()
        r['static_weight_prepare_s']=time.perf_counter()-t
        options={'max_autotune_gemm':True,'max_autotune_gemm_backends':'ATEN,TRITON','force_fuse_int_mm_with_mul':True,'triton.cudagraphs':False}
        r['options']=options
        fn=torch.compile(core.w8a8_fused,backend='inductor',fullgraph=True,dynamic=False,options=options)
        fns={'BF16':lambda:core.bf16(x,w,bias),'W8A8':lambda:fn(x,wt,sw,bias)}
        values={};r['first_call_s']={}
        for arm,f in fns.items():
            torch.cuda.synchronize();t=time.perf_counter();values[arm]=f();torch.cuda.synchronize();r['first_call_s'][arm]=time.perf_counter()-t
        qx,sx=core.quant(x.reshape(-1,x.shape[-1]));ref=qx.float()@wt.float();ref=ref*sx*sw.t()+bias.float()
        qmetric=core.metric(values['W8A8'].reshape_as(ref),ref);bmetric=core.metric(values['W8A8'],y)
        gates={'native_bytes':core.bytes_equal(values['BF16'],y),
               'candidate_matches_A2':core.bytes_equal(values['W8A8'].cpu(),old_bank['outputs']['W8A8_fused']),
               'quantized_relative':qmetric['rel_l2']<=.002,
               'quantized_elementwise':bool(((values['W8A8'].reshape_as(ref).float()-ref).abs()<=.02+.02*ref.abs()).all()),
               'bf16_relative':bmetric['rel_l2']<=.05,
               'finite_shape_dtype':bool(qmetric['finite'] and values['W8A8'].shape==y.shape and values['W8A8'].dtype==y.dtype)}
        r.update(validity=gates,quantized_metric=qmetric,bf16_metric=bmetric)
        artifact=DATA/'outputs.pt';torch.save({a:v.cpu() for a,v in values.items()},artifact)
        r['outputs']={'path':str(artifact),'sha256':core.sha(artifact)}
        core.save(path,r)
        if not all(gates.values()):r['status']='invalid';return
        del values,qx,sx,qw,ref,old_bank,y,sample
        gc.collect();torch.cuda.empty_cache()
        for f in fns.values():
            for _ in range(10):f()
        torch.cuda.synchronize()
        for i in range(12):
            order=['BF16','W8A8'] if i%2==0 else ['W8A8','BF16'];row={'round':i,'order':order}
            for arm in order:row[arm]=core.timed(fns[arm])
            r['timing'].append(row);core.save(path,r)
        d=np.array([100*(1-t['W8A8']['wall_ms']/t['BF16']['wall_ms']) for t in r['timing']]);rng=np.random.default_rng(20261104)
        lo,hi=np.quantile(np.median(rng.choice(d,(10000,12)),axis=1),[.025,.975]);med=float(np.median(d))
        r['comparison']={'reductions_percent':d.tolist(),'median_percent':med,'bootstrap_95ci_percent':[float(lo),float(hi)],'passes':bool(med>=5 and lo>0)}
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for arm,f in fns.items():
                with torch.profiler.record_function(arm):z=f();del z
        torch.cuda.synchronize();tr=DATA/'trace.json';prof.export_chrome_trace(str(tr));r['trace']={'path':str(tr),'sha256':core.sha(tr)}
        r['generated_code']=[{'path':str(p),'sha256':core.sha(p)} for p in sorted((DATA/'inductor').rglob('*.py')) if 'def call(args):' in p.read_text()]
        r['status']='completed'
    except BaseException as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        r['finished_utc']=time.time();core.save(path,r);print(r['status'],flush=True)


if __name__=='__main__':
    DATA.mkdir(parents=True,exist_ok=True)
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(DATA/'inductor')
    os.environ['TRITON_CACHE_DIR']=str(DATA/'triton')
    with torch.inference_mode():main()
