"""A-1: fixed default compiler, three saved inputs, no model invocation."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'src/experiments/linear_w8a8_compile_stage1'
DATA = Path('/home/acorn0415/sfq_data/linear_w8a8_compile_stage1')
PREVIOUS = ROOT / 'src/experiments/linear_w8a8_stage1/results.json'
START = 1793800369.0  # 2026-11-04 13:52:49 UTC
DEADLINE = START + 1800


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*2**20), b''): h.update(b)
    return h.hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def check():
    if time.time() >= DEADLINE: raise TimeoutError('A-1 registered deadline')


def quant(x):
    f = x.float()
    scale = f.abs().amax(-1, keepdim=True).clamp_min(1e-12)/127.
    return torch.round(f/scale).clamp(-127,127).to(torch.int8), scale


def bf16(x, w, bias):
    return torch.nn.functional.linear(x, w, bias)


def intmm(a, b):
    return torch._int_mm(a, b)


def w8a8(x, wt, sw, bias):
    qx, sx = quant(x.reshape(-1, x.shape[-1]))
    acc = torch._int_mm(qx, wt)
    out = acc.float()*sx*sw.t()
    if bias is not None: out = out + bias.float()
    return out.to(torch.bfloat16).reshape(*x.shape[:-1], wt.shape[-1])


def dequant(acc, sx, sw, bias):
    out = acc.float()*sx*sw.t()
    if bias is not None: out = out + bias.float()
    return out.bfloat16()


def metric(x, ref):
    d = x.float()-ref.float()
    return {'rel_l2': float(d.norm()/ref.float().norm().clamp_min(1e-30)),
            'max_abs': float(d.abs().max()), 'finite': bool(torch.isfinite(x).all())}


def bytes_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def timed(fn, repetitions=10):
    check(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start = time.perf_counter(); a.record()
    for _ in range(repetitions):
        value = fn(); del value
    b.record(); torch.cuda.synchronize()
    return {'wall_ms': (time.perf_counter()-start)*1000/repetitions,
            'gpu_ms': a.elapsed_time(b)/repetitions, 'common_resident_bytes': resident,
            'incremental_peak_bytes': torch.cuda.max_memory_allocated()-resident}


def child(index):
    global torch
    import torch
    import numpy as np
    import torch._inductor.config as cfg
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name in ('max_autotune','max_autotune_gemm','search_autotune_cache','force_fuse_int_mm_with_mul'):
        setattr(cfg,name,False)
    cfg.epilogue_fusion = True
    cfg.triton.cudagraphs = False
    folder = DATA / f'shape{index}'; folder.mkdir(parents=True,exist_ok=True)
    path = OUT / f'shape{index}.json'
    if path.exists(): raise RuntimeError('Refusing overwrite')
    r = {'index': index, 'status': 'running', 'started_utc': time.time(), 'validity': {},
         'compile_first_call_s': {}, 'timing': [], 'model_forwards': 0,
         'compiler_config': {n:getattr(cfg,n) for n in ('max_autotune','max_autotune_gemm','search_autotune_cache','force_fuse_int_mm_with_mul','epilogue_fusion')}}

    def phase(name):
        check();r['phase']=name;save(path,r);print(index,name,flush=True)

    try:
        phase('load_and_verify')
        cap = json.loads(PREVIOUS.read_text())['captures'][index]
        assert sha(cap['path']) == cap['sha256']
        sample = torch.load(cap['path'],map_location='cpu',weights_only=True)
        x,w,y = (sample[k].cuda() for k in ('x','w','y'))
        bias = sample['bias'].cuda() if sample['bias'] is not None else None
        m,n,k = sample['shape_MNK']
        r.update(shape_MNK=[m,n,k], module=sample['name'], capture=cap,
                 gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
        ps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        assert all(int(pid)==os.getpid() for pid in ps.split()), 'GPU has other compute processes'
        torch.cuda.synchronize(); t = time.perf_counter()
        qw,sw = quant(w); wt = qw.t().contiguous()
        torch.cuda.synchronize(); r['static_weight_prepare_s']=time.perf_counter()-t
        bfc = torch.compile(bf16,backend='inductor',fullgraph=True,dynamic=False)
        qc = torch.compile(w8a8,backend='inductor',fullgraph=True,dynamic=False)
        ic = torch.compile(intmm,backend='inductor',fullgraph=True,dynamic=False)
        xqc = torch.compile(quant,backend='inductor',fullgraph=True,dynamic=False)
        fns = {'BF16_eager': lambda:bf16(x,w,bias), 'BF16_compiled':lambda:bfc(x,w,bias),
               'W8A8_eager':lambda:w8a8(x,wt,sw,bias), 'W8A8_compiled':lambda:qc(x,wt,sw,bias)}
        values = {}
        for arm,fn in fns.items():
            phase('first_call_'+arm)
            torch.cuda.synchronize();t=time.perf_counter();values[arm]=fn();torch.cuda.synchronize()
            r['compile_first_call_s'][arm]=time.perf_counter()-t
            save(path,r)
        phase('numerical_validation')
        qx,sx=quant(x.reshape(m,k));acc=intmm(qx,wt)
        t=time.perf_counter();acc_c=ic(qx,wt);torch.cuda.synchronize()
        r['integer_validation_compile_s']=time.perf_counter()-t
        t=time.perf_counter();qx_c,sx_c=xqc(x.reshape(m,k));torch.cuda.synchronize()
        r['quant_validation_compile_s']=time.perf_counter()-t
        ref=qx.float()@wt.float();ref=ref*sx*sw.t()
        if bias is not None: ref=ref+bias.float()
        rows=sorted(set(list(range(min(4,m)))+list(range(max(0,m-4),m))))
        integer_ref=qx[rows].cpu().long()@qw[:16].cpu().long().t()
        valid=r['validity']
        valid['native_replay_bytes_equal']=bytes_equal(values['BF16_eager'],y)
        valid['compiled_bf16_bytes_equal']=bytes_equal(values['BF16_compiled'],values['BF16_eager'])
        valid['compiled_int32_all_equal']=bytes_equal(acc,acc_c)
        valid['int64_sample_equal']=torch.equal(integer_ref,acc[rows,:16].cpu().long())
        valid['int_range_and_bound']=bool(qx.int().abs().max()<=127 and qw.int().abs().max()<=127 and k*127**2<2**31)
        r['isolated_compiled_quant']={'int8_different_elements':int((qx_c!=qx).sum()),
                                      'scale_max_abs':float((sx_c-sx).abs().max()),
                                      'scope':'standalone compiled quant; not an observation inside the full compiled invocation'}
        r['metrics']={}
        for arm in ('W8A8_eager','W8A8_compiled'):
            v=values[arm];qm=metric(v.reshape(m,n),ref);bm=metric(v,y)
            pointwise=bool(((v.reshape(m,n).float()-ref).abs()<=.02+.02*ref.abs()).all())
            r['metrics'][arm]={'vs_quantized_fp32':qm,'vs_bf16':bm,'pointwise_valid':pointwise,
                              'shape_valid':v.shape==y.shape,'dtype_valid':v.dtype==torch.bfloat16}
            valid[arm]=bool(qm['finite'] and qm['rel_l2']<=.002 and pointwise and bm['rel_l2']<=.05
                            and v.shape==y.shape and v.dtype==torch.bfloat16)
        r['compiled_vs_eager_w8a8']=metric(values['W8A8_compiled'],values['W8A8_eager'])
        pack={'outputs':{a:v.cpu() for a,v in values.items()},'qx':qx.cpu(),'sx':sx.cpu(),
              'qw':qw.cpu(),'sw':sw.cpu(),'int32_eager':acc.cpu(),'int32_compiled':acc_c.cpu(),
              'isolated_compiled_qx':qx_c.cpu(),'isolated_compiled_sx':sx_c.cpu()}
        datafile=folder/'validation.pt';torch.save(pack,datafile)
        r['validation_artifact']={'path':str(datafile),'sha256':sha(datafile),'bytes':datafile.stat().st_size}
        r['all_valid']=all(valid.values());save(path,r)
        if not r['all_valid']:
            r['status']='invalid';return
        del values,pack,qx,sx,qw,qx_c,sx_c,acc,acc_c,ref,integer_ref,v,y
        gc.collect();torch.cuda.empty_cache()
        phase('steady_state_timing')
        for fn in fns.values():
            for _ in range(10):fn()
        torch.cuda.synchronize()
        arms=list(fns)
        for i in range(12):
            order=arms[i%4:]+arms[:i%4]
            row={'round':i,'order':order}
            for arm in order:row[arm]=timed(fns[arm])
            r['timing'].append(row);save(path,r)
        comparisons={}
        for base in ('BF16_eager','BF16_compiled'):
            reductions=np.array([100*(1-t['W8A8_compiled']['wall_ms']/t[base]['wall_ms']) for t in r['timing']])
            rng=np.random.default_rng(20261104)
            boot=np.median(rng.choice(reductions,(10000,12)),axis=1)
            lo,hi=np.quantile(boot,[.025,.975]);med=float(np.median(reductions))
            comparisons[base]={'reductions_percent':reductions.tolist(),'median_percent':med,
                               'bootstrap_median_95ci_percent':[float(lo),float(hi)],'passes':bool(med>=5 and lo>0)}
        r['comparisons']=comparisons;r['promising']=all(v['passes'] for v in comparisons.values())
        phase('eager_component_diagnostic')
        qx,sx=quant(x.reshape(m,k));acc=intmm(qx,wt)
        stages={'input_quant':lambda:quant(x.reshape(m,k)),'int_gemm':lambda:intmm(qx,wt),
                'epilogue':lambda:dequant(acc,sx,sw,bias),'whole_eager':fns['W8A8_eager']}
        r['components']={}
        for name,fn in stages.items():
            for _ in range(10):fn()
            r['components'][name]=[timed(fn) for _ in range(3)]
        del qx,sx,acc
        phase('profile_warmed_arms')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            for arm,fn in fns.items():
                with torch.profiler.record_function(arm):z=fn();del z
        torch.cuda.synchronize();trace=folder/'trace.json';prof.export_chrome_trace(str(trace))
        r['trace']={'path':str(trace),'sha256':sha(trace),'bytes':trace.stat().st_size}
        generated=[]
        for file in sorted((folder/'inductor').rglob('*.py')):
            text=file.read_text(errors='replace')
            generated.append({'path':str(file),'sha256':sha(file),'bytes':file.stat().st_size,
                              'aten_int_mm':'extern_kernels._int_mm' in text,
                              'aten_addmm':'extern_kernels.addmm' in text,
                              'triton_definitions':text.count('@triton.jit')})
        r['generated_code']=generated
        r['status']='completed'
    except BaseException as exc:
        r['status']='failed';r['error']=repr(exc);r['traceback']=traceback.format_exc()
    finally:
        r['finished_utc']=time.time();save(path,r)
        print(index,r['status'],flush=True)


def parent():
    check();DATA.mkdir(parents=True,exist_ok=True)
    if (OUT/'results.json').exists():raise RuntimeError('Refusing overwrite')
    result={'status':'running','registered_start_utc':START,'deadline_utc':DEADLINE,'model_forwards':0,
            'code_sha256':sha(Path(__file__)),'registration_sha256':sha(OUT/'REGISTRATION_KO.txt'),
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'shapes':[]}
    save(OUT/'results.json',result)
    for index in range(3):
        if time.time()>=DEADLINE:
            result['shapes'].append({'index':index,'status':'not_started_deadline'});continue
        env=os.environ.copy();folder=DATA/f'shape{index}'
        env.update(TORCHINDUCTOR_CACHE_DIR=str(folder/'inductor'),TRITON_CACHE_DIR=str(folder/'triton'),
                   TORCH_LOGS='output_code',TOKENIZERS_PARALLELISM='false')
        log=OUT/f'shape{index}.log'
        with log.open('x') as f:
            p=subprocess.Popen([sys.executable,__file__,'--index',str(index)],env=env,cwd=ROOT,
                               stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            timeout=min(300,max(1,DEADLINE-time.time()))
            timed_out=False
            try:p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        path=OUT/f'shape{index}.json'
        row=json.loads(path.read_text()) if path.exists() else {'index':index,'status':'failed_without_record'}
        row['process_exit_code']=p.returncode;row['process_timeout']=timed_out
        if timed_out:row['status']='timeout'
        result['shapes'].append(row);save(OUT/'results.json',result)
        print('shape',index,row['status'],flush=True)
    result['status']='completed' if all(r['status']=='completed' for r in result['shapes']) else 'partial_or_invalid'
    result['finished_utc']=time.time();result['elapsed_from_registration_s']=time.time()-START
    save(OUT/'results.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int);args=parser.parse_args()
    if args.index is None:parent()
    else:
        with __import__('torch').inference_mode():child(args.index)
