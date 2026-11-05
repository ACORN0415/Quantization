"""Existing Sage1 components, explicit padding adapter, one synthetic baseline probe.

No Torch imports, installation, compilation, or CUDA work at module import time.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
PREP=HERE.parent/'native_attention_baseline_feasibility'
ARMS=('fa2','full_components','prepared_KV_components')
LENGTHS=(4680,18720,32760,56160)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def require(ok,why):
    if not ok:raise RuntimeError(why)


def save(name,value):
    p=HERE/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)


def imported(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def validate(m):
    require(m.get('status')=='approved_one_synthetic_native_baseline_probe','unapproved')
    require(m.get('root_execution_authorized') is True and m.get('actual_Opus_source_review_completed') is True,'review/authority absent')
    require(m['code_sha256']==sha(__file__) and m['spec_sha256']==sha(HERE/'ROOT_SPEC.json'),'code/spec changed')
    require(m['limits']==dict(wall_seconds=900,CPU=4,RSS_bytes=8*1024**3,GPU_allocator_bytes=4*1024**3,output_bytes=256*1024**2),'limits changed')
    require(m['interpreter']==sys.executable,'interpreter changed')
    for f,h in m['bound_files'].items():
        require(Path(f).is_absolute() and Path(f).resolve()==Path(f) and sha(f)==h,'bound file changed: '+f)


def pad_inputs(torch,q,k,v,with_query=True):
    require(q.ndim==k.ndim==v.ndim==4 and k.shape==v.shape and q.shape[:2]==k.shape[:2] and q.shape[-1]==k.shape[-1],'input shape')
    require(q.dtype==k.dtype==v.dtype==torch.bfloat16,'raw input dtype')
    qlen,klen=q.shape[2],k.shape[2];qp=((qlen+127)//128)*128;kp=((klen+63)//64)*64
    centered=k-k.mean(dim=2,keepdim=True)  # Valid keys only, same BF16 operations as core.
    qpad=torch.nn.functional.pad(q,(0,0,0,qp-qlen)) if with_query else None
    kpad=torch.nn.functional.pad(centered,(0,0,0,kp-klen))
    vpad=torch.nn.functional.pad(v.to(torch.float16),(0,0,0,kp-klen))
    valid=(torch.arange(kp,device=k.device)<klen).reshape(1,1,1,kp)
    mask=valid.expand(q.shape[0],q.shape[1],qp,kp)  # Zero-stride view, no Q*K allocation.
    return qpad,kpad,vpad,mask


class LaunchObserver:
    def __init__(self,real):self.real=real;self.calls=0;self.evidence={}
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            compiled=self.real[grid](*args,**kwargs);self.calls+=1
            key=id(compiled)
            if key not in self.evidence:
                ptx=getattr(compiled,'asm',{}).get('ptx','')
                lines=[x.strip() for x in ptx.splitlines() if 'mma.sync' in x]
                self.evidence[key]={'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
                    'int8_mma_present':any('.s8.' in x for x in lines),'mma_lines':sorted(set(lines))}
            return compiled
        return call


class Pipeline:
    def __init__(self,torch,quant,attention):
        self.torch=torch;self.quant=quant;self.attention=attention
        self.observer=LaunchObserver(attention._attn_fwd);attention._attn_fwd=self.observer
        self.calls={'Q_quant':0,'K_quant':0}
    def pack(self,x,block,factor,label):
        t=self.torch;b,h,n,d=x.shape
        require(x.is_cuda and n%block==0 and x.stride(-1)==1,'padded quant contract')
        out=t.empty(x.shape,dtype=t.int8,device=x.device);sc=t.empty((b,h,n//block),dtype=t.float32,device=x.device)
        self.quant.quant_per_block_int8_kernel[(n//block,h,b)](x,out,sc,n,*x.stride()[:3],*out.stride()[:3],sc.stride(0),sc.stride(1),sm_scale=factor,C=d,BLK=block)
        self.calls[label]+=1;return out,sc
    def prepare(self,q,k,v):
        _,kp,vp,mask=pad_inputs(self.torch,q,k,v,with_query=False)
        ki,ks=self.pack(kp,64,1.,'K_quant')
        return dict(k=ki,ks=ks,v=vp,mask=mask)
    def evaluate(self,q,k,v,prepared=None):
        t=self.torch
        if prepared is None:prepared=self.prepare(q,k,v)
        qp=t.nn.functional.pad(q,(0,0,0,((-q.shape[2])%128)))
        qi,qs=self.pack(qp,128,q.shape[-1]**-.5*1.44269504,'Q_quant')
        out,_=self.attention.forward(qi,prepared['k'],prepared['v'],qs,prepared['ks'],tensor_layout='HND',attn_mask=prepared['mask'],output_dtype=q.dtype,return_lse=False)
        return out[:,:,:q.shape[2]],dict(q=qi,qs=qs,**prepared)


def byte_equal(torch,a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def small_reference(torch,pack,qlen,klen):
    # Independent host arithmetic on the exact chosen quantized representation.
    qi=pack['q'].cpu().double()[:,:,:qlen];ki=pack['k'].cpu().double()[:,:,:klen]
    qs=pack['qs'].cpu().double().repeat_interleave(128,dim=2)[:,:,:qlen]
    ks=pack['ks'].cpu().double().repeat_interleave(64,dim=2)[:,:,:klen]
    scores=(qi*qs.unsqueeze(-1))@(ki*ks.unsqueeze(-1)).transpose(-1,-2)
    probabilities=torch.softmax(scores*0.6931471805599453,dim=-1)
    return probabilities@pack['v'].cpu().double()[:,:,:klen]


def run(manifest_path):
    start=time.time();r={'status':'failed','started_epoch':start,'models_datasets_loads':0,'installs':0,
        'scope':'Synthetic existing Sage1 component baseline with pad/mask adapter, not unmodified core API, SOTA, video quality or novel method.'};phase='manifest';torch=None
    require(not (HERE/'RUN.json').exists() and not (HERE/'RESULT.json').exists(),'one run only')
    try:
        m=json.loads(Path(manifest_path).read_text());validate(m)
        r.update(code_sha256=sha(__file__),manifest_sha256=sha(manifest_path))
        with (HERE/'RUN.json').open('x') as f:json.dump({'started_epoch':start,'manifest_sha256':r['manifest_sha256']},f)
        def stop(sig,frame):raise KeyboardInterrupt('root signal '+str(sig))
        signal.signal(signal.SIGTERM,stop)
        def denied(*args,**kwargs):raise RuntimeError('network forbidden')
        socket.create_connection=denied
        connect=socket.socket.connect
        def local_only(sock,address):
            if sock.family in (socket.AF_INET,socket.AF_INET6):denied()
            return connect(sock,address)
        socket.socket.connect=local_only;socket.socket.connect_ex=local_only
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
        adapter=imported(PREP/'adapter.py','_baseline_source_loader');modules=adapter.load_sources()
        import torch as torch_module
        torch=torch_module;torch.set_num_threads(4);torch.set_num_interop_threads(1)
        from flash_attn import flash_attn_func
        require(not torch.cuda.is_initialized(),'CUDA initialized during imports')
        require(torch.__version__=='2.6.0+cu124','Torch version changed')
        fa_module=sys.modules.get('flash_attn_2_cuda')
        require(fa_module is not None and str(Path(fa_module.__file__).resolve()) in m['bound_files'],'FA2 native CUDA module not bound')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
        require(torch.cuda.get_device_capability(0)==(8,9),'SM89 required')
        torch.cuda.reset_peak_memory_stats();g=torch.Generator(device='cuda').manual_seed(321)
        pipe=Pipeline(torch,modules['quant_per_block'],modules['attn_qk_int8_per_block'])
        def guard():
            require(time.time()-start<=900,'wall cap')
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3,'RSS cap')
            require(torch.cuda.max_memory_allocated()<=4*1024**3,'GPU cap')
            require(sum(f.stat().st_size for f in HERE.rglob('*') if f.is_file())<=256*1024**2,'output cap')
        def fixture(qn,kn,heads,dim):
            return tuple(torch.randn((1,heads,n,dim),generator=g,device='cuda',dtype=torch.bfloat16) for n in (qn,kn,kn))
        phase='small_quantized_reference';small=[];t0=time.perf_counter()
        with torch.inference_mode():
            for name in ('random','zero_Q'):
                guard();q,k,v=fixture(37,73,2,64)
                if name=='zero_Q':q.zero_()
                cache=pipe.prepare(q,k,v);full,pf=pipe.evaluate(q,k,v);cached,pc=pipe.evaluate(q,k,v,cache);torch.cuda.synchronize()
                for field in ('q','qs','k','ks','v','mask'):require(byte_equal(torch,pf[field],pc[field]),'small prepared field differs: '+field)
                require(byte_equal(torch,full,cached),'small output byte mismatch')
                require(torch.isfinite(full).all(),'small nonfinite')
                ref=small_reference(torch,pf,37,73);err=full.cpu().double()-ref
                rel=(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(ref).clamp_min(1e-12)).item();ma=err.abs().max().item()
                item={'name':name,'relative_Frobenius_error':rel,'max_abs_error':ma,'full_prepared_bytes_equal':True}
                small.append(item);r['small']=small;save('PROGRESS.json',r)
                require(rel<=.01 and ma<=.05,'small numerical reference gate')
                torch.save({'raw_q':q.cpu(),'raw_k':k.cpu(),'raw_v':v.cpu(),'packing':{n:x.cpu() for n,x in pf.items()},'output':full.cpu(),'reference':ref},HERE/('SMALL_'+name+'.pt'))
            r['small_validation_seconds_including_first_JIT']=time.perf_counter()-t0
            phase='large_shapes';r['large']=[];fa2calls=0
            for case,kl in enumerate(LENGTHS):
                guard();case_start=time.perf_counter();q0,k,v=fixture(4680,kl,12,128)
                q1=torch.randn(q0.shape,generator=g,device=q0.device,dtype=q0.dtype)
                qs=(q0,q1);cache=pipe.prepare(q0,k,v);qual=[]
                def fa2(q):
                    nonlocal fa2calls
                    fa2calls+=1
                    return flash_attn_func(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),causal=False,softmax_scale=128**-.5).transpose(1,2)
                def evaluate(arm,q):
                    if arm=='fa2':return fa2(q)
                    return pipe.evaluate(q,k,v,None if arm=='full_components' else cache)[0]
                for i,q in enumerate(qs):
                    f,_=pipe.evaluate(q,k,v);c,_=pipe.evaluate(q,k,v,cache);reference=fa2(q);torch.cuda.synchronize()
                    require(byte_equal(torch,f,c) and torch.isfinite(f).all() and torch.isfinite(reference).all(),'large equality/finite gate')
                    error=f.float()-reference.float()
                    qual.append({'Q_fixture':i,'full_prepared_bytes_equal':True,'FA2_relative_Frobenius_difference':(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)).item(),'FA2_max_abs_difference':error.abs().max().item()})
                    del _
                require(any(v['int8_mma_present'] for v in pipe.observer.evidence.values()),'INT8 MMA PTX not observed')
                setup=time.perf_counter()-case_start;times=[];before=(dict(pipe.calls),pipe.observer.calls,fa2calls)
                for cycle in range(10):
                    order=ARMS[cycle%3:]+ARMS[:cycle%3]
                    for arm in order:
                        guard();q=qs[cycle%2];torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                        a.record();t0=time.perf_counter();out=evaluate(arm,q);z.record();torch.cuda.synchronize();wall=time.perf_counter()-t0
                        if cycle>=2:times.append({'cycle':cycle-2,'Q_fixture':cycle%2,'arm':arm,'wall_seconds':wall,'CUDA_event_ms':a.elapsed_time(z)})
                after=(dict(pipe.calls),pipe.observer.calls,fa2calls)
                require(after[0]['Q_quant']-before[0]['Q_quant']==20 and after[0]['K_quant']-before[0]['K_quant']==10 and after[1]-before[1]==20 and after[2]-before[2]==10,'timing launch counters')
                item={'K_valid':kl,'Q_valid':4680,'quality_descriptive_only':qual,'case_setup_seconds_including_qual_and_first_use_JIT':setup,'timings':times,
                    'timed_plus_warmup_calls':dict(Q_quant=20,K_quant=10,attention=20,FA2=10),'prepared_KV_cost_is_in_setup_not_per_call':True}
                r['large'].append(item);save('PROGRESS.json',r)
                del q0,q1,q,k,v,qs,cache,f,c,reference,error,out
        r['actual_calls']=dict(pipe.calls,attention=pipe.observer.calls,FA2=fa2calls)
        require(r['actual_calls']==dict(Q_quant=100,K_quant=56,attention=100,FA2=48),'total launch counters')
        r['compiled_attention_evidence']=list(pipe.observer.evidence.values())
        r['GPU']={'name':torch.cuda.get_device_name(0),'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cuDNN':torch.backends.cudnn.allow_tf32}
        r['cost_scope']='Component pipeline includes Q quantization/padding, attention, and full-path K mean/quant/V cast+pad. Prepared-cache one-time cost reported in setup; mask is a broadcast view. Checks/hook overhead present. No original core API, C++ Sage2 or complete video timing.'
        require(sha(manifest_path)==r['manifest_sha256'],'manifest changed')
        require(all(sha(f)==h for f,h in m['bound_files'].items()),'bound file changed')
        guard();r['status']='completed_synthetic_older_native_baseline_not_method_effect'
    except BaseException as exc:r.update(failed_phase=phase,error=repr(exc),traceback=traceback.format_exc())
    finally:
        r.update(finished_epoch=time.time(),wall_seconds=time.time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save('RESULT.json',r)
    print(json.dumps({'status':r['status'],'phase':r.get('failed_phase'),'seconds':r['wall_seconds']}));return 0 if r['status'].startswith('completed_') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();raise SystemExit(run(a.manifest))
