"""Validity investigation only. Same source and binary; no quality or timing."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
from pathlib import Path
import hashlib,importlib.util,json,signal,subprocess,sys,time,traceback
P=Path(__file__).resolve().parent;SRC=P.parent/'obq_gpu_encoder_stage2_envfix';DEADLINE=1793875658.1258163
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    p=P/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def worker():
    import torch
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    spec=importlib.util.spec_from_file_location('fixed_stage2',SRC/'experiment.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    binary=Path(json.loads((SRC/'BUILD.json').read_text())['module'])
    assert sha(binary)=='ae6e1a50caeff3b13327ab15b27a3e20695530938d4068ab5e578291697eff75'
    assert sha(SRC/'experiment.py')=='8d5aea59e19d5bdc860c076963335af4e852898959c82b2a2aaac100565bb614'
    bs=importlib.util.spec_from_file_location('obq_stage2_50b86fd47474',binary);m.EXT=importlib.util.module_from_spec(bs);bs.loader.exec_module(m.EXT)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    start=time.monotonic();r={'status':'running','started_utc':time.time(),'compile_calls':0,'quality_calls':0,'timing_samples':0,'impl':{},'source_sha256':sha(SRC/'experiment.py'),'binary_sha256':sha(binary)};save('RESULT.json',r);artifacts={}
    def guard():
        assert time.time()<DEADLINE and time.monotonic()-start<115
    def meta(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'stride':list(t.stride()),'contiguous':t.is_contiguous(),'storage_offset':t.storage_offset()}
    def diff(a,b):
        a=a.detach().cpu();b=b.detach().cpu();assert a.shape==b.shape and a.dtype==b.dtype
        return {'all_element_bytes_equal':m.eq(a,b),'unequal_elements':int((a!=b).sum()),'max_abs_diff':float((a.float()-b.float()).abs().max())}
    def pdiff(a,b):return {key:diff(x,y) for key,x,y in zip(('packed','scale','sentinel'),a,b)}
    try:
        d=torch.load(m.S1/'prepared_l18_c1.pt',map_location='cpu',weights_only=True);hm=m.c.hadamard().cuda();args=(d['k'].cuda(),d['u'].cuda())
        second=[x.clone() for x in args];second[0]=(-second[0].float()*.75).bfloat16();second[-1]=(second[-1]*torch.linspace(.7,1.3,128,device='cuda')[None,None,:]).contiguous()
        r['input_meta']={'CPU_original_k':meta(d['k']),'GPU_first_k':meta(args[0]),'GPU_second_k':meta(second[0])}
        before=[m.digest_inputs(args),m.digest_inputs(second)];fs,setup=m.build_functions('T_encoder',args,hm);r['setup']=setup
        for impl,f in fs.items():
            guard();alg='R_fp32' if impl.startswith('RTN') else 'O_offline'
            ref1=m.cpu_reference('T_encoder',args,hm,alg);ref2=m.cpu_reference('T_encoder',second,hm,alg)
            one=f(args);snap=m.to_cpu(one);first=pdiff(one,ref1)
            two=f(second);second_diff=pdiff(two,ref2);preservation=pdiff(one,snap)
            r['impl'][impl]={'first_reference':first,'second_reference':second_diff,'first_preserved':preservation,
                'distinct_output_storage':all(a.data_ptr()!=b.data_ptr() for a,b in zip(one,two))}
            artifacts[impl]={'first':snap,'ref1':ref1,'second':m.to_cpu(two),'ref2':ref2,'first_after_second':m.to_cpu(one)}
            save('RESULT.json',r)
        r['inputs_preserved']=before==[m.digest_inputs(args),m.digest_inputs(second)]
        # Isolate the unchanged GPU preprocessing helper; no new math/precision.
        static=args[0].clone();r['input_meta']['preprocess_static_k']=meta(static)
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):warm=m.v1.gpu_frames(static,hm)
        torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();del warm
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=m.v1.gpu_frames(static,hm)
        r['preprocess']={}
        for label,a in (('first',args),('second',second)):
            guard();eager1=m.v1.gpu_frames(a[0],hm);eager2=m.v1.gpu_frames(a[0],hm)
            static.copy_(a[0]);graph.replay();z,s=(x.clone() for x in captured)
            r['preprocess'][label]={'eager_repeat_z':diff(eager1[0],eager2[0]),'eager_repeat_scale':diff(eager1[1],eager2[1]),
                'graph_vs_eager_z':diff(z,eager1[0]),'graph_vs_eager_scale':diff(s,eager1[1])}
            artifacts['preprocess_'+label]={'eager_z':eager1[0].cpu(),'eager_scale':eager1[1].cpu(),'graph_z':z.cpu(),'graph_scale':s.cpu()}
        torch.save(artifacts,P/'TENSORS.pt');r['tensors']={'sha256':sha(P/'TENSORS.pt'),'bytes':(P/'TENSORS.pt').stat().st_size}
        r['status']='completed';guard()
    except BaseException as e:r.update(status='failed_or_incomplete',error=repr(e),traceback=traceback.format_exc());raise
    finally:r.update(elapsed_seconds=time.monotonic()-start,finished_utc=time.time(),peak_cuda_bytes=torch.cuda.max_memory_allocated());save('RESULT.json',r)
def main():
    start=time.monotonic();r={'started_utc':time.time(),'absolute_deadline_utc':DEADLINE};proc=None
    try:
        assert not (P/'SUPERVISOR.json').exists()
        remaining=min(120,DEADLINE-time.time());assert remaining>5
        with (P/'RUN_LOG.txt').open('x') as f:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=proc.wait(timeout=remaining-5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise TimeoutError('2-minute/absolute validity-probe cap')
        r.update(status='completed' if rc==0 else 'worker_failed',return_code=rc)
    except BaseException as e:
        if proc is not None and proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        r.update(status='failed_or_incomplete',error=repr(e));raise
    finally:r.update(elapsed_seconds=time.monotonic()-start,finished_utc=time.time());save('SUPERVISOR.json',r)
if __name__=='__main__':
    worker() if len(sys.argv)>1 and sys.argv[1]=='worker' else main()
