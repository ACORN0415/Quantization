"""Preregisurable single-kernel OBQ code port and native-GPU encoder variant.
Imports are CPU-only. No GPU or compiler use until explicit run/worker action.
"""
import argparse,gc,hashlib,importlib.util,json,math,os,resource,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['MAX_JOBS']='2'
os.environ['TORCH_CUDA_ARCH_LIST']='8.9'
sys.dont_write_bytecode=True
import torch
torch.set_num_threads(4);torch.set_grad_enabled(False)
P=Path(__file__).resolve().parent;C=P.parent/'causal_obq';S1=P.parent/'obq_gpu_cost_stage1'
ROOT=P.parents[3]
spec=importlib.util.spec_from_file_location('frozen_stage1',S1/'benchmark.py');v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
c=v1.common
CASES=((18,1),(24,6));CAP=4*1024**3
PAIRS=(('RTN_eager','OBQ_graph'),('RTN_graph','OBQ_graph'),('OBQ_fused','OBQ_graph'))
IMPLS=('RTN_eager','RTN_graph','OBQ_graph','OBQ_fused')
EXT=None

def guard(begin):
    if time.monotonic()-begin>2700:raise TimeoutError('45-minute worker limit')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:raise MemoryError('8GiB RSS limit')
    if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>CAP:raise MemoryError('4GiB observed allocated peak limit')
    if sum(f.stat().st_size for f in P.rglob('*') if f.is_file())>2*1024**3:raise OSError('2GiB new disk limit')

def record(p):return {'path':str(p),'sha256':c.sha(p),'bytes':Path(p).stat().st_size}
def eq(a,b):return v1.same(a,b)
def pcheck(a,b):v1.packet_check(a,b)
def to_cpu(p):return tuple(x.cpu().clone() for x in p)
def digest_inputs(args):return [c.tsha(x.cpu()) for x in args]

def build_extension():
    from torch.utils.cpp_extension import load
    build=P/'build';build.mkdir(exist_ok=False)
    name='obq_stage2_'+c.sha(P/'obq_pack.cu')[:12]
    attempts=[]
    for attempt in range(2):
        t=time.monotonic()
        try:
            ext=load(name=name,sources=[str(P/'obq_pack.cu')],build_directory=str(build),
                extra_cuda_cflags=['-O3','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false','-lineinfo','--ptxas-options=-v'],
                extra_cflags=['-O3'],with_cuda=True,verbose=True)
            attempts.append({'attempt':attempt+1,'seconds':time.monotonic()-t,'status':'success'})
            c.save(P/'BUILD.json',{'attempts':attempts,'source_sha256':c.sha(P/'obq_pack.cu'),'module':ext.__file__})
            return ext
        except BaseException as e:
            attempts.append({'attempt':attempt+1,'seconds':time.monotonic()-t,'status':'failed','error':repr(e)})
            c.save(P/'BUILD.json',{'attempts':attempts,'source_sha256':c.sha(P/'obq_pack.cu')})
            if attempt==1:raise

def preflight(z,s,u):
    assert z.dtype==torch.bfloat16 and s.dtype==torch.float16 and u.dtype==torch.float32
    assert z.is_contiguous() and s.is_contiguous() and u.is_contiguous()
    assert bool(torch.isfinite(z).all()) and bool(torch.isfinite(s).all()) and bool((s>0).all())
    assert bool(torch.isfinite(u).all()) and bool((u.diagonal(dim1=1,dim2=2)>0).all())
    assert torch.count_nonzero(torch.tril(u,-1))==0

def raw(stage,impl,args,hm):
    if stage=='T_code':z,s,u=args
    else:
        k,u=args;z,s=v1.gpu_frames(k,hm)
    if impl.startswith('RTN'):return v1.code_packet(z,s,u,'R_fp32')
    if impl=='OBQ_fused':return EXT.obq_pack(z,s,u),s,torch.full((1,),255,dtype=torch.uint8,device=z.device)
    return v1.code_packet(z,s,u,'O_offline')

def cpu_reference(stage,args,hm,alg):
    if stage=='T_code':z,s,u=args
    else:
        k,u=args;z,s=v1.gpu_frames(k,hm)
    z,s,u=[x.cpu() for x in (z,s,u)];preflight(z,s,u)
    return to_cpu(v1.code_packet(z,s,u,alg))

def build_functions(stage,example,hm):
    """One reusable Graph per algorithm and stage; U copied each OBQ call."""
    fs={};detail={};graphs=[]
    for impl in IMPLS:
        t=time.monotonic()
        if not impl.endswith('graph'):
            def eager(args,impl=impl):
                p,s,t=raw(stage,impl,args,hm)
                return p,s.clone() if stage=='T_code' else s,t
            fs[impl]=eager
            detail[impl]={'kind':'owned_eager','setup_seconds':time.monotonic()-t}
            continue
        static=[x.clone() for x in example]
        # RTN neither reads nor copies U. Its static U is an unused shape fixture.
        indices=range(len(static)-1) if impl=='RTN_graph' else range(len(static))
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):warm=raw(stage,impl,static,hm)
        torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();del warm
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=raw(stage,impl,static,hm)
        def replay(args,static=static,indices=tuple(indices),graph=graph,out=out):
            for i in indices:static[i].copy_(args[i])
            graph.replay()
            # Outside capture: ownership survives reuse of static output.
            return tuple(x.clone() for x in out)
        fs[impl]=replay;graphs.append(graph)
        detail[impl]={'kind':'owned_graph','setup_seconds':time.monotonic()-t,
            'input_copy_bytes':sum(example[i].numel()*example[i].element_size() for i in indices),
            'owned_output_copy_bytes':4515841,'factor_copied_each_call':impl=='OBQ_graph'}
    torch.cuda.synchronize()
    return fs,detail

def check_lifetime(stage,fs,args,hm):
    # Distinct data AND distinct upper-triangular factor. No quality selection.
    second=[x.clone() for x in args]
    second[0]=(-second[0].float()*.75).bfloat16()
    if stage=='T_code':second[1]=(second[1].float()*1.25).half()
    second[-1]=(second[-1]*torch.linspace(.7,1.3,128,device=second[-1].device)[None,None,:]).contiguous()
    before=[digest_inputs(args),digest_inputs(second)];records={}
    for impl,f in fs.items():
        alg='R_fp32' if impl.startswith('RTN') else 'O_offline'
        ref1=cpu_reference(stage,args,hm,alg);ref2=cpu_reference(stage,second,hm,alg)
        assert not eq(ref1[0],ref2[0]), 'lifetime test requires genuinely distinct packet codes'
        one=f(args);snapshot=to_cpu(one);pcheck(one,ref1)
        assert all(t.data_ptr()!=x.data_ptr() for t in one for x in args)
        two=f(second);pcheck(two,ref2);pcheck(one,snapshot)
        # Returned scale must also be independent of resident input buffers.
        assert all(one[i].data_ptr()!=two[i].data_ptr() for i in range(3))
        records[impl]={'two_inputs_reference_equal':True,'first_packet_preserved':True,'distinct_output_storage':True}
        del one,two,snapshot
    assert before==[digest_inputs(args),digest_inputs(second)]
    return records

def qstats(values):
    q=statistics.quantiles(values,n=4,method='inclusive')
    return {'median':statistics.median(values),'q1':q[0],'q3':q[2],'iqr':q[2]-q[0]}

def timed_pairs(stage,fs,args,hm,begin):
    rows=[];summaries=[]
    for a,b in PAIRS:
        for _ in range(5):fs[a](args);fs[b](args)
        torch.cuda.synchronize();samples=[]
        for rep in range(20):
            sample={'round':rep,'order':[a,b] if rep%2==0 else [b,a],'times':{}}
            for name in sample['order']:
                torch.cuda.synchronize();s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
                tick=time.perf_counter();s.record();out=fs[name](args);e.record();e.synchronize()
                sample['times'][name]={'gpu_seconds':s.elapsed_time(e)/1000,'host_seconds':time.perf_counter()-tick}
            samples.append(sample);guard(begin)
        summary={name:{kind:qstats([x['times'][name][kind] for x in samples]) for kind in ('host_seconds','gpu_seconds')} for name in (a,b)}
        paired={kind:qstats([x['times'][b][kind]-x['times'][a][kind] for x in samples]) for kind in ('host_seconds','gpu_seconds')}
        rows.append({'pair':[a,b],'samples':samples,'summary':summary,'paired_second_minus_first':paired})
    for impl,f in fs.items():pcheck(f(args),cpu_reference(stage,args,hm,'R_fp32' if impl.startswith('RTN') else 'O_offline'))
    return rows

def verified(path,manifest):return v1.verified(path,manifest)

def restore_gpu(packet,hm):
    """Fixed future-reader path: GPU unpack, BF16 dequant, GPU inverse rotation."""
    packed=packet['packed'];flat=torch.empty(packed.numel()*2,dtype=torch.uint8,device=packed.device)
    flat[::2]=packed&15;flat[1::2]=packed>>4
    signed=flat.to(torch.int16);signed=torch.where(signed>=8,signed-16,signed).to(torch.int8).reshape(packet['shape'])
    restored=signed.bfloat16()*packet['scale'].bfloat16()
    if packet['kind']=='V':return restored.reshape(1,-1,12,128)[:,:4680].contiguous()
    restored=restored.reshape(3,1568,12,128)[:,:1560].reshape(1,4680,12,128)
    return (restored.float()@hm.t()).bfloat16()

def quality(begin,hm,factors):
    sys.path.insert(0,str(ROOT/'src'))
    from wan.modules.attention import attention
    cm=json.loads((C/'capture/manifest.json').read_text());pm=json.loads((C/'packets/manifest.json').read_text())
    target=P/'QUALITY.json';r={'status':'running','endpoints':[],'packets':[]}
    c.save(target,r)
    for layer in (18,24):
        kp=[];vp=[];vq=[];kq={a:[] for a in ('R_bf16','R_fp32','O_gpu')};u=factors[layer]
        for chunk in range(1,7):
            guard(begin);w=verified(C/f'capture/l{layer}_write{chunk}.pt',cm);kp.append(w['k']);vp.append(w['v'])
            v_packet=verified(C/f'packets/l{layer}_c{chunk}_V.pt',pm)
            vg={key:value.cuda() if isinstance(value,torch.Tensor) else value for key,value in v_packet.items()}
            v_recovered=restore_gpu(vg,hm);assert eq(v_recovered,c.restore(v_packet)), 'V GPU elementwise restore differs'
            vq.append(v_recovered)
            k=w['k'].cuda();z,s=v1.gpu_frames(k,hm);preflight(z,s,u)
            cpu_z,cpu_s=z.cpu(),s.cpu();state_before=digest_inputs((k,z,s,u))
            for alg in ('R_bf16','R_fp32','O_gpu'):
                ref_alg='O_offline' if alg=='O_gpu' else alg
                ref=to_cpu(v1.code_packet(cpu_z,cpu_s,u.cpu(),ref_alg))
                out=(EXT.obq_pack(z,s,u),s.clone(),torch.full((1,),255,dtype=torch.uint8,device=z.device)) if alg=='O_gpu' else v1.code_packet(z,s,u,alg)
                pcheck(out,ref);pc=to_cpu(out)
                packet={'packed':pc[0],'scale':pc[1],'sentinel':pc[2],'shape':tuple(z.shape),'kind':'K'}
                assert c.nbytes(packet)==4515841
                path=P/f'quality_l{layer}_c{chunk}_{alg}.pt';torch.save(packet,path)
                gp={'packed':out[0],'scale':out[1],'sentinel':out[2],'shape':tuple(z.shape),'kind':'K'}
                restored_gpu=restore_gpu(gp,hm);kq[alg].append(restored_gpu)
                r['packets'].append({**record(path),'GPU_reader_vs_CPU_unequal_elements':int((restored_gpu.cpu()!=c.restore(packet)).sum()),'V_GPU_restore_exact':True})
            assert state_before==digest_inputs((k,z,s,u));del k,z,s,w
        wo=verified(C/f'capture/l{layer}_wo.pt',cm)
        for call in (0,3):
            guard(begin);d=verified(C/f'capture/l{layer}_read{call}.pt',cm);q=d['q'].cuda()
            k=torch.cat(kp+[d['k_current']],1).cuda();v=torch.cat(vp+[d['v_current']],1).cuda()
            ref=attention(q,k,v)[0,c.QIDS].cpu();assert eq(ref,d['native_av'][0]),'unchanged native BF16 replay mismatch'
            del k,v;v=torch.cat(vq+[d['v_current'].cuda()],1);saved={'reference':ref};row={'layer':layer,'call':call,'native_replay_byte_equal':True,'K_payload_bytes_per_chunk':4515841,'arms':{}}
            yy=ref.float().reshape(128,-1)@wo['wo'].float().t()+wo['bias'].float();energy=float(yy.double().square().sum())
            for alg in ('R_bf16','R_fp32','O_gpu'):
                k=torch.cat(kq[alg]+[d['k_current'].cuda()],1)
                for _ in range(2):attention(q,k,v)
                outputs=[attention(q,k,v)[0,c.QIDS].cpu() for _ in range(3)]
                assert all(eq(outputs[0],o) for o in outputs[1:]);out=outputs[0];saved[alg]=out
                delta=(out.float()-ref.float()).reshape(128,-1)@wo['wo'].float().t()
                sse=float(delta.double().square().sum());row['arms'][alg]={'sse':sse,'nse':sse/energy,'reference_energy':energy,'repeat_bytes_equal':True}
                del k,outputs
            outpath=P/f'quality_l{layer}_read{call}.pt';torch.save(saved,outpath);row['output_artifact']=record(outpath)
            row['gains']={a:1-row['arms']['O_gpu']['sse']/row['arms'][a]['sse'] for a in ('R_bf16','R_fp32')}
            r['endpoints'].append(row);c.save(target,r);del q,v,d
        del kp,vp,vq,kq,wo;gc.collect();torch.cuda.empty_cache()
    r['quality_gate']=all(e['gains'][a]>=.2 for e in r['endpoints'] for a in ('R_bf16','R_fp32'))
    r['status']='completed';c.save(target,r);return record(target)

def worker():
    global EXT
    begin=time.monotonic();target=P/'RESULT.json';r={'status':'running','started_utc':time.time(),'models_loaded':0,'generations':0,'timings':[]}
    assert not target.exists();c.save(target,r)
    try:
        lock=json.loads((P/'LOCK.json').read_text());assert all(c.sha(p)==h for p,h in lock['sha256'].items());r['lock_sha256']=c.sha(P/'LOCK.json')
        assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,9)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        torch.cuda.reset_peak_memory_stats();hm=c.hadamard().cuda();EXT=build_extension();guard(begin)
        r['device']=torch.cuda.get_device_name();r['torch_version']=torch.__version__;r['cuda_version']=torch.version.cuda
        data=[];factors={};tx=time.monotonic()
        for layer,chunk in CASES:
            p=S1/f'prepared_l{layer}_c{chunk}.pt';d=torch.load(p,map_location='cpu',weights_only=True)
            args={k:d[k].cuda() for k in ('k','z','s','u')};preflight(args['z'],args['s'],args['u']);factors[layer]=args['u'];data.append((d,args))
        torch.cuda.synchronize();r['initial_H2D_seconds']=time.monotonic()-tx
        # Validate all code/lifetime paths first; no timing expansion before native quality.
        runtimes={};r['code_lifetime_validation']=[]
        for stage in ('T_code','T_encoder'):
            first=data[0][1];example=(first['z'],first['s'],first['u']) if stage=='T_code' else (first['k'],first['u'])
            fs,setup=build_functions(stage,example,hm);guard(begin);runtimes[stage]=(fs,setup)
            for d,gpu in data:
                args=(gpu['z'],gpu['s'],gpu['u']) if stage=='T_code' else (gpu['k'],gpu['u'])
                if stage=='T_code':
                    for impl,f in fs.items():pcheck(f(args),d['expected']['R_fp32' if impl.startswith('RTN') else 'O_offline'])
                life=check_lifetime(stage,fs,args,hm)
                r['code_lifetime_validation'].append({'layer':d['layer'],'chunk':d['chunk'],'stage':stage,'lifetime':life,'setup':setup,'shared_graph_setup_across_cases':True})
                c.save(target,r)
        r['quality']=quality(begin,hm,factors);guard(begin)
        if not json.loads((P/'QUALITY.json').read_text())['quality_gate']:
            assert all(c.sha(p)==h for p,h in lock['sha256'].items())
            r['status']='completed_quality_gate_failed';r['timing_skipped_reason']='native GPU quality below preregistered 20 percent on at least one comparison';return
        for stage in ('T_code','T_encoder'):
            fs,setup=runtimes[stage]
            for d,gpu in data:
                guard(begin);args=(gpu['z'],gpu['s'],gpu['u']) if stage=='T_code' else (gpu['k'],gpu['u'])
                old=digest_inputs(args);rows=timed_pairs(stage,fs,args,hm,begin);assert old==digest_inputs(args)
                z2,s2=v1.gpu_frames(gpu['k'],hm)
                r['timings'].append({'layer':d['layer'],'chunk':d['chunk'],'stage':stage,'setup':setup,'pairs':rows,
                    'input_preserved':True,'shared_graph_setup_across_cases':True,'CPU_original_rotated_equal':eq(z2,d['z']),'CPU_original_scale_equal':eq(s2,d['s']),
                    'CPU_original_rotated_unequal_elements':int((z2.cpu()!=d['z']).sum()),'CPU_original_scale_unequal_elements':int((s2.cpu()!=d['s']).sum()),
                    'fixed_CPU_packet_gate':stage=='T_code','GPU_preprocessed_CPU_code_reference_gate':True,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'BF16_reference_chunk_seconds':d['BF16_reference_consumer_chunk_seconds']})
                c.save(target,r);print('TIMING_DONE',d['layer'],d['chunk'],stage,flush=True)
        guard(begin)
        assert all(c.sha(p)==h for p,h in lock['sha256'].items());r['status']='completed'
    except BaseException as e:r.update(status='failed_or_incomplete',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        r.update(elapsed_seconds=time.monotonic()-begin,finished_utc=time.time(),max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            disk_bytes=sum(f.stat().st_size for f in P.rglob('*') if f.is_file()))
        c.save(target,r)

def source_files():
    files=[P/'experiment.py',P/'obq_pack.cu',P/'REGISTRATION_DRAFT_KO.txt',P/'SELFTEST.json',P/'selftest.py',S1/'benchmark.py',S1/'LOCK.json',S1/'RESULT.json',C/'common.py',C/'LOCK.json',C/'replay.py',C/'capture/manifest.json',C/'packets/manifest.json',C/'NATIVE_RESULT.json']
    files.extend(S1/f'prepared_l{l}_c{k}.pt' for l,k in CASES)
    files.extend(ROOT/x for x in ('src/wan/modules/attention.py','src/wan/modules/causal_model.py','src/kvq/store.py','src/kvq/lockstep.py'))
    cm=json.loads((C/'capture/manifest.json').read_text());pm=json.loads((C/'packets/manifest.json').read_text())
    for m in (cm,pm):files.extend(Path(x['path']) for x in m['artifacts'])
    return files

def lock():
    assert not (P/'LOCK.json').exists();st=json.loads((P/'SELFTEST.json').read_text())
    assert st['passed'] and st['python_sha256']==c.sha(__file__) and st['cuda_sha256']==c.sha(P/'obq_pack.cu')
    c.save(P/'LOCK.json',{'registered_utc':time.time(),'seconds_cap':2700,'sha256':{str(p):c.sha(p) for p in source_files()}})

def run():
    assert not (P/'SUPERVISOR.json').exists();start=time.monotonic();r={'started_utc':time.time()};process=None
    try:
        with (P/'RUN_LOG.txt').open('x') as out:
            process=subprocess.Popen([sys.executable,str(P/'experiment.py'),'worker'],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=max(1,2695-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                r['timeout']=True;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=max(.01,2700-(time.monotonic()-start)))
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise TimeoutError('45-minute budget including reserved cleanup; only own worker process group terminated')
        r['return_code']=code;r['status']='finished' if not code else 'worker_failed'
    except BaseException as e:
        if process is not None and process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
        r.update(status='supervisor_failed',error=repr(e));raise
    finally:r.update(elapsed_seconds=time.monotonic()-start,finished_utc=time.time());c.save(P/'SUPERVISOR.json',r)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('lock','run','worker'));a=ap.parse_args()
    {'lock':lock,'run':run,'worker':worker}[a.action]()
