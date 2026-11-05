"""Fixed offline-factor OBQ GPU cost diagnostic. No custom kernels or model load."""
import argparse
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
import traceback

if len(sys.argv)>1 and sys.argv[1]=='selftest':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
sys.dont_write_bytecode=True
import torch
torch.set_num_threads(4);torch.set_grad_enabled(False)
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'causal_obq'
spec=importlib.util.spec_from_file_location('frozen_causal_common',SOURCE/'common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
CASES=((18,1),(24,6));ALGS=('R_fp32','O_offline')
STAGES=('T_code','T_encoder');CAP_BYTES=4*1024**3


def packet(c,s):
    return common.pack(c),s,torch.full((1,),255,dtype=torch.uint8,device=c.device)


def batch_codes(z,s,u):
    w=z.float().reshape(-1,12,128)
    scales=s.float().expand_as(z).reshape(-1,12,128)
    out=torch.empty_like(w,dtype=torch.int8)
    for j in range(128):
        c=(w[:,:,j]/scales[:,:,j]).round().clamp(-8,7)
        out[:,:,j]=c.to(torch.int8)
        if j<127:
            e=(w[:,:,j]-c*scales[:,:,j])/u[:,j,j][None,:]
            w[:,:,j+1:]-=e[:,:,None]*u[:,j,j+1:][None,:,:]
    return out.reshape(z.shape)


def code_packet(z,s,u,alg):
    if alg=='O_offline':c=batch_codes(z,s,u)
    elif alg=='R_fp32':c=(z.float()/s.float()).round().clamp(-8,7).to(torch.int8)
    elif alg=='R_bf16':c=(z/s.bfloat16()).round().clamp(-8,7).to(torch.int8)
    else:raise ValueError(alg)
    return packet(c,s)


def gpu_frames(k,hm):
    z=(k.float()@hm).to(k.dtype)[0].reshape(3,1560,12,128)
    z=torch.cat((z,z.new_zeros(3,8,12,128)),1).reshape(3,98,16,12,128)
    s=(z.abs().amax(2,keepdim=True)/7).clamp_min(1e-8)
    return z,s.to(torch.float16)


def full_packet(k,hm,u,alg):
    z,s=gpu_frames(k,hm)
    return code_packet(z,s,u,alg)


def cpu_tuple(p):return p['packed'],p['scale'],p['sentinel']


def same(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.cpu().contiguous().view(torch.uint8),b.cpu().contiguous().view(torch.uint8))


def packet_check(out,ref):
    checks=[same(a,b) for a,b in zip(out,ref)]
    assert all(checks),f'packet bytes differ from frozen CPU reference: packed/scale/sentinel={checks}'
    assert sum(a.numel()*a.element_size() for a in out)==4515841


def sources():
    files=[Path(__file__),HERE/'REGISTRATION_DRAFT_KO.txt',SOURCE/'common.py',SOURCE/'LOCK.json',
        SOURCE/'capture/manifest.json',SOURCE/'packets/manifest.json',SOURCE/'reference/manifest.json']
    for l,c in CASES:
        files.append(SOURCE/f'capture/l{l}_write{c}.pt')
        files.extend(SOURCE/f'packets/l{l}_c{c}_{a}.pt' for a in ALGS+('R_bf16',))
        files.append(Path(f'/home/acorn0415/sfq_data/kv_alllayer_capture/p00_s101/l{l}_c7.pt'))
    return files


def verified(path,manifest):
    r=next(x for x in manifest['artifacts'] if Path(x['path']).name==path.name)
    assert common.sha(path)==r['sha256'] and path.stat().st_size==r['bytes']
    return torch.load(path,map_location='cpu',weights_only=True)


def selftest():
    g=torch.Generator().manual_seed(20261105)
    z=torch.randn((1,1,16,12,128),generator=g).bfloat16()
    s=((z.abs().amax(2,keepdim=True)/7).clamp_min(1e-8)).half()
    a=torch.randn((12,128,128),generator=g).double();h=a@a.transpose(1,2)+torch.eye(128)*10
    u=common.factorize(h);before=common.tsha(z)
    ref=common.codes_obq(z,s,h,u);got=batch_codes(z,s,u)
    assert torch.equal(ref,got) and common.tsha(z)==before
    diag=torch.eye(128).repeat(12,1,1)
    assert torch.equal(batch_codes(z,s,diag),(z.float()/s.float()).round().clamp(-8,7).to(torch.int8))
    packed=common.pack(got);assert torch.equal(common.unpack(packed,got.shape),got)
    k=torch.randn((1,4680,12,128),generator=g).bfloat16();hm=common.hadamard()
    x,snative,ss=common.frames(k);x2,ss2=gpu_frames(k,hm)
    assert torch.equal(x,x2) and torch.equal(ss,ss2)
    for alg in ('R_fp32','R_bf16'):
        if alg=='R_fp32':expected=packet((x.float()/ss.float()).round().clamp(-8,7).to(torch.int8),ss)
        else:expected=cpu_tuple(common.encode_k(k,'R_bf16'))
        packet_check(full_packet(k,hm,diag,alg),expected)
    assert not torch.cuda.is_initialized()
    result={'passed':True,'code_sha256':common.sha(__file__),'actual_capture_loads':0,'gpu_initialized':False,
        'checks':['batch==head_loop_full_H_codes','input_preserved','identity_factor==RTN',
            'pack_roundtrip','CPU_preprocessing_exact','RTN_FP32/BF16_full_packet_exact']}
    common.save(HERE/'SELFTEST.json',result);print(json.dumps(result,indent=2))


def lock():
    p=HERE/'LOCK.json'
    if p.exists():raise RuntimeError('refuse overwrite lock')
    st=json.loads((HERE/'SELFTEST.json').read_text());assert st['passed'] and st['code_sha256']==common.sha(__file__)
    common.save(p,{'registered_utc':time.time(),'sha256':{str(p):common.sha(p) for p in sources()},
        'selftest_sha256':common.sha(HERE/'SELFTEST.json'),'cases':CASES,'seconds_cap':900})


def prepare_case(layer,chunk):
    start=time.monotonic();cm=json.loads((SOURCE/'capture/manifest.json').read_text())
    pm=json.loads((SOURCE/'packets/manifest.json').read_text());rm=json.loads((SOURCE/'reference/manifest.json').read_text())
    assert cm['status']==pm['status']==rm['status']=='completed'
    wp=SOURCE/f'capture/l{layer}_write{chunk}.pt';w=verified(wp,cm)
    calp=Path(f'/home/acorn0415/sfq_data/kv_alllayer_capture/p00_s101/l{layer}_c7.pt')
    cal=verified(calp,json.loads((calp.parent/'manifest.json').read_text()))
    th=time.monotonic();h=common.hessian(cal['q']);u=common.factorize(h);off_s=time.monotonic()-th
    info=next(x for x in pm['offline'] if x['layer']==layer);assert common.tsha(h)==info['H_sha256']
    tf=time.monotonic();z,s,sf=common.frames(w['k']);frame_s=time.monotonic()-tf
    expected={}
    for alg in ALGS+('R_bf16',):
        packet_ref=verified(SOURCE/f'packets/l{layer}_c{chunk}_{alg}.pt',pm)
        expected[alg]=cpu_tuple(packet_ref)
        packet_check(code_packet(z,sf,u,alg),expected[alg])
    bftime=next(x['seconds'] for x in rm['chunks'] if x['chunk']==chunk+1)
    data={'k':w['k'],'z':z,'s':sf,'u':u,'expected':expected,'layer':layer,'chunk':chunk,
        'offline_H_factor_CPU_seconds':off_s,'CPU_rotation_scale_seconds':frame_s,
        'CPU_prepare_verify_seconds':time.monotonic()-start,'BF16_reference_consumer_chunk_seconds':bftime,
        'source_write_sha256':common.sha(wp),'calibration_sha256':common.sha(calp)}
    path=HERE/f'prepared_l{layer}_c{chunk}.pt';assert not path.exists();torch.save(data,path)
    return {'path':str(path),'sha256':common.sha(path),'bytes':path.stat().st_size,
        **{k:v for k,v in data.items() if not isinstance(v,torch.Tensor) and k!='expected'}}


class GraphUnavailable(RuntimeError):pass


def gpu_peak_guard(mode):
    peak=torch.cuda.max_memory_allocated()
    if peak>CAP_BYTES:
        message=f'GPU peak allocated resource cap exceeded: {peak} > {CAP_BYTES}'
        if mode=='graph':raise GraphUnavailable(message)
        raise MemoryError(message)


def build_functions(stage,mode,x,hm,u):
    funcs={};graphs=[];t0=time.perf_counter();copy_bytes=0
    for alg in ALGS:
        if mode=='eager':
            if stage=='T_code':funcs[alg]=lambda alg=alg:code_packet(x[0],x[1],u,alg)
            else:funcs[alg]=lambda alg=alg:full_packet(x[0],hm,u,alg)
            continue
        static=[a.clone() for a in x]
        copy_bytes=sum(a.numel()*a.element_size() for a in x)
        def execute(alg=alg,static=static):
            return code_packet(static[0],static[1],u,alg) if stage=='T_code' else full_packet(static[0],hm,u,alg)
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):warm=execute()
        torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();del warm
        graph=torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):out=execute()
        except RuntimeError as exc:raise GraphUnavailable(str(exc)) from exc
        gpu_peak_guard(mode)
        def replay(static=static,graph=graph,out=out):
            for dest,source in zip(static,x):dest.copy_(source)
            graph.replay();return out
        funcs[alg]=replay;graphs.append(graph)
    torch.cuda.synchronize()
    gpu_peak_guard(mode)
    return funcs,{'setup_seconds':time.perf_counter()-t0,'replay_input_copy_bytes':copy_bytes,
        'setup_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'setup_peak_reserved_bytes':torch.cuda.max_memory_reserved()}


def quartiles(values):
    qs=statistics.quantiles(values,n=4,method='inclusive')
    return {'median':statistics.median(values),'q1':qs[0],'q3':qs[2],'iqr':qs[2]-qs[0]}


def benchmark_stage(stage,mode,x,hm,u,expected):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    try:funcs,setup=build_functions(stage,mode,x,hm,u)
    except RuntimeError as exc:
        if mode=='graph':raise GraphUnavailable(str(exc)) from exc
        raise
    for alg in ALGS:packet_check(funcs[alg](),expected[alg])
    for _ in range(5):
        for alg in ALGS:funcs[alg]()
    torch.cuda.synchronize()
    gpu_peak_guard(mode)
    rows=[]
    for rep in range(20):
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:
            raise MemoryError('8GiB CPU worker RSS cap')
        order=ALGS if rep%2==0 else ALGS[::-1];sample={'round':rep,'order':order,'times':{}}
        for alg in order:
            torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            tick=time.perf_counter();start.record();out=funcs[alg]();end.record();end.synchronize()
            sample['times'][alg]={'gpu_seconds':start.elapsed_time(end)/1000,'host_seconds':time.perf_counter()-tick}
        rows.append(sample)
        gpu_peak_guard(mode)
    for alg in ALGS:packet_check(funcs[alg](),expected[alg])
    summary={alg:{kind:quartiles([r['times'][alg][kind] for r in rows]) for kind in ('gpu_seconds','host_seconds')} for alg in ALGS}
    paired={kind:quartiles([r['times']['O_offline'][kind]-r['times']['R_fp32'][kind] for r in rows]) for kind in ('gpu_seconds','host_seconds')}
    return {'status':'completed','stage':stage,'mode':mode,'correctness_all_packet_bytes':True,'setup':setup,
        'samples':rows,'summary':summary,'paired_OBQ_minus_RTN':paired,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}


def worker(prepared,mode,dest):
    start=time.monotonic();data=torch.load(prepared,map_location='cpu',weights_only=True)
    result={'status':'running','mode':mode,'layer':data['layer'],'chunk':data['chunk'],'stages':{},
        'started_utc':time.time(),'prepared_sha256':common.sha(prepared),'eager_code_validation':{}}
    common.save(dest,result)
    try:
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
        tx=time.monotonic();k,z,s,u=[data[a].cuda() for a in ('k','z','s','u')];hm=common.hadamard().cuda()
        torch.cuda.synchronize();result['initial_H2D_seconds']=time.monotonic()-tx
        result['device']=torch.cuda.get_device_name();result['capability']=torch.cuda.get_device_capability()
        result['torch_version']=torch.__version__;result['cuda_runtime']=torch.version.cuda
        result['resident_input_bytes']={name:t.numel()*t.element_size() for name,t in zip(('k','z','s','u','hadamard'),(k,z,s,u,hm))}
        initial={name:common.tsha(t.cpu()) for name,t in zip(('k','z','s','u'),(k,z,s,u))}
        # GPU eager must match the actual frozen CPU packet, even for graph workers.
        for alg in ALGS+('R_bf16',):
            packet_check(code_packet(z,s,u,alg),data['expected'][alg])
            result['eager_code_validation'][alg]='all_packet_bytes_equal'
        result['initial_validation_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        gpu_peak_guard('eager')
        for stage,x in (('T_code',(z,s)),('T_encoder',(k,))):
            if stage=='T_encoder':
                zg,sg=gpu_frames(k,hm)
                check={'rotated_bytes_equal':same(zg,data['z']),'scale_bytes_equal':same(sg,data['s']),
                    'rotated_unequal_elements':int((zg.cpu()!=data['z']).sum()),'scale_unequal_elements':int((sg.cpu()!=data['s']).sum())}
                if not check['rotated_bytes_equal'] or not check['scale_bytes_equal']:
                    result['stages'][stage]={'status':'unsupported_numerical_equivalence','preprocess':check}
                    del zg,sg;continue
                for alg in ALGS+('R_bf16',):packet_check(full_packet(k,hm,u,alg),data['expected'][alg])
                del zg,sg
            try:result['stages'][stage]=benchmark_stage(stage,mode,x,hm,u,data['expected'])
            except GraphUnavailable as exc:
                result['stages'][stage]={'status':'graph_unavailable','reason':str(exc)}
            common.save(dest,result);gc.collect();torch.cuda.empty_cache()
        assert initial=={name:common.tsha(t.cpu()) for name,t in zip(('k','z','s','u'),(k,z,s,u))}
        result['input_bytes_preserved']=True;result['status']='completed'
    except BaseException as exc:
        result.update(status='invalid',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(finished_utc=time.time(),elapsed_seconds=time.monotonic()-start,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            last_stage_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            last_stage_gpu_cap_exceeded=(torch.cuda.max_memory_allocated()>CAP_BYTES) if torch.cuda.is_initialized() else None)
        common.save(dest,result)


def run():
    target=HERE/'RESULT.json'
    if target.exists():raise RuntimeError('refuse overwrite result')
    start=time.monotonic();reg=json.loads((HERE/'LOCK.json').read_text())
    assert all(common.sha(p)==h for p,h in reg['sha256'].items())
    assert common.sha(HERE/'SELFTEST.json')==reg['selftest_sha256']
    result={'status':'running','started_utc':time.time(),'lock_sha256':common.sha(HERE/'LOCK.json'),
        'prepared':[],'workers':[],'models_loaded':0,'generations':0}
    common.save(target,result)
    def guard():
        if time.monotonic()-start>=900:raise TimeoutError('900-second total cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:raise MemoryError('8GiB CPU RSS cap')
        if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())>2*1024**3:raise OSError('2GiB disk cap')
    try:
        for layer,chunk in CASES:
            guard();item=prepare_case(layer,chunk);result['prepared'].append(item);common.save(target,result)
            guard()
            for mode in ('eager','graph'):
                guard();dest=HERE/f'l{layer}_c{chunk}_{mode}.json';log=HERE/f'l{layer}_c{chunk}_{mode}.log'
                assert not dest.exists()
                assert common.sha(item['path'])==item['sha256']
                with log.open('w') as out:
                    completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker',
                        '--prepared',item['path'],'--mode',mode,'--output',str(dest)],stdout=out,stderr=subprocess.STDOUT,
                        timeout=max(1,900-(time.monotonic()-start)),check=False)
                wd=json.loads(dest.read_text()) if dest.exists() else {'status':'missing_worker_output'}
                assert common.sha(item['path'])==item['sha256']
                if wd['status']=='completed':assert wd['prepared_sha256']==item['sha256']
                result['workers'].append({'layer':layer,'chunk':chunk,'mode':mode,'path':str(dest),
                    'sha256':common.sha(dest) if dest.exists() else None,'exit_code':completed.returncode,
                    'status':wd['status']});common.save(target,result)
                if completed.returncode or wd['status']!='completed':raise RuntimeError(f'worker {layer}/{chunk}/{mode} invalid')
                print('WORKER_DONE',layer,chunk,mode,{k:v['status'] for k,v in wd['stages'].items()},flush=True)
            guard()
        result['status']='completed'
    except BaseException as exc:
        result.update(status='incomplete_or_invalid',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(finished_utc=time.time(),elapsed_seconds=time.monotonic()-start,
            cpu_parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            new_directory_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()))
        common.save(target,result)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['selftest','lock','run','worker'])
    p.add_argument('--prepared',type=Path);p.add_argument('--mode',choices=['eager','graph']);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.action=='worker':worker(a.prepared,a.mode,a.output)
    else:{'selftest':selftest,'lock':lock,'run':run}[a.action]()

