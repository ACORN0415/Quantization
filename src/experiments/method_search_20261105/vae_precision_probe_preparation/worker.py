"""Preparation only. Root LOCK is required; exactly two fresh-checkpoint decodes."""
import argparse, ast, collections, gc, hashlib, importlib.util, json, math
import os, resource, signal, sys, threading, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'src'
LATENT = Path('/home/acorn0415/sfq_data/mechanism/fr/p00_s303_BF16/latent.pt')
CHECKPOINT = SOURCE / 'wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
SELECTED = (0,80,160,240,320,400,480,500)
LIMITS = {'wall_seconds':600,'CPU_threads':4,'RSS_bytes':12*1024**3,
          'GPU_allocator_bytes':22*1024**3,'output_bytes':256*1024**2,
          'VAE_decodes':2,'generator_calls':0}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()


def save(path, data):
    tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def output_bytes(folder):
    total=0
    for path in Path(folder).rglob('*'):
        try:
            if path.is_file(): total+=path.stat().st_size
        except FileNotFoundError:
            # Atomic writer may rename a .tmp between glob and stat.
            continue
    return total


def read_lock(path):
    d=json.loads(Path(path).read_text())
    assert d['root_execution_authorized'] is True
    assert d['status']=='approved_for_one_two_arm_precision_probe'
    assert d['limits']==LIMITS
    assert d['decision_rule']=='late_RGB_RMSE_ge_0.02_sensitivity_only'
    required=[HERE/'worker.py',HERE/'PROPOSAL_KO.txt',HERE/'SELFTEST_V2.json',
              SOURCE/'utils/wan_wrapper.py',SOURCE/'wan/modules/vae.py',LATENT,CHECKPOINT]
    for p in required:
        assert str(p) in d['sha256'],f'unbound input {p}'
    return d


def verify_sources(lock):
    for p,s in lock['sha256'].items():
        assert sha(p)==s,f'source/input hash mismatch: {p}'


def exact_wrapper_class(torch):
    """Execute unchanged WanVAEWrapper AST with the unchanged native VAE module."""
    path=SOURCE/'wan/modules/vae.py'
    spec=importlib.util.spec_from_file_location('precision_probe_native_vae',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    tree=ast.parse((SOURCE/'utils/wan_wrapper.py').read_text())
    cls=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WanVAEWrapper']
    assert len(cls)==1
    module=ast.Module(body=cls,type_ignores=[])
    scope={'torch':torch,'_video_vae':mod._video_vae,'__name__':'precision_probe_native_wrapper'}
    exec(compile(ast.fix_missing_locations(module),str(SOURCE/'utils/wan_wrapper.py'),'exec'),scope)
    return scope['WanVAEWrapper'], hashlib.sha256(ast.dump(cls[0],include_attributes=False).encode()).hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def statistics(a,b):
    """RGB[0,1] TCHW float32 CPU; exact per-frame order statistic, no subsampling."""
    import numpy as np
    assert a.shape==b.shape and len(a.shape)==4 and a.shape[1]==3
    assert a.dtype==b.dtype==np.float32
    rows=[]
    for i in range(len(a)):
        aa=a[i].astype(np.float64);bb=b[i].astype(np.float64);delta=bb-aa
        ad=np.abs(delta)
        rows.append({'frame':i,'MSE':float(np.mean(delta*delta)),
                     'MAE':float(np.mean(ad)),'q99_abs':float(np.quantile(ad,0.99,method='linear')),
                     'mean_RGB_bf16':aa.mean(axis=(1,2)).tolist(),
                     'mean_RGB_fp32':bb.mean(axis=(1,2)).tolist()})
    return rows


def summarize(rows):
    import numpy as np
    assert len(rows)==501
    late_rmse=math.sqrt(sum(r['MSE'] for r in rows[249:501])/252)
    def drift(key):
        means=np.array([r[key] for r in rows],dtype=np.float64)
        v=means[375:501].mean(0)-means[0:126].mean(0)
        return {'vector':v.tolist(),'L2':float(np.linalg.norm(v))}
    bf=drift('mean_RGB_bf16');fp=drift('mean_RGB_fp32')
    return {'late_frames':[249,500],'late_RGB_RMSE':late_rmse,
            'early_mean_frames':[0,125],'late_mean_frames':[375,500],
            'mean_RGB_change_bf16':bf,'mean_RGB_change_fp32':fp,
            'mean_RGB_change_reduction_descriptive':None if bf['L2']==0 else 1-fp['L2']/bf['L2'],
            'decision':'review_precision_sensitivity' if late_rmse>=0.02 else 'stop_precision_expansion',
            'decision_scope':'engineering sensitivity only; FP32 is not visual ground truth; no quality/causal defect claim'}


def run(lock_path):
    # All source validation, imports, loading, two decodes, I/O and statistics share this clock.
    started=time.time();deadline=started+LIMITS['wall_seconds']
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    lock=read_lock(lock_path)  # Reject missing/unapproved contract without writing RUN.
    if (HERE/'RUN.json').exists(): raise RuntimeError('one attempt only')
    with (HERE/'RUN.json').open('x') as f:
        json.dump({'status':'started','started_utc':started,'absolute_deadline_utc':deadline,'attempt':1},f)
    state={'status':'running','started_utc':started,'deadline_utc':deadline,'vae_decodes':0,
           'generator_calls':0,'arms':[],'limits':LIMITS,'monitor_peak_RSS_bytes':0,
           'monitor_peak_GPU_allocator_bytes':0,'monitor_peak_output_bytes':0}
    stop=threading.Event();torch_ref=[None];watchdog_errors=[]
    import psutil
    process=psutil.Process()
    def monitor_loop():
        while not stop.wait(.1):
            reason=None
            rss=process.memory_info().rss
            disk=output_bytes(HERE)
            state['monitor_peak_RSS_bytes']=max(state['monitor_peak_RSS_bytes'],rss)
            state['monitor_peak_output_bytes']=max(state['monitor_peak_output_bytes'],disk)
            if time.time()>deadline: reason='wall cap'
            if rss>LIMITS['RSS_bytes']: reason='RSS cap'
            if disk>LIMITS['output_bytes']: reason='disk cap'
            t=torch_ref[0]
            if t is not None and t.cuda.is_initialized():
                peak=t.cuda.max_memory_allocated()
                state['monitor_peak_GPU_allocator_bytes']=max(state['monitor_peak_GPU_allocator_bytes'],peak)
                if peak>LIMITS['GPU_allocator_bytes']:reason='GPU allocator cap'
            if reason:
                save(HERE/'WATCHDOG_FAILURE.json',{'status':'inconclusive_resource_limit','reason':reason,'observed':dict(state),'utc':time.time()})
                os._exit(124)
    def watchdog():
        try:
            monitor_loop()
        except BaseException as exc:
            watchdog_errors.append(repr(exc))
            try:
                save(HERE/'WATCHDOG_FAILURE.json',{'status':'inconclusive_monitor_failure','error':repr(exc),'utc':time.time()})
            finally:
                os._exit(125)
    watcher=threading.Thread(target=watchdog,daemon=True);watcher.start()
    def interrupt(sig,frame): raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        verify_sources(lock)
        import torch
        import numpy as np
        from PIL import Image
        torch_ref[0]=torch
        torch.set_num_threads(4);torch.set_num_interop_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        assert torch.cuda.device_count()>=1
        total=torch.cuda.get_device_properties(0).total_memory
        assert total>=LIMITS['GPU_allocator_bytes']
        torch.cuda.set_per_process_memory_fraction(LIMITS['GPU_allocator_bytes']/total,0)
        torch.cuda.reset_peak_memory_stats()
        state['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                          'GPU':torch.cuda.get_device_name(0),'TF32_matmul':False,'TF32_cudnn':False,
                          'autocast':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
        cls,astsha=exact_wrapper_class(torch);state['wrapper_AST_sha256']=astsha
        os.chdir(SOURCE)
        loaded=torch.load(LATENT,map_location='cpu',weights_only=True)
        latent=loaded['latent'];del loaded
        assert latent.device.type=='cpu' and latent.shape==(1,126,16,60,104)
        assert latent.dtype==torch.bfloat16 and torch.isfinite(latent).all()
        latent_sha=tensor_sha(latent);state['latent_tensor_sha256']=latent_sha
        videos=[]
        for name,dtype in [('BF16',torch.bfloat16),('FP32',torch.float32)]:
            t0=time.monotonic()
            assert sha(CHECKPOINT)==lock['sha256'][str(CHECKPOINT)]
            vae=cls().eval().requires_grad_(False)  # New checkpoint load in each arm, never BF16->FP32 reuse.
            hist=collections.Counter(str(p.dtype) for p in vae.model.parameters())
            assert all(p.dtype==torch.float32 for p in vae.model.parameters() if p.is_floating_point()),'checkpoint not native FP32'
            vae=vae.to(device='cuda',dtype=dtype)
            z=latent.to(device='cuda',dtype=dtype)
            assert torch.equal(z.cpu().float(),latent.float()),'latent numeric values changed'
            arm={'name':name,'fresh_checkpoint_loaded':True,'loaded_parameter_dtype_histogram':dict(hist),
                 'checkpoint_sha256':lock['sha256'][str(CHECKPOINT)],'latent_dtype':str(dtype),
                 'mean_std_rule':'native wrapper CPU FP32 attributes -> latent dtype; reciprocal in latent dtype',
                 'use_cache':False,'setup_seconds':time.monotonic()-t0}
            state['vae_decodes']+=1;assert state['vae_decodes']<=2
            torch.cuda.synchronize();td=time.monotonic()
            with torch.inference_mode(),torch.autocast('cuda',enabled=False),torch.autocast('cpu',enabled=False):
                decoded=vae.decode_to_pixel(z,use_cache=False)
            torch.cuda.synchronize();arm['decode_seconds']=time.monotonic()-td
            assert decoded.dtype==torch.float32 and decoded.shape==(1,501,3,480,832)
            # No second whole-video GPU normalization temporary; same FP32 affine/clamp on CPU.
            cpu=decoded[0].cpu()
            del decoded,z
            vae.model.clear_cache();del vae
            gc.collect();torch.cuda.empty_cache()
            assert torch.isfinite(cpu).all()
            cpu.mul_(.5).add_(.5).clamp_(0,1)
            videos.append(cpu)
            arm['raw_RGB_sha256']=None  # Whole RGB deliberately retained only in memory.
            arm['GPU_peak_allocated_bytes_cumulative']=torch.cuda.max_memory_allocated()
            arm['GPU_peak_reserved_bytes_cumulative']=torch.cuda.max_memory_reserved()
            selected=cpu[list(SELECTED)].contiguous()
            floatpath=HERE/f'{name}_selected_RGB.pt';torch.save({'frames':SELECTED,'RGB':selected},floatpath)
            files=[{'path':str(floatpath),'sha256':sha(floatpath),'bytes':floatpath.stat().st_size}]
            for j,frame in enumerate(SELECTED):
                png=HERE/f'{name}_frame{frame:03d}.png'
                im=(selected[j].permute(1,2,0)*255).to(torch.uint8).numpy()
                Image.fromarray(im,mode='RGB').save(png)
                files.append({'path':str(png),'sha256':sha(png),'bytes':png.stat().st_size})
            del selected,im
            arm['selected_artifacts']=files;state['arms'].append(arm)
            save(HERE/'PROGRESS.json',state)
        assert state['vae_decodes']==2 and tensor_sha(latent)==latent_sha
        rows=statistics(videos[0].numpy(),videos[1].numpy())
        state['per_frame']=rows;state['summary']=summarize(rows)
        del videos,latent;gc.collect()
        verify_sources(lock)
        assert time.time()<=deadline
        state['process_lifetime_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        assert state['process_lifetime_peak_RSS_bytes']<=LIMITS['RSS_bytes']
        assert torch.cuda.max_memory_allocated()<=LIMITS['GPU_allocator_bytes']
        assert output_bytes(HERE)<=LIMITS['output_bytes']
        assert watcher.is_alive() and not watchdog_errors,'watchdog did not remain healthy'
        stop.set();watcher.join(timeout=1)
        assert not watcher.is_alive() and not watchdog_errors,'watchdog shutdown failed'
        state['watchdog_completed_cleanly']=True
        assert time.time()<=deadline
        state.update(status='completed_two_arm_sensitivity_diagnostic',finished_utc=time.time(),
                     inputs_unchanged=True,whole_RGB_saved_to_disk=False)
        save(HERE/'RESULT.json',state)
    except BaseException as exc:
        state.update(status='inconclusive_invalid_or_resource_failure',finished_utc=time.time(),
                     exception=repr(exc),traceback=traceback.format_exc())
        save(HERE/'FAILURE.json',state)
        raise
    finally:
        stop.set();watcher.join(timeout=1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lock',required=True)
    args=parser.parse_args();run(args.lock)
