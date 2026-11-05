"""Two-arm BF16 native/output-streaming execution baseline; root LOCK required."""
import argparse, ast, collections, copy, gc, hashlib, importlib.util, json
import os, resource, signal, sys, threading, time, traceback, types
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]; SOURCE=ROOT/'src'
LATENT=Path('/home/acorn0415/sfq_data/mechanism/fr/p00_s303_BF16/latent.pt')
CHECKPOINT=SOURCE/'wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
LIMITS={'wall_seconds':600,'CPU_threads':4,'RSS_bytes':12*1024**3,
        'GPU_allocator_bytes':22*1024**3,'output_bytes':256*1024**2,
        'whole_video_decodes':2,'decoder_forward_calls':252,'generator_calls':0}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def save(path,data):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def disk_bytes(folder):
    n=0
    for p in Path(folder).rglob('*'):
        try:
            if p.is_file():n+=p.stat().st_size
        except FileNotFoundError:pass
    return n

def dump(n):return ast.dump(n,include_attributes=False)
def stmt(s):return ast.parse(s).body[0]

def native_decode_ast():
    tree=ast.parse((SOURCE/'wan/modules/vae.py').read_text())
    cls=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='WanVAE_']
    assert len(cls)==1
    f=[x for x in cls[0].body if isinstance(x,ast.FunctionDef) and x.name=='decode']
    assert len(f)==1
    return copy.deepcopy(f[0])

def transform_decode(native):
    """Only output accumulation/lifetime changes; invert transformation to verify."""
    fn=copy.deepcopy(native)
    loops=[x for x in fn.body if isinstance(x,ast.For)]
    assert len(loops)==1;loop=loops[0]
    assert dump(loop.target)==dump(ast.Name(id='i',ctx=ast.Store()))
    assert dump(loop.iter)==dump(ast.parse('range(iter_)',mode='eval').body)
    assert len(loop.body)==2 and dump(loop.body[0])==dump(stmt('self._conv_idx = [0]'))
    branch=loop.body[1];assert isinstance(branch,ast.If)
    assert dump(branch.test)==dump(ast.parse('i == 0',mode='eval').body)
    assert len(branch.body)==1 and len(branch.orelse)==2
    first,last=branch.body[0],branch.orelse[0]
    assert dump(first)==dump(stmt('out = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)'))
    assert dump(last)==dump(stmt('out_ = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)'))
    oldcat=stmt('out = torch.cat([out, out_], 2)')
    assert dump(branch.orelse[1])==dump(oldcat)
    assert dump(fn.body[0])==dump(stmt('self.clear_cache()'))
    assert dump(fn.body[-2])==dump(stmt('self.clear_cache()'))
    assert dump(fn.body[-1])==dump(stmt('return out'))
    branch.body.extend([stmt('_emit_slab(out, i)'),stmt('del out')])
    branch.orelse[1:]=[stmt('_emit_slab(out_, i)'),stmt('del out_')]
    fn.body[-1]=stmt('return None')
    inverse=copy.deepcopy(fn);il=[x for x in inverse.body if isinstance(x,ast.For)][0]
    ib=il.body[1];ib.body=ib.body[:1];ib.orelse=ib.orelse[:1]+[oldcat]
    inverse.body[-1]=stmt('return out')
    assert dump(inverse)==dump(native),'non-whitelisted AST change'
    return ast.fix_missing_locations(fn),{'original_ast_sha256':hashlib.sha256(dump(native).encode()).hexdigest(),
        'stream_ast_sha256':hashlib.sha256(dump(fn).encode()).hexdigest(),'inverse_equals_native':True,
        'python':sys.version,'whitelist':['first output emit+delete','replace later output cat with emit+delete','return None']}

def compile_decode(torch,fn,emit=None):
    env={'torch':torch,'_emit_slab':emit}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn)],type_ignores=[])),str(SOURCE/'wan/modules/vae.py'),'exec'),env)
    return env['decode']

def exact_wrapper(torch):
    p=SOURCE/'wan/modules/vae.py'
    spec=importlib.util.spec_from_file_location('stream_probe_native_vae',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    tree=ast.parse((SOURCE/'utils/wan_wrapper.py').read_text())
    cls=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WanVAEWrapper'];assert len(cls)==1
    env={'torch':torch,'_video_vae':mod._video_vae,'__name__':'stream_probe_wrapper'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=cls,type_ignores=[])),str(SOURCE/'utils/wan_wrapper.py'),'exec'),env)
    # Copy precisely the native wrapper's pure input preparation assignments.
    dec=[n for n in cls[0].body if isinstance(n,ast.FunctionDef) and n.name=='decode_to_pixel'][0]
    prep=[n for n in dec.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('zs','scale') or isinstance(t,ast.Tuple) and [e.id for e in t.elts]==['device','dtype'] for t in n.targets)]
    assert len(prep)==3
    f=ast.parse('def prepare(self, latent):\n    pass').body[0]
    f.body=copy.deepcopy(prep)+[stmt('return zs, scale')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'native_wrapper_input_assignments','exec'),env)
    return env['WanVAEWrapper'],env['prepare'],hashlib.sha256(dump(cls[0]).encode()).hexdigest()

def normalize_slab(out):
    """BCTHW BF16 -> FP32 RGB [0,1]; all arithmetic stays on input device."""
    import torch
    assert out.ndim==5 and out.shape[0]==1 and out.shape[1]==3 and out.dtype==torch.bfloat16
    return out.float().clamp_(-1,1).mul_(.5).add_(.5).clamp_(0,1)

def flat_bytes(t):
    import torch
    assert t.device.type=='cpu'
    x=t.detach().contiguous().reshape(-1)
    if x.stride(0)!=1:x=torch.empty(x.numel(),dtype=x.dtype).copy_(x)
    return x.view(torch.uint8)

def frame_digest(t):
    return hashlib.sha256(memoryview(flat_bytes(t).numpy())).hexdigest()

def bytes_equal(a,b):
    import torch
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(flat_bytes(a),flat_bytes(b))

def cache_summary(model):
    import torch
    slots=model._feat_map;groups={};entries=[];marks=[]
    for i,x in enumerate(slots):
        if not isinstance(x,torch.Tensor):
            marks.append({'slot':i,'value':x});continue
        s=x.untyped_storage();key=(str(x.device),s.data_ptr())
        if key not in groups:groups[key]={'group':len(groups),'bytes':s.nbytes()}
        entries.append({'slot':i,'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device),
                        'logical_bytes':x.numel()*x.element_size(),'storage_group':groups[key]['group']})
    return {'list_slots':len(slots),'tensor_count':len(entries),'unique_storage_count':len(groups),
            'logical_bytes':sum(x['logical_bytes'] for x in entries),'unique_storage_bytes':sum(x['bytes'] for x in groups.values()),
            'entries':entries,'non_tensor_slots':marks}

def observe(model):
    rows={'conv2':[],'decoder':[],'cache_samples':[],'clear':[]};handles=[]
    def c2(m,args,out):rows['conv2'].append({'input_shape':list(args[0].shape),'output_shape':list(out.shape),'dtype':str(out.dtype)})
    def dec(m,args,out):
        idx=len(rows['decoder']);rows['decoder'].append({'index':idx,'input_shape':list(args[0].shape),'output_shape':list(out.shape),'dtype':str(out.dtype)})
        if idx in (0,64,125):rows['cache_samples'].append({'index':idx,**cache_summary(model)})
    handles.extend([model.conv2.register_forward_hook(c2),model.decoder.register_forward_hook(dec)])
    original=model.clear_cache
    def clear(self):
        before=cache_summary(self);original();after=cache_summary(self)
        rows['clear'].append({'before_tensor_count':before['tensor_count'],'after_tensor_count':after['tensor_count']})
    model.clear_cache=types.MethodType(clear,model)
    return rows,handles

def assert_trace(rows):
    assert rows['conv2']==[{'input_shape':[1,16,126,60,104],'output_shape':[1,16,126,60,104],'dtype':'torch.bfloat16'}]
    assert len(rows['decoder'])==126
    for i,r in enumerate(rows['decoder']):
        assert r=={'index':i,'input_shape':[1,16,1,60,104],'output_shape':[1,3,1 if i==0 else 4,480,832],'dtype':'torch.bfloat16'}
    assert rows['clear']==[{'before_tensor_count':0,'after_tensor_count':0},{'before_tensor_count':32,'after_tensor_count':0}]
    assert [x['index'] for x in rows['cache_samples']]==[0,64,125]
    assert [x['tensor_count'] for x in rows['cache_samples']]==[30,32,32]
    assert [x['unique_storage_bytes'] for x in rows['cache_samples']]==[920325120,1888573440,1888573440]

def read_lock(path):
    d=json.loads(Path(path).read_text())
    assert d['root_execution_authorized'] is True and d['status']=='approved_for_one_native_streaming_probe'
    assert d['limits']==LIMITS and d['gate']=='all501_RGB_logical_bytes_equal_and_trace_contract'
    required=[HERE/'worker.py',HERE/'selftest.py',HERE/'SELFTEST.json',HERE/'PROPOSAL_KO.txt',HERE/'ROOT_SPEC.json',SOURCE/'utils/wan_wrapper.py',SOURCE/'wan/modules/vae.py',LATENT,CHECKPOINT]
    for p in required:assert str(p) in d['sha256'],f'unbound input/source: {p}'
    return d

def verify_sources(lock):
    for p,s in lock['sha256'].items():assert sha(p)==s,f'source/input changed: {p}'

def run(lock_path):
    started=time.time();deadline=started+LIMITS['wall_seconds']
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    lock=read_lock(lock_path)
    with (HERE/'RUN.json').open('x') as f:json.dump({'status':'started','started_utc':started,'absolute_deadline_utc':deadline,'attempt':1},f)
    state={'status':'running','started_utc':started,'deadline_utc':deadline,'arms':[],'limits':LIMITS,
           'whole_video_decodes':0,'decoder_forward_calls':0,'generator_calls':0,
           'monitor_peak_RSS_bytes':0,'monitor_peak_GPU_allocator_bytes':0,'monitor_peak_output_bytes':0}
    stop=threading.Event();torch_ref=[None];errors=[]
    import psutil
    process=psutil.Process()
    def monitor():
        try:
            while not stop.wait(.1):
                rss=process.memory_info().rss;disk=disk_bytes(HERE);reason=None
                state['monitor_peak_RSS_bytes']=max(state['monitor_peak_RSS_bytes'],rss)
                state['monitor_peak_output_bytes']=max(state['monitor_peak_output_bytes'],disk)
                if time.time()>deadline:reason='wall cap'
                if rss>LIMITS['RSS_bytes']:reason='RSS cap'
                if disk>LIMITS['output_bytes']:reason='disk cap'
                t=torch_ref[0]
                if t is not None and t.cuda.is_initialized():
                    peak=t.cuda.max_memory_allocated();state['monitor_peak_GPU_allocator_bytes']=max(state['monitor_peak_GPU_allocator_bytes'],peak)
                    if peak>LIMITS['GPU_allocator_bytes']:reason='GPU allocator cap'
                if reason:
                    save(HERE/'WATCHDOG_FAILURE.json',{'status':'inconclusive_resource_limit','reason':reason,'observed':dict(state),'utc':time.time()});os._exit(124)
        except BaseException as exc:
            errors.append(repr(exc))
            try:save(HERE/'WATCHDOG_FAILURE.json',{'status':'inconclusive_monitor_failure','error':repr(exc),'utc':time.time()})
            finally:os._exit(125)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    def interrupt(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    try:
        verify_sources(lock)
        import torch
        torch_ref[0]=torch;torch.set_num_threads(4);torch.set_num_interop_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        total=torch.cuda.get_device_properties(0).total_memory;assert total>=LIMITS['GPU_allocator_bytes']
        torch.cuda.set_per_process_memory_fraction(LIMITS['GPU_allocator_bytes']/total,0)
        state['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'GPU':torch.cuda.get_device_name(0),'TF32':False,'autocast':False,'benchmark':False,'deterministic':True}
        cls,prepare,wrapper_sha=exact_wrapper(torch);state['wrapper_AST_sha256']=wrapper_sha
        fn,astrecord=transform_decode(native_decode_ast());state['decode_AST_contract']=astrecord
        os.chdir(SOURCE)
        obj=torch.load(LATENT,map_location='cpu',weights_only=True);latent=obj['latent'];del obj
        assert latent.device.type=='cpu' and latent.shape==(1,126,16,60,104) and latent.dtype==torch.bfloat16
        assert torch.isfinite(latent).all();latent_before=frame_digest(latent)
        reference=None;frame_rows=[]
        for name in ('native_BF16','stream_BF16'):
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
            assert torch.cuda.memory_allocated()==0,'previous arm GPU storage retained'
            torch.cuda.reset_peak_memory_stats()
            arm={'name':name,'peak_scope':'fresh arm including model load,H2D,decode,normalization,CPU emit; reset after previous GPU storage freed'}
            setup=time.monotonic();assert sha(CHECKPOINT)==lock['sha256'][str(CHECKPOINT)]
            vae=cls().eval().requires_grad_(False)
            assert all(p.dtype==torch.float32 for p in vae.model.parameters() if p.is_floating_point())
            vae=vae.to(device='cuda',dtype=torch.bfloat16);z=latent.to('cuda')
            rows,handles=observe(vae.model)
            arm['setup_seconds']=time.monotonic()-setup;arm['fresh_checkpoint_loaded']=True
            cursor=[0];hashes=[]
            def consume(rgb):
                assert rgb.dtype==torch.float32 and rgb.device.type=='cuda' and rgb.shape[0]==1 and rgb.shape[1]==3
                for j in range(rgb.shape[2]):
                    frame=rgb[0,:,j].to('cpu').contiguous();idx=cursor[0]
                    assert idx<501 and frame.shape==(3,480,832) and torch.isfinite(frame).all()
                    digest=frame_digest(frame);hashes.append(digest)
                    if name=='native_BF16':reference[idx].copy_(frame)
                    else:
                        exact=bytes_equal(reference[idx],frame)
                        frame_rows.append({'frame':idx,'native_sha256':state['arms'][0]['frame_sha256'][idx],'stream_sha256':digest,'logical_bytes_equal':exact})
                        assert exact,f'RGB logical byte mismatch at frame {idx}'
                    cursor[0]+=1
            def emit(raw,idx):
                assert idx==len(rows['decoder'])-1
                rgb=normalize_slab(raw);consume(rgb)
                del rgb
            if name=='native_BF16':reference=torch.empty((501,3,480,832),dtype=torch.float32,device='cpu')
            state['whole_video_decodes']+=1;assert state['whole_video_decodes']<=2
            torch.cuda.synchronize();t0=time.monotonic()
            with torch.inference_mode(),torch.autocast('cuda',enabled=False),torch.autocast('cpu',enabled=False):
                if name=='native_BF16':
                    decoded=vae.decode_to_pixel(z,use_cache=False)
                    assert decoded.shape==(1,501,3,480,832) and decoded.dtype==torch.float32
                    decoded.mul_(.5).add_(.5).clamp_(0,1)
                    consume(decoded.permute(0,2,1,3,4));del decoded
                else:
                    zs,scale=prepare(vae,z)
                    assert zs.shape==(1,16,126,60,104)
                    stream=compile_decode(torch,fn,emit)
                    assert stream(vae.model,zs,scale) is None
                    del stream,zs,scale
            torch.cuda.synchronize();arm['decode_emit_validation_seconds']=time.monotonic()-t0
            assert cursor[0]==501;assert_trace(rows)
            state['decoder_forward_calls']+=len(rows['decoder'])
            arm['trace']=rows;arm['frame_sha256']=hashes
            arm['GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated();arm['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            state['monitor_peak_GPU_allocator_bytes']=max(state['monitor_peak_GPU_allocator_bytes'],arm['GPU_peak_allocated_bytes'])
            assert arm['GPU_peak_allocated_bytes']<=LIMITS['GPU_allocator_bytes']
            for h in handles:h.remove()
            # Clear hook closures and model self-cycle explicitly before next peak reset.
            vae.model.clear_cache=types.MethodType(type(vae.model).clear_cache,vae.model)
            del vae,z,handles,consume,emit;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
            assert torch.cuda.memory_allocated()==0
            state['arms'].append(arm);save(HERE/'PROGRESS.json',state)
        assert state['whole_video_decodes']==2 and state['decoder_forward_calls']==252
        assert len(frame_rows)==501 and all(x['logical_bytes_equal'] for x in frame_rows)
        assert state['arms'][0]['trace']==state['arms'][1]['trace']
        assert frame_digest(latent)==latent_before
        del reference,latent;gc.collect();verify_sources(lock)
        state['process_lifetime_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        assert state['process_lifetime_peak_RSS_bytes']<=LIMITS['RSS_bytes']
        assert disk_bytes(HERE)<=LIMITS['output_bytes'] and time.time()<=deadline
        assert watcher.is_alive() and not errors
        stop.set();watcher.join(timeout=1);assert not watcher.is_alive() and not errors
        state.update(status='completed_exact_streaming_baseline',finished_utc=time.time(),per_frame=frame_rows,
            source_and_input_hashes_unchanged=True,watchdog_completed_cleanly=True,
            comparison_scope='one fixed order, both validation-instrumented; stream additionally compares CPU bytes; not repeated speed measurement',
            RGB_contract='both GPU float/clamp(-1,1)/mul(.5)/add(.5)/clamp(0,1); differs from prior precision probe CPU affine',
            decision='exact_output_baseline_valid_resource_effect_descriptive_not_new_method')
        save(HERE/'RESULT.json',state)
    except BaseException as exc:
        state.update(status='inconclusive_invalid_or_resource_failure',finished_utc=time.time(),exception=repr(exc),traceback=traceback.format_exc())
        save(HERE/'FAILURE.json',state);raise
    finally:stop.set();watcher.join(timeout=1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);run(p.parse_args().lock)
