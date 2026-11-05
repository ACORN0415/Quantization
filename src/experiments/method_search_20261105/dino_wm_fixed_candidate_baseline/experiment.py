"""One bounded fixed-candidate development diagnostic; no CEM/control/quantizer.

Preparation imports only the standard library. Execution needs a new root manifest.
"""
import argparse
import ast
import copy
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import pickle
import resource
import signal
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'dino_wm_model_smoke_maskfix'
ARMS = ('direct_fp32', 'reuse_fp32', 'reuse_bf16', 'reuse_fp16')
SEED = 20261106
BATCH = 4
N = 16
WARMUPS = 2
REPEATS = 8
ATOL = RTOL = 1e-5  # Reuse equivalence only; never a task-quality threshold.


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def save(name, record):
    p = HERE / name
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalization(path):
    """Read only literal constants/default from the pinned dataset AST, not imports."""
    require(sha(path) == '7e630c07f34f83ed06d6ccd558d189679ab7869a4cd817ad0199d151ffe939e7', 'dataset normalization source changed')
    nodes = ast.parse(Path(path).read_text()).body
    expected = {'ACTION_MEAN':[-.0087,.0068], 'ACTION_STD':[.2019,.2002],
        'PROPRIO_MEAN':[236.6155,264.5674,-2.93032027,2.54307914],
        'PROPRIO_STD':[101.1202,87.0112,74.84556075,74.14009094]}
    values = {}
    for n in nodes:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in expected:
            call=n.value
            require(isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name)
                    and call.func.value.id=='torch' and call.func.attr=='tensor' and len(call.args)==1 and not call.keywords, 'nonliteral dataset constant')
            values[n.targets[0].id]=ast.literal_eval(call.args[0])
    require(values==expected,'normalization differs from reviewed constants')
    cls=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name=='PushTDataset')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    defaults=dict(zip([n.arg for n in init.args.args][-len(init.args.defaults):],init.args.defaults))
    values['action_scale']=ast.literal_eval(defaults['action_scale'])
    require(values['action_scale']==100.,'dataset action_scale changed')
    return values


def validate(m):
    require(m.get('status') == 'approved_one_fixed_candidate_development_diagnostic', 'not approved')
    require(m.get('root_execution_authorized') is True and m.get('actual_opus_source_review_completed') is True, 'authority/review missing')
    require(m['code_sha256'] == sha(__file__) and m['spec_sha256'] == sha(HERE/'ROOT_SPEC.json'), 'code/spec changed')
    require(m['limits'] == dict(wall_seconds=900, CPU=4, RSS_bytes=8*1024**3,
                               GPU_allocator_bytes=8*1024**3, output_bytes=128*1024**2), 'limits changed')
    require(m['interpreter'] == sys.executable, 'wrong interpreter')
    require(m['design'] == dict(arms=list(ARMS), seed=SEED, candidates=N, microbatch=BATCH,
                               warmups=WARMUPS, repetitions=REPEATS, episode=0, obs_frame=0,
                               goal_frame=25, horizon=5, atol=ATOL, rtol=RTOL), 'design changed')
    require(sha(BASE/'model_smoke.py') == '828d294dbe62b4a42c845f70f4dacabbac91a239e30b472a7a152c99a507bbc4', 'base changed')
    base = module_at('_reviewed_smoke_reference', BASE/'model_smoke.py')
    sm = json.loads((BASE/'ROOT_APPROVED_MANIFEST.json').read_text())
    hashes = base.validate_manifest(sm)
    for p, h in m['bound_files'].items():
        require(Path(p).is_absolute() and Path(p).resolve() == Path(p) and sha(p) == h, 'bound input changed: '+p)
        require(p not in hashes or hashes[p] == h, 'conflicting hash')
        hashes[p] = h
    for p, h in sm['site_packages_guard_files'].items():
        require(sha(p) == h, 'installed dependency changed: '+p)
        hashes[p] = h
    return base, sm, hashes


def load_model(base, m, record):
    """Same namespace/offline wrapper and fresh strict predictor as successful smoke."""
    base.block_network()
    os.environ['XFORMERS_DISABLED'] = '1'
    qroot, droot = Path(m['quantwm_source_root']), Path(m['dinov2_source_root'])
    before = list(sys.path)
    require(not any(x == 'models' or x.startswith('models.') for x in sys.modules), 'models namespace conflict')
    sys.path[:0] = [str(qroot), str(droot)]
    package = types.ModuleType('models'); package.__package__ = 'models'
    package.__path__ = [str(qroot/'models')]
    package.__spec__ = importlib.machinery.ModuleSpec('models', loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules['models'] = package
    import torch
    import yaml
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(SEED)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.hub.load = torch.hub.load_state_dict_from_url = torch.hub.download_url_to_file = base.block_network()
    vit = importlib.import_module('models.vit')
    importlib.import_module('models.vqvae'); importlib.import_module('models.proprio')
    world = importlib.import_module('models.visual_world_model')
    dino = importlib.import_module('dinov2.models.vision_transformer')
    require(importlib.import_module('dinov2.layers.attention').XFORMERS_AVAILABLE is False, 'unexpected xformers')
    sys.path[:] = [str(qroot), str(droot)] + before
    cfg = yaml.safe_load(Path(m['hydra']['path']).read_text()); c = base.config_contract(cfg)
    payload = torch.load(m['checkpoint']['path'], map_location='cpu', weights_only=False)
    record['loads']['model_checkpoint'] += 1
    require(set(payload) == {'epoch','predictor','predictor_optimizer','decoder','decoder_optimizer','action_encoder','proprio_encoder'}, 'checkpoint keys')
    fresh = vit.ViTPredictor(**c['predictor_kwargs']); fresh.load_state_dict(payload['predictor'].state_dict(), strict=True)
    expected = {'transformer.layers.'+str(i)+'.0.bias' for i in range(6)}
    require(not dict(payload['predictor'].named_buffers()) and set(dict(fresh.named_buffers())) == expected, 'mask layouts')
    oldmods, newmods = dict(payload['predictor'].named_modules()), dict(fresh.named_modules())
    for i in range(6):
        k = 'transformer.layers.'+str(i)+'.0'
        require(torch.equal(oldmods[k].bias, newmods[k].bias), 'mask value changed')
    backbone = dino.vit_small(**base.DINO_KWARGS)
    state = torch.load(m['dino_weight']['path'], map_location='cpu', weights_only=True)
    record['loads']['DINO_weight'] += 1
    backbone.load_state_dict(state, strict=True)
    class OfflineEncoder(torch.nn.Module):
        def __init__(self, net):
            super().__init__(); self.base_model=net; self.name='dinov2_vits14'
            self.feature_key='x_norm_patchtokens'; self.latent_ndim=2
            self.emb_dim=net.num_features; self.patch_size=net.patch_size
        def forward(self, x):
            return self.base_model.forward_features(x)[self.feature_key]
    wm = world.VWorldModel(**c['wm_kwargs'], encoder=OfflineEncoder(backbone),
        predictor=fresh, action_encoder=payload['action_encoder'], proprio_encoder=payload['proprio_encoder'],
        decoder=payload['decoder'], proprio_dim=10, action_dim=10, concat_dim=1,
        num_action_repeat=1, num_proprio_repeat=1)
    wm.eval()
    require(not torch.cuda.is_initialized(), 'CUDA before CPU preparation done')
    record['module_origins'] = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if f:
            f = str(Path(f).resolve())
            if Path(f).is_relative_to(qroot) or Path(f).is_relative_to(droot):
                require(f in m['source_files'] and sha(f) == m['source_files'][f], 'unbound local module: '+name)
                record['module_origins'][name] = f
    return torch, wm


def inputs(torch, m, record):
    """Direct recorded-input adaptation, not PlanWorkspace simulator target replay."""
    import cv2
    require(cv2.__version__ == m['cv2_version'], 'cv2 version changed')
    root = Path(m['dataset_root'])
    norm=normalization(HERE/'source/datasets/pusht_dset.py')
    tensors = {}
    for key in ('states','rel_actions','velocities'):
        tensors[key] = torch.load(root/(key+'.pth'), map_location='cpu', weights_only=True)
        record['loads']['dataset_tensor'] += 1
        require(isinstance(tensors[key], torch.Tensor) and tensors[key].ndim == 3, 'dataset tensor layout')
    # Trusted, independently hashed official data member; explicit whole pickle policy.
    with (root/'seq_lengths.pkl').open('rb') as f: lengths = pickle.load(f)
    record['loads']['seq_length_pickle'] += 1
    require(int(lengths[0]) > 25 and tensors['states'].shape[-1] == 5 and tensors['velocities'].shape[-1] == 2, 'selected development case invalid')
    cap = cv2.VideoCapture(str(root/'obses/episode_000.mp4')); frames=[]
    try:
        require(cap.isOpened(), 'video open')
        for i in range(26):
            ok, frame = cap.read(); require(ok and frame is not None, 'video shorter than selected frame')
            if i in (0,25): frames.append(torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy()))
    finally: cap.release()
    require(len(frames)==2 and frames[0].shape==frames[1].shape, 'decoded RGB layout')
    raw = torch.stack(frames)
    transform = module_at('_fixed_image_transform',HERE/'source/datasets/img_transforms.py').default_transform(224)
    image = transform(raw.permute(0,3,1,2).float()/255.)
    state = tensors['states'][0,[0,25]].float()
    vel = tensors['velocities'][0,[0,25]].float()
    proprio = (torch.cat([state[:,:2],vel],dim=-1)-torch.tensor(norm['PROPRIO_MEAN']))/torch.tensor(norm['PROPRIO_STD'])
    actions = (tensors['rel_actions'][0,:25].float()/norm['action_scale']-torch.tensor(norm['ACTION_MEAN']))/torch.tensor(norm['ACTION_STD'])
    require(list(actions.shape)==[25,2], 'recorded action layout')
    g = torch.Generator(device='cpu').manual_seed(SEED)
    candidates = torch.randn(N,5,10,generator=g,dtype=torch.float32)
    candidates[0].zero_(); candidates[1] = actions.reshape(5,10)
    require(torch.isfinite(image).all() and torch.isfinite(proprio).all() and torch.isfinite(candidates).all(), 'input finite')
    record['input_adaptation']={'video_decoder':'OpenCV '+cv2.__version__,'original_decoder':'decord (not installed; not claimed byte equivalent)',
        'goal':'recorded frame25, not original PlanWorkspace simulator replay', 'episode':0,'frames':[0,25], 'raw_RGB_shape':list(raw.shape),
        'candidate0':'zero normalized action','candidate1':'recorded normalized relative action sequence','candidate2_15':'fixed CPU Gaussian seed20261106, no clipping/search',
        'normalization':norm, 'normalization_source_sha256':sha(HERE/'source/datasets/pusht_dset.py'),
        'transform':'Pinned default_transform224; frameskip5 contiguous bundle'}
    artifact={'raw_RGB':raw, 'visual':image, 'proprio':proprio, 'actions':candidates}
    torch.save(artifact,HERE/'INPUTS.pt')
    obs={'visual':image[0:1].unsqueeze(0),'proprio':proprio[0:1].unsqueeze(0)}
    goal={'visual':image[1:2].unsqueeze(0),'proprio':proprio[1:2].unsqueeze(0)}
    return obs,goal,candidates


def reused_rollout(torch, wm, zobs, act):
    # Exact source recurrence; only action-independent encode_obs was moved out.
    b=act.shape[0]; visual=zobs['visual'].expand(b,-1,-1,-1)
    proprio=zobs['proprio'].expand(b,-1,-1)
    act0=wm.encode_act(act[:,:1])
    z=torch.cat([visual,proprio.unsqueeze(2).expand(-1,-1,visual.shape[2],-1),
                 act0.unsqueeze(2).expand(-1,-1,visual.shape[2],-1)],dim=3)
    for t in range(1,act.shape[1]):
        pred=wm.predict(z[:,-wm.num_hist:]); new=pred[:,-1:]
        new=wm.replace_actions_from_z(new,act[:,t:t+1]);z=torch.cat([z,new],dim=1)
    pred=wm.predict(z[:,-wm.num_hist:]); z=torch.cat([z,pred[:,-1:]],dim=1)
    out,_=wm.separate_emb(z)
    return out,z


def run(manifest_path):
    start=time.time(); phase='manifest'; torch=None
    r={'status':'failed','started_epoch':start,'loads':dict(model_checkpoint=0,DINO_weight=0,dataset_tensor=0,seq_length_pickle=0),
       'scope':'One recorded development case, identical fixed candidates, no CEM/environment/quantization/control/novelty claim.'}
    require(not (HERE/'RUN.json').exists() and not (HERE/'RESULT.json').exists(),'one run only')
    try:
        m=json.loads(Path(manifest_path).read_text());base,sm,hashes=validate(m)
        r.update(code_sha256=sha(__file__),manifest_sha256=sha(manifest_path),bound_files=hashes)
        with (HERE/'RUN.json').open('x') as f:json.dump({'started_epoch':start,'manifest_sha256':r['manifest_sha256']},f)
        def stop(sig, frame):raise KeyboardInterrupt('stop signal '+str(sig))
        signal.signal(signal.SIGTERM,stop)
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
        def guard():
            require(time.time()-start<=900,'wall cap')
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3,'RSS cap')
            require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<=128*1024**2,'output cap')
            if torch is not None and torch.cuda.is_initialized():require(torch.cuda.max_memory_allocated()<=8*1024**3,'GPU cap')
        phase='CPU_restore_and_inputs';t0=time.perf_counter()
        torch,wm=load_model(base,sm,r);obs,goal,act=inputs(torch,m,r);guard()
        r['CPU_preparation_seconds']=time.perf_counter()-t0
        require(not torch.cuda.is_initialized(),'CPU stage initialized CUDA')
        phase='GPU_setup'; gpu_setup_start=time.perf_counter()
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
        torch.cuda.reset_peak_memory_stats();wm.to('cuda',dtype=torch.float32)
        obs={k:v.to('cuda') for k,v in obs.items()};goal={k:v.to('cuda') for k,v in goal.items()};act=act.to('cuda')
        original=wm.predictor
        class PredictorBoundary(torch.nn.Module):
            def __init__(self, net, dtype):super().__init__();self.net=net;self.dtype=dtype
            def forward(self,x):return self.net(x.to(self.dtype)).float()
        predictors={'direct_fp32':original,'reuse_fp32':original,
            'reuse_bf16':PredictorBoundary(copy.deepcopy(original).to(torch.bfloat16),torch.bfloat16),
            'reuse_fp16':PredictorBoundary(copy.deepcopy(original).to(torch.float16),torch.float16)}
        for model in predictors.values():model.eval()
        torch.cuda.synchronize()
        r['GPU_setup_seconds']=time.perf_counter()-gpu_setup_start
        r['predictor_parameter_bytes']={k:sum(t.numel()*t.element_size() for t in v.parameters()) for k,v in predictors.items()}
        counters={'encoder':0,'predictor':0,'decoder':0};dtype_seen={}
        hooks=[]
        def counter(key):
            def hook(mod,args):counters[key]+=1
            return hook
        hooks.append(wm.encoder.base_model.patch_embed.register_forward_pre_hook(counter('encoder')))
        hooks.append(wm.decoder.register_forward_pre_hook(counter('decoder')))
        for name, model in predictors.items():
            if name=='direct_fp32':continue  # direct/reuse FP32 share one Module.
            hooks.append(model.register_forward_pre_hook(counter('predictor')))
            inner=model.net if hasattr(model,'net') else model
            first=next(mod for mod in inner.modules() if isinstance(mod,torch.nn.Linear))
            def inspect(mod,args,name=name):
                if name not in dtype_seen:dtype_seen[name]={'input':str(args[0].dtype),'weight':str(mod.weight.dtype)}
            hooks.append(first.register_forward_pre_hook(inspect))
        objective=module_at('_fixed_objective',HERE/'source/planning/objectives.py').create_objective_fn(alpha=.1,base=1,mode='last')
        r['objective_choice']={'alpha':.1,'base':1,'mode':'last','intent':'Fixed diagnostic design, not a claim to reproduce plan_pusht.yaml defaults. Original objective function only; base unused in last mode.'}
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            t0=time.perf_counter();target=wm.encode_obs(goal);torch.cuda.synchronize()
            r['goal_encoding_seconds_including_sync']=time.perf_counter()-t0
            torch.save({k:v.cpu() for k,v in target.items()},HERE/'GOAL_FEATURES.pt')
            def evaluate(name):
                wm.predictor=predictors[name];pieces=[];zs=[]
                zobs=wm.encode_obs(obs) if name!='direct_fp32' else None
                for j in range(0,N,BATCH):
                    a=act[j:j+BATCH]
                    if name=='direct_fp32':out,z=wm.rollout({k:v.expand(BATCH,*v.shape[1:]) for k,v in obs.items()},a)
                    else:out,z=reused_rollout(torch,wm,zobs,a)
                    require(list(z.shape)==[BATCH,6,196,404],'rollout shape')
                    pieces.append(objective(out,target));zs.append(z[:,-1:].clone())
                return torch.cat(pieces),torch.cat(zs)
            phase='same_candidate_numerics';numeric={};saved={}
            for name in ARMS:
                before=dict(counters);scores,z=evaluate(name);torch.cuda.synchronize();guard()
                require(scores.shape==(N,) and torch.isfinite(scores).all() and torch.isfinite(z).all(),'finite scores/output')
                numeric[name]={'scores':scores.cpu().tolist(),'argmin':scores.argmin().item(),
                    'calls':{k:counters[k]-before[k] for k in counters},'output_dtype':str(z.dtype)}
                require(numeric[name]['calls']==dict(encoder=4 if name=='direct_fp32' else 1,predictor=20,decoder=0),'call contract')
                saved[name]=z.cpu()
            delta=(saved['direct_fp32']-saved['reuse_fp32']).abs()
            equal=torch.allclose(saved['direct_fp32'],saved['reuse_fp32'],atol=ATOL,rtol=RTOL)
            r['reuse_equivalence']={'passed':equal,'atol':ATOL,'rtol':RTOL,'max_abs':delta.max().item(),
                'torch_equal':torch.equal(saved['direct_fp32'],saved['reuse_fp32']),'scope':'all final latent coordinates for16 candidates; not all intermediate tensors or all inputs'}
            r['numerical']=numeric;save('PROGRESS.json',r)
            require(equal,'FP32 reuse equivalence failed; do not relax gate or continue timing')
            for name in ('reuse_bf16','reuse_fp16'):
                expected='torch.bfloat16' if name=='reuse_bf16' else 'torch.float16'
                require(dtype_seen[name]==dict(input=expected,weight=expected),'predictor precision not actual')
            r['dtype_seen']=dtype_seen
            r['precision_contract']='FP32 encoder/action/proprio/recurrence and score; predictor weights/inputs/internal operations BF16 or FP16, returned prediction FP32. Autocast and TF32 off.'
            torch.save(saved,HERE/'FINAL_LATENTS.pt')
            phase='crossed_timing';timings=[]
            for cycle in range(WARMUPS+REPEATS):
                order=ARMS[cycle%4:]+ARMS[:cycle%4]
                for name in order:
                    guard();torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin.record();t0=time.perf_counter();scores,z=evaluate(name);end.record();torch.cuda.synchronize();wall=time.perf_counter()-t0
                    if cycle>=WARMUPS:timings.append({'cycle':cycle-WARMUPS,'arm':name,'wall_seconds':wall,'CUDA_event_ms':begin.elapsed_time(end),
                        'scores':scores.cpu().tolist(),'argmin':scores.argmin().item()})
            r['timings']=timings;r['timing_scope']='Each16-candidate scoring call includes initial observation encoding, microbatch loop, predictor casts/recurrence/objective and checks. Common goal encoding, CPU loading, model transfers/extra predictor copies excluded and reported separately. Hooks/checks instrumented; not a production benchmark.'
        for hook in hooks:hook.remove()
        require(all(sha(f)==h for f,h in hashes.items()),'bound file changed')
        r['GPU']={'name':torch.cuda.get_device_name(0),'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),
            'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cuDNN':torch.backends.cudnn.allow_tf32,'autocast':False,'copies_resident':'FP32,BF16,FP16 predictor copies plus FP32 encoder/action/proprio/decoder'}
        r['actual_total_calls']=counters;guard();r['status']='completed_fixed_candidate_development_diagnostic_not_control'
    except BaseException as exc:r.update(error=repr(exc),failed_phase=phase,traceback=traceback.format_exc())
    finally:
        r.update(finished_epoch=time.time(),wall_seconds=time.time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save('RESULT.json',r)
    print(json.dumps({'status':r['status'],'failed_phase':r.get('failed_phase'),'seconds':r['wall_seconds']}))
    return 0 if r['status'].startswith('completed_') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();raise SystemExit(run(a.manifest))
