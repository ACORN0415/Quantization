"""Offline, explicitly adapted FP32 connection smoke. Preparation/selftest imports no Torch.

Only root may run with a separately completed approval manifest. No planner/environment,
dataset loading, quantization, optimizer update, retry, hub acquisition, or decoder forward.
"""
import argparse
import gc
import hashlib
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
CPU_EQ_ATOL = 1e-6
CPU_EQ_RTOL = 1e-6
SEED = 0
ACTION_STEPS = 4  # history3 + one action replacement; source rollout returns5 frames.
QFILES = ('models/vit.py', 'models/vqvae.py', 'models/proprio.py',
          'models/visual_world_model.py', 'distributed_fn/__init__.py',
          'distributed_fn/distributed.py', 'distributed_fn/launch.py')
DFILES = ('dinov2/__init__.py', 'dinov2/models/__init__.py',
          'dinov2/models/vision_transformer.py', 'dinov2/layers/__init__.py',
          'dinov2/layers/attention.py', 'dinov2/layers/block.py',
          'dinov2/layers/dino_head.py', 'dinov2/layers/drop_path.py',
          'dinov2/layers/layer_scale.py', 'dinov2/layers/mlp.py',
          'dinov2/layers/patch_embed.py', 'dinov2/layers/swiglu_ffn.py')
DINO_KWARGS = dict(img_size=518, patch_size=14, init_values=1.0, ffn_layer='mlp',
                   block_chunks=0, num_register_tokens=0, interpolate_antialias=False,
                   interpolate_offset=0.1, in_chans=3, channel_adaptive=False)
KNOWN_CKPT_SHA = 'e909f0cec958fc0b49a79f2e85730ae6b5f83dee61a224705f91883eb3bb74c5'
KNOWN_HYDRA_SHA = 'e4d7f21b85c778ab8b83c0bb83bc7625fdfb193a6a0619fef25f4a2afd7f8049'

# Fixed source identities from the preceding reviewed closure; runtime paths are external.
EXPECTED_QSHA = {'models/__init__.py': '9ae80cddb2fc9ea28de4c6f97f9b558e88efb3dcef6853f980e36983d8254009', 'models/visual_world_model.py': '931d44c0d541d95be8236f65247de9b40ac3843fe047c1a44d4a2482d1bf636f', 'utils.py': 'dd9ab7aeac2a70beac4733dccc8dfc3fad4a96803cf7cf1844f79252b1b984d4', 'distributed_fn/__init__.py': '2f4541598b2a2fb41681db38bd8230f8a7fbcc9ea16e7abb97fe7c9bacad30d1', 'distributed_fn/distributed.py': 'bc3263fb5ce314ce484d13e8fb2094da6f640f11d79617c8dd9d73a7ed8341be', 'distributed_fn/launch.py': '79bb9d45c50f1b5d57bde66b6f8ad9744fb4cf5eaee62ee4a972716c5ffc7a6d', 'models/dino.py': '3e77b3990cdb16fd6620f03fda51e12e236a36804ac2e859b530fc6d501ac746', 'models/vit.py': 'ae79e86c1be2fc9ac9c2642882c2ddccf2cea65e85cf29b094672aec5d030387', 'models/vqvae.py': '377005da5c79355da6c7db88eded7a04e432c29ddf4279ea89e5364730ff1461', 'models/proprio.py': 'b0f7c761c45a6a889dc51f6d9cf4539480d1a0bd2bd200ad44c4b9373a573c72'}
EXPECTED_DSHA = {'dinov2/__init__.py': '0d2b87b7e71c7f7279ccc4b4f8b541d26071425284a0ea01a5a435fac4511cfe', 'dinov2/hub/backbones.py': '871fca671b12a9ff02e810654baf509e97ccf461bf8196ce5ddeefff2fd87d3e', 'dinov2/hub/utils.py': '579613e3d7b82c2a387eddc215ad9f076b8740f9618229628395fe5755dbd131', 'dinov2/models/__init__.py': 'b93329ce676f3eb0c9a7790d1b3b2b114a0855ea8dc76e32e262d1a7dc6c55ae', 'dinov2/models/vision_transformer.py': '7799a260f2d7d0fe197331d08502fb8c542f9b7424723650f6a39b64fa2639ea', 'dinov2/layers/__init__.py': '1b55deed39d5ab0b589bef421bbfe06f24a10cd590a0f8403234c9ee4e34109d', 'dinov2/layers/attention.py': '79c7be7a452b3aad96698ec38d5d5150b9f4d8ac084fa93324510dc9f624775d', 'dinov2/layers/block.py': '60c0ac7dfa4474be313fabfa5a23d82faf6f0cecd4e720a88be35de9788cb636', 'dinov2/layers/dino_head.py': '9fdb1fa61c0609dc0876f711f9d0d828c33bf9fd972aab3d9c968d44e16b98d1', 'dinov2/layers/drop_path.py': 'b9f8236e86054b9d9a71275efcad2a9ecaa1f86b529d4b8d6109ddb5e806f67a', 'dinov2/layers/layer_scale.py': 'dadd5aafe178f1bf72a205a02a6645c7e635cacbad585d4a7369c200c6e89135', 'dinov2/layers/mlp.py': '255825c73b60a916dd00eb1e38aacbcdbf316e40d6a005efb46e245b7edb43aa', 'dinov2/layers/patch_embed.py': '40da6add3d811198ea3e17cb99cdd4e5cda59e369efbbe3d18d89308618cf142', 'dinov2/layers/swiglu_ffn.py': 'e46d2948fb97e497cf991ff82ce30cb59b59fdf2e12a19568417113716b4f119'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(name, record):
    path = HERE / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def config_contract(cfg):
    """Source/config-defined PushT case only; no default dimension guesses."""
    expected = dict(img_size=224, frameskip=5, concat_dim=1, action_emb_dim=10,
                    proprio_emb_dim=10, num_action_repeat=1, num_proprio_repeat=1,
                    num_hist=3, num_pred=1, has_predictor=True, has_decoder=True,
                    normalize_action=True, state_based=False)
    for key, value in expected.items():
        require(cfg.get(key) == value, 'unexpected stored configuration: ' + key)
    require(cfg['encoder'] == {'_target_': 'models.dino.DinoV2Encoder',
                              'name': 'dinov2_vits14', 'feature_key': 'x_norm_patchtokens'},
            'encoder config differs')
    pred = dict(cfg['predictor'])
    require(pred.pop('_target_') == 'models.vit.ViTPredictor', 'predictor target')
    require(pred == dict(depth=6, heads=16, mlp_dim=2048, dropout=0.1,
                         emb_dropout=0, pool='mean'), 'predictor config differs')
    wm = dict(cfg['model'])
    require(wm.pop('_target_') == 'models.visual_world_model.VWorldModel', 'model target')
    require(wm == dict(image_size=224, num_hist=3, num_pred=1, train_encoder=False,
                       train_predictor=True, train_decoder=True), 'world model config differs')
    require(cfg['env']['name'] == 'pusht' and cfg['env']['kwargs']['with_velocity'] is True,
            'source-derived proprio4 requires PushT velocity')
    patches = (cfg['img_size'] // 16)**2
    dim = 384 + cfg['action_emb_dim'] * cfg['num_action_repeat'] + cfg['proprio_emb_dim'] * cfg['num_proprio_repeat']
    return {'predictor_kwargs': dict(pred, num_patches=patches, num_frames=cfg['num_hist'], dim=dim),
            'wm_kwargs': wm, 'image_size': cfg['img_size'], 'encoder_image_size': (cfg['img_size'] // 16) * 14,
            'num_hist': cfg['num_hist'], 'num_patches': patches, 'emb_dim': dim,
            'proprio_input_dim': 4, 'action_input_dim': 2 * cfg['frameskip'],
            'predictor_input': [1, cfg['num_hist'] * patches, dim],
            'encoded': [1, cfg['num_hist'], patches, dim],
            'rollout': [1, ACTION_STEPS + 1, patches, dim]}


def validate_manifest(m):
    require(m.get('status') == 'approved_one_offline_model_smoke', 'manifest not approved')
    for key in ('root_execution_authorized', 'actual_opus_source_review_completed',
                'independent_source_and_asset_audit_completed'):
        require(m.get(key) is True, 'approval missing: ' + key)
    require(m['code_sha256'] == sha(__file__), 'smoke source hash mismatch')
    require(m['spec_sha256'] == sha(HERE / 'ROOT_SPEC.json'), 'spec hash mismatch')
    require(m['checkpoint_trust_policy'] == 'verified_official_acquisition_explicit_weights_only_false_cpu',
            'trusted checkpoint policy absent')
    require(m['model_epoch'] == 'latest', 'only existing model_latest.pth')
    require(m['checkpoint']['sha256'] == KNOWN_CKPT_SHA and m['hydra']['sha256'] == KNOWN_HYDRA_SHA,
            'different model/config not registered')
    require(m['dino_weight']['bytes'] == 88283115 and len(m['dino_weight']['sha256'] or '') == 64,
            'DINO weight independent post-acquisition SHA absent')
    require(m['limits'] == {'wall_seconds': 600, 'CPU': 4, 'RSS_bytes': 8*1024**3,
                            'GPU_allocator_bytes': 8*1024**3, 'output_bytes': 128*1024**2},
            'execution limits changed')
    require(m['cpu_equivalence'] == {'atol': CPU_EQ_ATOL, 'rtol': CPU_EQ_RTOL}, 'numerical gate changed')
    require(Path(m['interpreter']) == Path(sys.executable), 'wrong interpreter')
    require(m['review_and_audit_evidence'], 'no independent review/audit evidence')
    hashes = dict(m['source_files'])
    for p, s in m['review_and_audit_evidence'].items():
        require(p not in hashes or hashes[p] == s, 'conflicting evidence hash')
        hashes[p] = s
    for label in ('checkpoint', 'hydra', 'dino_weight'):
        asset = m[label]; p = Path(asset['path'])
        require(p.is_file() and p.stat().st_size == asset['bytes'], label + ' size/path mismatch')
        require(str(p) not in hashes or hashes[str(p)] == asset['sha256'], 'conflicting asset hash')
        hashes[str(p)] = asset['sha256']
    qroot, droot = Path(m['quantwm_source_root']), Path(m['dinov2_source_root'])
    for root, files, expected_sha in ((qroot, QFILES, EXPECTED_QSHA), (droot, DFILES, EXPECTED_DSHA)):
        require(root.is_absolute(), 'source root must be absolute')
        for name in files:
            require(m['source_files'].get(str(root / name)) == expected_sha[name], 'missing/changed pinned source: ' + name)
    for path, expected in hashes.items():
        p = Path(path)
        require(p.is_absolute() and p.is_file() and p.resolve() == p, 'noncanonical source/asset path: ' + path)
        require(isinstance(expected, str) and len(expected) == 64 and sha(p) == expected,
                'bound hash mismatch: ' + path)
    return hashes


def block_network():
    def denied(*args, **kwargs):
        raise RuntimeError('network/hub acquisition prohibited in offline smoke')
    socket.create_connection = denied
    original = socket.socket.connect
    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            denied()
        return original(sock, address)
    socket.socket.connect = connect
    socket.socket.connect_ex = connect
    return denied


def project_module_paths(m, require_all=False):
    records = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith(('models.', 'distributed_fn', 'dinov2')):
            continue
        path = getattr(module, '__file__', None)
        require(path is not None, 'unbound project module: ' + name)
        p = str(Path(path).resolve())
        require(p in m['source_files'] and sha(p) == m['source_files'][p], 'unexpected project source: ' + p)
        records[name] = p
    if require_all:
        for key in ('models.vit', 'models.vqvae', 'models.proprio', 'models.visual_world_model',
                    'distributed_fn', 'dinov2.models.vision_transformer'):
            require(key in records, 'required module not loaded: ' + key)
    return records


def run(manifest_path):
    start = time.time(); phase = 'manifest'; torch = None
    record = {'status': 'failed', 'started_epoch': start, 'stages': {}, 'network_calls': 0,
              'dataset_loads': 0, 'environment_steps': 0, 'planner_calls': 0,
              'quantization_calls': 0, 'tensor_pickle_loads': 0, 'decoder_forward_calls': 0,
              'nonclaims': 'Synthetic model connection only; no task quality, CEM/MPC, timing benchmark or novel method.'}
    require(not (HERE / 'RUN.json').exists() and not (HERE / 'RESULT.json').exists(), 'one run only')
    try:
        m = json.loads(Path(manifest_path).read_text())
        hashes = validate_manifest(m)
        manifest_sha = sha(manifest_path)
        with (HERE / 'RUN.json').open('x') as f:
            json.dump({'attempt': 1, 'started_epoch': start, 'manifest_sha256': manifest_sha}, f)
        record.update(manifest_sha256=manifest_sha, code_sha256=sha(__file__), bound_files=hashes)
        def interrupted(sig, frame):
            raise KeyboardInterrupt('root stop signal ' + str(sig))
        signal.signal(signal.SIGTERM, interrupted)
        def guard():
            require(time.time() - start <= m['limits']['wall_seconds'], 'wall cap')
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= m['limits']['RSS_bytes'], 'RSS cap')
            if torch is not None and torch.cuda.is_initialized():
                require(torch.cuda.max_memory_allocated() <= m['limits']['GPU_allocator_bytes'], 'GPU allocator cap')
            require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) <= m['limits']['output_bytes'], 'output cap')
        sys.dont_write_bytecode = True
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'; os.environ['XFORMERS_DISABLED'] = '1'
        require(os.environ.get('PYTHONNOUSERSITE') == '1', 'root launcher must set PYTHONNOUSERSITE=1')
        if hasattr(os, 'sched_getaffinity'):
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:m['limits']['CPU']])
        denied = block_network()
        for prefix in ('models', 'distributed_fn', 'dinov2'):
            require(not any(x == prefix or x.startswith(prefix + '.') for x in sys.modules), 'conflicting project namespace')
        before_path = list(sys.path)
        for path in reversed(m['overlay_paths']):
            require(Path(path).is_dir() and Path(path).is_absolute(), 'overlay path')
            sys.path.insert(0, path)
        qroot, droot = Path(m['quantwm_source_root']), Path(m['dinov2_source_root'])
        sys.path[:0] = [str(qroot), str(droot)]
        package = types.ModuleType('models'); package.__package__ = 'models'
        package.__path__ = [str(qroot / 'models')]
        package.__spec__ = importlib.machinery.ModuleSpec('models', loader=None, is_package=True)
        package.__spec__.submodule_search_locations = list(package.__path__)
        sys.modules['models'] = package
        phase = 'imports'
        import torch as torch_module
        torch = torch_module
        import yaml
        torch.set_num_threads(m['limits']['CPU']); torch.set_num_interop_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.manual_seed(SEED)
        torch.hub.load = denied; torch.hub.load_state_dict_from_url = denied; torch.hub.download_url_to_file = denied
        require(not torch.cuda.is_initialized(), 'CUDA initialized before CPU stage')
        packages = {}
        for name, expected in m['packages'].items():
            mod = importlib.import_module(name)
            path = Path(mod.__file__).resolve()
            require(path.is_relative_to(Path(expected['root']).resolve()), 'wrong installed package path: ' + name)
            require(str(getattr(mod, '__version__', '')) == expected['version'], 'package version mismatch: ' + name)
            packages[name] = {'file': str(path), 'version': str(getattr(mod, '__version__', ''))}
        for name in ('torch', 'torchvision', 'numpy', 'einops', 'accelerate', 'yaml'):
            require(name in packages, 'package manifest incomplete: ' + name)
        vit = importlib.import_module('models.vit')
        importlib.import_module('models.vqvae'); importlib.import_module('models.proprio')
        world = importlib.import_module('models.visual_world_model')
        dino = importlib.import_module('dinov2.models.vision_transformer')
        require(importlib.import_module('dinov2.layers.attention').XFORMERS_AVAILABLE is False,
                'expected official Torch attention fallback not selected')
        record['stages']['imports'] = {'completed': True, 'packages': packages,
            'module_paths': project_module_paths(m, True), 'models_init_bypassed': True,
            'sys_path_added_by_source': [x for x in sys.path if x not in before_path and x not in (str(qroot), str(droot)) and x not in m['overlay_paths']]}
        write_json('PROGRESS.json', record)
        # vqvae appends '..'; remove that side effect before trusted deserialization.
        sys.path[:] = [str(qroot), str(droot)] + m['overlay_paths'] + before_path
        cfg = yaml.safe_load(Path(m['hydra']['path']).read_text())
        contract = config_contract(cfg); record['contract'] = contract
        guard(); phase = 'cpu_checkpoint_restore'; t0 = time.perf_counter()
        payload = torch.load(m['checkpoint']['path'], map_location='cpu', weights_only=False)
        record['tensor_pickle_loads'] += 1
        required = {'epoch', 'predictor', 'predictor_optimizer', 'decoder', 'decoder_optimizer', 'action_encoder', 'proprio_encoder'}
        require(isinstance(payload, dict) and set(payload) == required, 'unexpected original checkpoint keys')
        require(not torch.cuda.is_initialized(), 'checkpoint restore initialized CUDA')
        record['stages'][phase] = {'completed': True, 'seconds': time.perf_counter()-t0,
            'keys': sorted(payload), 'epoch': int(payload['epoch']), 'map_location': 'cpu',
            'weights_only': False, 'trust_policy': m['checkpoint_trust_policy'],
            'module_paths_after_load': project_module_paths(m, True)}
        write_json('PROGRESS.json', record)
        old = payload['predictor']; decoder = payload['decoder']
        action = payload['action_encoder']; proprio = payload['proprio_encoder']
        for label, module in [('old_predictor', old), ('decoder', decoder), ('action', action), ('proprio', proprio)]:
            require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in module.parameters()), label + ' is not original CPU FP32')
        for label, module, inp in [('action', action, contract['action_input_dim']), ('proprio', proprio, contract['proprio_input_dim'])]:
            require(list(module.patch_embed.weight.shape) == [10, inp, 1] and module.in_chans == inp and module.emb_dim == 10 and module.tubelet_size == 1,
                    label + ' actual input/output shape differs from source evidence')
        require(list(old.pos_embedding.shape) == [1, 588, 404], 'predictor checkpoint positional shape')
        fresh = vit.ViTPredictor(**contract['predictor_kwargs'])
        strict = fresh.load_state_dict(old.state_dict(), strict=True)
        require(not strict.missing_keys and not strict.unexpected_keys, 'predictor strict load')
        old.eval(); fresh.eval()
        old_buffers, new_buffers = dict(old.named_buffers()), dict(fresh.named_buffers())
        expected_masks = {'transformer.layers.' + str(i) + '.0.bias' for i in range(6)}
        require(set(old_buffers) == set(), 'unexpected legacy registered buffers')
        require(set(new_buffers) == expected_masks, 'unexpected fresh registered buffers')
        old_modules, new_modules = dict(old.named_modules()), dict(fresh.named_modules())
        mask_comparisons = []
        for i in range(6):
            key = 'transformer.layers.' + str(i) + '.0'
            om, nm = old_modules[key], new_modules[key]
            require('bias' in om.__dict__ and 'bias' not in om._buffers,
                    'legacy mask is not the observed plain tensor attribute')
            require('bias' in nm._buffers and 'bias' in nm._non_persistent_buffers_set,
                    'fresh mask is not a nonpersistent registered buffer')
            ob, nb = om.bias, nm.bias
            for value in (ob, nb):
                require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                        and value.device.type == 'cpu' and list(value.shape) == [1,1,588,588],
                        'mask shape/dtype/device differs')
            require(torch.equal(ob, nb), 'forward mask values changed: ' + key)
            mask_comparisons.append({'module': key, 'shape': list(ob.shape),
                                     'dtype': str(ob.dtype), 'torch_equal': True,
                                     'old_layout': 'plain_tensor_attribute',
                                     'new_layout': 'nonpersistent_registered_buffer'})
        phase = 'cpu_predictor_equivalence'; guard(); t0 = time.perf_counter()
        x = torch.linspace(-0.5, 0.5, 588*404, dtype=torch.float32).reshape(1, 588, 404)
        with torch.inference_mode():
            y_old = old(x); y_new = fresh(x)
        require(list(y_old.shape) == list(y_new.shape) == [1,588,404], 'predictor output shape')
        require(torch.isfinite(y_old).all().item() and torch.isfinite(y_new).all().item(), 'predictor finite')
        difference = (y_old-y_new).abs()
        equivalent = torch.allclose(y_old, y_new, atol=CPU_EQ_ATOL, rtol=CPU_EQ_RTOL)
        record['stages'][phase] = {'completed': equivalent, 'seconds': time.perf_counter()-t0,
            'shape': list(y_new.shape), 'atol': CPU_EQ_ATOL, 'rtol': CPU_EQ_RTOL,
            'max_abs_difference': difference.max().item(), 'torch_equal': torch.equal(y_old,y_new),
            'strict_state_dict': True, 'forward_mask_tensor_comparisons': mask_comparisons, 'all_forward_masks_exact_equal': True, 'actual_device': str(y_new.device), 'dtype': str(y_new.dtype)}
        require(equivalent, 'CPU old/fresh predictor equivalence failed; no tolerance adjustment')
        write_json('PROGRESS.json', record)
        require(not torch.cuda.is_initialized(), 'CPU forward initialized CUDA')
        del old, payload, x, y_old, y_new, difference, old_buffers, new_buffers; gc.collect()
        phase = 'cpu_dino_local_restore'; guard(); t0 = time.perf_counter()
        backbone = dino.vit_small(**DINO_KWARGS)
        state = torch.load(m['dino_weight']['path'], map_location='cpu', weights_only=True)
        record['tensor_pickle_loads'] += 1
        require(isinstance(state, dict) and all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()), 'DINO artifact must be plain tensor state_dict')
        strict = backbone.load_state_dict(state, strict=True)
        require(not strict.missing_keys and not strict.unexpected_keys, 'DINO strict state_dict')
        del state; gc.collect()
        class OfflineEncoder(torch.nn.Module):
            def __init__(self, base):
                super().__init__(); self.base_model=base; self.name='dinov2_vits14'
                self.feature_key='x_norm_patchtokens'; self.latent_ndim=2
                self.emb_dim=base.num_features; self.patch_size=base.patch_size
            def forward(self, value):
                return self.base_model.forward_features(value)[self.feature_key]
        encoder = OfflineEncoder(backbone)
        wm = world.VWorldModel(**contract['wm_kwargs'], encoder=encoder, proprio_encoder=proprio,
            action_encoder=action, predictor=fresh, decoder=decoder, proprio_dim=cfg['proprio_emb_dim'],
            action_dim=cfg['action_emb_dim'], concat_dim=cfg['concat_dim'],
            num_action_repeat=cfg['num_action_repeat'], num_proprio_repeat=cfg['num_proprio_repeat'])
        wm.eval()  # Original override returns None: never assign/chains this call.
        require(wm.encoder_image_size == 196 and encoder.emb_dim == 384 and encoder.patch_size == 14, 'WM/DINO exact resize contract')
        require(not torch.cuda.is_initialized(), 'CUDA initialized before GPU stage')
        record['stages'][phase] = {'completed': True, 'seconds':time.perf_counter()-t0,
            'constructor':DINO_KWARGS, 'strict_state_dict':True, 'weights_only':True,
            'official_weight_local_sha256':m['dino_weight']['sha256'], 'offline_wrapper_adaptation':True}
        write_json('PROGRESS.json', record)
        phase = 'gpu_connection'; guard()
        require(torch.cuda.is_available(), 'GPU unavailable; CPU stages retained')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        torch.cuda.set_per_process_memory_fraction(m['limits']['GPU_allocator_bytes']/torch.cuda.get_device_properties(0).total_memory, 0)
        torch.cuda.reset_peak_memory_stats(); wm.to(device='cuda',dtype=torch.float32)
        require(all(p.dtype == torch.float32 and p.device.type == 'cuda' for p in wm.parameters()), 'GPU model not FP32')
        g = torch.Generator(device='cpu').manual_seed(SEED)
        obs_cpu = {'visual':torch.rand((1,3,3,224,224),generator=g,dtype=torch.float32)*2-1,
                   'proprio':torch.rand((1,3,4),generator=g,dtype=torch.float32)*2-1}
        act_cpu = torch.rand((1,ACTION_STEPS,10),generator=g,dtype=torch.float32)*2-1
        obs = {k:v.to('cuda') for k,v in obs_cpu.items()}; act=act_cpu.to('cuda')
        # VWorldModel calls encoder.forward directly, bypassing encoder Module hooks.
        # PatchEmbed is invoked via Module.__call__ inside official forward_features.
        seen={'patch_embed_inputs':[], 'patch_embed_outputs':[], 'predictor_calls':0, 'decoder_calls':0}
        def enc_pre(module,args):seen['patch_embed_inputs'].append(list(args[0].shape))
        def enc_post(module,args,value):seen['patch_embed_outputs'].append(list(value.shape))
        def pred_pre(module,args):seen['predictor_calls']+=1
        def dec_pre(module,args):seen['decoder_calls']+=1
        hooks=[backbone.patch_embed.register_forward_pre_hook(enc_pre), backbone.patch_embed.register_forward_hook(enc_post),
               fresh.register_forward_pre_hook(pred_pre), decoder.register_forward_pre_hook(dec_pre)]
        t0=time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type='cuda',enabled=False):
            encoded=wm.encode(obs,act[:,:3]); guard()
            predicted=wm.predict(encoded); guard()
            rollout_obs,rolled=wm.rollout(obs,act); guard()
        torch.cuda.synchronize()
        outputs={'encoded':encoded,'predicted':predicted,'rolled':rolled,
                 'rollout_visual':rollout_obs['visual'],'rollout_proprio':rollout_obs['proprio']}
        shapes={'encoded':[1,3,196,404],'predicted':[1,3,196,404], 'rolled':[1,5,196,404],
                'rollout_visual':[1,5,196,384],'rollout_proprio':[1,5,10]}
        stats={}
        for name,value in outputs.items():
            require(list(value.shape)==shapes[name] and value.dtype==torch.float32 and torch.isfinite(value).all().item(), 'GPU output contract: '+name)
            stats[name]={'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device),'finite':True,'abs_max':value.abs().max().item()}
        require(seen['patch_embed_inputs']==[[3,3,196,196]]*2 and seen['patch_embed_outputs']==[[3,196,384]]*2, 'actual encoder resize/token path')
        require(seen['predictor_calls']==3 and seen['decoder_calls']==0, 'actual model call counts')
        for hook in hooks:hook.remove()
        record['stages'][phase]={'completed':True,'seconds_including_checks':time.perf_counter()-t0,
            'outputs':stats,'actual_calls':seen,'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,
            'TF32_cudnn':torch.backends.cudnn.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'autocast_enabled':False,'XFORMERS_DISABLED':os.environ['XFORMERS_DISABLED'],
            'DINO_attention_path':'official Tensor-input Torch SDPA fallback; low-level CUDA kernel not profiled',
            'gpu_name':torch.cuda.get_device_name(0),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        require(sha(manifest_path)==manifest_sha, 'manifest changed')
        for path,expected in hashes.items():require(sha(path)==expected, 'bound input/source changed')
        record['final_project_modules']=project_module_paths(m,True); guard()
        record['status']='completed_synthetic_model_connection_not_planning'
    except BaseException as exc:
        record.update(error=repr(exc), failed_phase=phase, traceback=traceback.format_exc())
    finally:
        record.update(finished_epoch=time.time(),wall_seconds=time.time()-start,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write_json('RESULT.json',record)
    print(json.dumps({'status':record['status'],'failed_phase':record.get('failed_phase'),'wall_seconds':record['wall_seconds']}))
    return 0 if record['status'].startswith('completed_') else 1


def selftest():
    """Preparation check: Python scalars/config only, no tensor/package/model load."""
    import copy
    cfg=dict(img_size=224,frameskip=5,concat_dim=1,action_emb_dim=10,proprio_emb_dim=10,
             num_action_repeat=1,num_proprio_repeat=1,num_hist=3,num_pred=1,
             has_predictor=True,has_decoder=True,normalize_action=True,state_based=False,
             encoder={'_target_':'models.dino.DinoV2Encoder','name':'dinov2_vits14','feature_key':'x_norm_patchtokens'},
             predictor={'_target_':'models.vit.ViTPredictor','depth':6,'heads':16,'mlp_dim':2048,'dropout':.1,'emb_dropout':0,'pool':'mean'},
             model={'_target_':'models.visual_world_model.VWorldModel','image_size':224,'num_hist':3,'num_pred':1,'train_encoder':False,'train_predictor':True,'train_decoder':True},
             env={'name':'pusht','kwargs':{'with_velocity':True}})
    original=copy.deepcopy(cfg); c=config_contract(cfg)
    require(c['predictor_input']==[1,588,404] and c['encoded']==[1,3,196,404] and c['rollout']==[1,5,196,404] and c['encoder_image_size']==196,'shape derivation')
    require(cfg==original,'config mutation'); checks=2
    for key,value in [('img_size',225),('frameskip',1),('num_hist',2),('concat_dim',0),('has_decoder',False)]:
        bad=copy.deepcopy(cfg);bad[key]=value
        try:config_contract(bad)
        except RuntimeError:checks+=1
        else:raise AssertionError(key)
    try:validate_manifest({'status':'UNAPPROVED'})
    except RuntimeError:checks+=1
    else:raise AssertionError('unapproved manifest')
    require(not any(k in sys.modules for k in ('torch','numpy','pickle','yaml')),'preparation imported forbidden payload/runtime module')
    checks+=1
    result={'status':'passed_pure_scalar_preparation_only','checks':checks,'source_sha256':sha(__file__),
            'contract':c,'GPU':0,'model_tensor_pickle_loads':0,'network':0,'runtime_imports':0}
    with (HERE/'SELFTEST_PREPARATION.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':result['status'],'checks':checks}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--selftest',action='store_true');group.add_argument('--manifest')
    args=parser.parse_args()
    if args.selftest:selftest()
    else:raise SystemExit(run(args.manifest))
