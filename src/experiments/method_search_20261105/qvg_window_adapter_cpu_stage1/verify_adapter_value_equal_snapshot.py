"""Independent synthetic verifier: no real PRQ, GPU, model or tensor artifacts."""
import ast
from enum import IntEnum
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import time
from typing import Optional
import torch

HERE = Path(__file__).resolve().parent
UPSTREAM = Path('/home/acorn0415/sfq_data/qvg-reuse-reference')
CACHE = UPSTREAM / 'quant_videogen/kv_cache.py'
ROPE = UPSTREAM / 'experiments/Self-Forcing/wan/modules/causal_model.py'
PIPELINE = UPSTREAM / 'experiments/Self-Forcing/pipeline/causal_inference.py'
REG = json.loads((HERE / 'REGISTRATION.json').read_text())
CHECKS = []
T0 = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard():
    if time.time() > REG['absolute_deadline_utc']:
        raise TimeoutError('Original shared 30 minute deadline')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8 * 1024 ** 3:
        raise MemoryError('8 GiB RSS')
    if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) > 1024 ** 3:
        raise OSError('1 GiB new file budget')
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA unexpectedly initialized')


def check(name, condition, detail=None):
    if not bool(condition):
        raise AssertionError(name + (': ' + str(detail) if detail is not None else ''))
    CHECKS.append({'name': name, 'passed': True})


def extracted(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Source AST symbols missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def fake_decode(a, b):
    return a['decoded'].permute(0, 2, 1, 3), b['decoded'].permute(0, 2, 1, 3)


def fake_pack(x):
    # Whole-span dependence detects missing raw fitting input. Not a PRQ model.
    return {'decoded': (x.float() + x.float().mean() / 64).bfloat16().clone()}


def source_environment():
    ns = {'torch': torch, 'IntEnum': IntEnum, 'Optional': Optional, 'uncompress_kv_cache': fake_decode}
    extracted(CACHE, {'ChunkState', 'ChunkedKVCache'}, ns)
    rn = {'torch': torch, 'math': math}
    extracted(ROPE, {'causal_rope_apply', 'causal_rope_apply_long_input'}, rn)
    return ns['ChunkedKVCache'], ns['ChunkState'], rn


def new_store(cls):
    return cls(1, 2, 2, 12, 126, torch.bfloat16, 'cpu', layout='BSHD')


def values(start, step, kind):
    # Absolute frame identity and overwrite identity remain observable in BF16.
    idx = torch.arange(6 * 2 * 12).reshape(1, 6, 2, 12)
    return ((idx.remainder(13).float() - 6) / 32 + start / 64 + step / 128 + kind / 8).bfloat16()


def snapshot(store):
    return {'states': tuple(int(s) for s in store.chunk_state),
            'chunks': [None if x is None else x.clone() for x in store.chunks],
            'spans': [(s['start_chunk'], s['end_chunk'], s['quant_data']['decoded'].clone()) for s in store.quantized_spans]}


def same_snapshot(a, b):
    if a['states'] != b['states'] or len(a['spans']) != len(b['spans']):
        return False
    for x, y in zip(a['chunks'], b['chunks']):
        if (x is None) != (y is None) or (x is not None and not torch.equal(x, y)):
            return False
    return all(x[:2] == y[:2] and torch.equal(x[2], y[2]) for x, y in zip(a['spans'], b['spans']))


def expect_reject(name, fn):
    try:
        fn()
    except (ValueError, AssertionError, RuntimeError, TypeError) as exc:
        check(name, True)
        return {'type': type(exc).__name__, 'message': str(exc)}
    raise AssertionError(name + ': accepted unsupported operation')


def literal_reference_update(ref_frames, start, step, kind):
    x = values(start, step, kind)
    for f in range(3):
        ref_frames[start + f] = x[:, 2 * f:2 * (f + 1)].clone()
    return x


def literal_reference_pack(ref_frames, start, end):
    raw = torch.cat(ref_frames[start:end], dim=1)
    packed = fake_pack(raw)['decoded']
    for f in range(start, end):
        ref_frames[f] = packed[:, 2 * (f - start):2 * (f - start + 1)].clone()
    return raw


# Adapter API connection and main are written after the independent author
# freezes the public API. All reference routines above are independent.

def load_adapter():
    path = HERE / 'adapter.py'
    spec = importlib.util.spec_from_file_location('independently_checked_adapter', path)
    mod = importlib.util.module_from_spec(spec)
    # Dataclass annotations can resolve through the module table.
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def initialize(mod, cls, window, layers=1):
    obj = mod.FiniteWindowAdapter(window_frames=window)
    caches = obj.initialize(cls, batch_size=1, frame_seq_length=2, num_heads=2,
                            head_dim=12, num_layers=layers, dtype=torch.bfloat16, device='cpu')
    return obj, caches


class AttentionSpy:
    def __init__(self):
        self.last = None
        self.calls = 0

    def __call__(self, q, k, v, *args, **kwargs):
        self.calls += 1
        self.last = (q.clone(), k.clone(), v.clone())
        return (q.float() + k.float().mean(1, keepdim=True) + v.float().mean(1, keepdim=True)).bfloat16()


def frequencies():
    phase = torch.arange(256, dtype=torch.float64)[:, None] * torch.linspace(.031, .087, 6, dtype=torch.float64)[None]
    return torch.polar(torch.ones_like(phase), phase)


def call_attend(obj, cache, start, step, rn, freq, spy, layer=0):
    q, k, v = values(start, step, 2), values(start, step, 0), values(start, step, 1)
    before = (q.clone(), k.clone(), v.clone(), freq.clone())
    out, ret = obj.attend(layer, cache, current_start=start * 2, q=q, k=k, v=v,
                         grid_sizes=torch.tensor([[3, 1, 2]]), freqs=freq,
                         rope_fn=rn['causal_rope_apply'], attention_fn=spy)
    check('input_preservation', all(torch.equal(a, b) for a, b in zip(before, (q, k, v, freq))))
    check('cache_identity', ret is cache)
    return out, (q, k, v)


def fill_first_eight(obj, caches, rn, freq):
    for block in range(8):
        for step in range(5):
            for layer, cache in enumerate(caches):
                call_attend(obj, cache, block * 3, step, rn, freq, AttentionSpy(), layer)


def trace_case(mod, cls, states, rn, window):
    obj, caches = initialize(mod, cls, window)
    cache = caches[0]
    rk, rv = new_store(cls), new_store(cls)
    literal_k, literal_v = [None] * 126, [None] * 126
    freq = frequencies()
    wrong_offsets = 0
    compression_ranges = []
    peak_physical = 0
    span_expirations = []
    for block in range(42):
        guard()
        start, end = block * 3, (block + 1) * 3
        if block and block % 8 == 0:
            a, b = start - 24, start
            rawk = rk.read(a * 2, b * 2)
            rawv = rv.read(a * 2, b * 2)
            check('pending_K_source_preserved', torch.equal(cache['k'].read(a * 2, b * 2), rawk))
            check('pending_V_source_preserved', torch.equal(cache['v'].read(a * 2, b * 2), rawv))
            check('literal_raw_K', torch.equal(rawk, literal_reference_pack(literal_k, a, b)))
            check('literal_raw_V', torch.equal(rawv, literal_reference_pack(literal_v, a, b)))
            calls = []
            def quantizer():
                calls.append((a, b))
                for name in ('k', 'v'):
                    src = cache[name].read(a * 2, b * 2)
                    cache[name].store_quantized(a * 2, b * 2, fake_pack(src))
            obj.quantize(a * 2, b * 2, start * 2, quantizer)
            check('original_quantizer_once', calls == [(a, b)])
            check('frontier_after_success', obj.frontier_tokens == b * 2)
            rk.store_quantized(a * 2, b * 2, fake_pack(rawk))
            rv.store_quantized(a * 2, b * 2, fake_pack(rawv))
            compression_ranges.append([a, b])
        for step in range(5):
            out, (q, k, v) = call_attend(obj, cache, start, step, rn, freq, spy := AttentionSpy())
            literal_reference_update(literal_k, start, step, 0)
            literal_reference_update(literal_v, start, step, 1)
            rk.write(start * 2, end * 2, k)
            rv.write(start * 2, end * 2, v)
            left = max(0, end - window)
            k_all, v_all = rk.read(0, end * 2), rv.read(0, end * 2)
            expected_k, expected_v = k_all[:, left * 2:], v_all[:, left * 2:]
            check('literal_allhistory_K', torch.equal(k_all, torch.cat(literal_k[:end], 1)))
            check('literal_allhistory_V', torch.equal(v_all, torch.cat(literal_v[:end], 1)))
            check('same_window_raw_K', torch.equal(cache['k'].read(left * 2, end * 2), expected_k))
            check('same_window_raw_V', torch.equal(cache['v'].read(left * 2, end * 2), expected_v))
            check('absolute_address_capacity', cache['k'].max_num_chunks == 126 and cache['v'].max_num_chunks == 126)
            expected_q = rn['causal_rope_apply'](q, torch.tensor([[3, 1, 2]]), freq, start_frame=start)
            expected_kr = rn['causal_rope_apply_long_input'](k_all, torch.tensor([[3, 1, 2]]), freq)[:, left * 2:]
            check('absolute_query_RoPE', torch.equal(spy.last[0], expected_q))
            check('absolute_key_RoPE', torch.equal(spy.last[1], expected_kr))
            check('attention_value_window', torch.equal(spy.last[2], expected_v))
            expected_out = AttentionSpy()(expected_q, expected_kr, expected_v)
            check('attention_wrapper_output', torch.equal(out, expected_out))
            # Independent wrong-origin negative control; queries stay absolute.
            if left and step == 4:
                wrong = rn['causal_rope_apply_long_input'](expected_k, torch.tensor([[3, 1, 2]]), freq)
                wrong_offsets += int(not torch.equal(wrong, expected_kr))
            visible = obj.visible_range(0)
            check('reported_visible_range', tuple(visible) == (left * 2, end * 2), visible)
            check('frontier_never_exceeds_completed_blocks', obj.frontier_tokens == (block // 8) * 24 * 2)
            for name in ('k', 'v'):
                s = cache[name]
                check('fully_expired_spans_released', all(t['end_chunk'] > left for t in s.quantized_spans))
                check('pending_frames_are_raw', all(s.chunk_state[f] == states.BF16 for f in range(obj.frontier_tokens // 2, end)))
            retained = sum(x == states.BF16 for x in cache['k'].chunk_state) + sum(t['end_chunk'] - t['start_chunk'] for t in cache['k'].quantized_spans)
            peak_physical = max(peak_physical, retained)
            if window == 21 and block == 7:
                check('W21_pending_0to3_survives', all(cache['k'].chunk_state[f] == states.BF16 for f in range(3)))
            if step == 4 and block in (14, 28):
                span_expirations.append({'block': block, 'spans': [[s['start_chunk'], s['end_chunk']] for s in cache['k'].quantized_spans]})
    check('five_original_group_compressions', compression_ranges == [[0,24],[24,48],[48,72],[72,96],[96,120]])
    check('no_offset_negative_detected', wrong_offsets > 0 if window < 126 else wrong_offsets == 0)
    check('last_six_frames_raw', all(cache['k'].chunk_state[f] == states.BF16 for f in range(120,126)))
    if window == 126:
        check('full_horizon_no_eviction_K', torch.equal(cache['k'].read(0,252), rk.read(0,252)))
        check('full_horizon_no_eviction_V', torch.equal(cache['v'].read(0,252), rv.read(0,252)))
    return {'window_frames': window, 'blocks':42, 'overwrites_per_block':5, 'reads':210,
            'quantization_ranges_frames':compression_ranges,'wrong_zero_origin_detected_calls':wrong_offsets,
            'peak_retained_frame_equivalents_not_bytes':peak_physical,'span_checkpoints':span_expirations}


def negative_cases(mod, cls, states, rn):
    freq = frequencies()
    errors = {}
    for name, kw in (
        ('sink_nonzero', {'window_frames':21,'sink_frames':1}),
        ('window_nonaligned', {'window_frames':22}),
        ('window_zero', {'window_frames':0}),
        ('window_exceeds_horizon', {'window_frames':129}),
        ('wrong_block_size', {'window_frames':21,'block_frames':1}),
        ('wrong_quant_group', {'window_frames':21,'quant_group_frames':21}),
        ('wrong_overwrite_count', {'window_frames':21,'calls_per_block':4}),
    ):
        errors[name] = expect_reject(name, lambda kw=kw: mod.FiniteWindowAdapter(**kw))
    # Concrete original-store failure for early pending eviction.
    naive = new_store(cls)
    for block in range(8):
        naive.write(block*6,(block+1)*6,values(block*3,4,0))
        end=(block+1)*3
        if end>21:naive.evict_range(0,(end-21)*2)
    errors['naive_pending_eviction'] = expect_reject('naive_pending_eviction_breaks_codec_input', lambda: naive.read(0,48))
    # Adapter checks must happen before the upstream destructive partial span write.
    obj,caches=initialize(mod,cls,21);fill_first_eight(obj,caches,rn,freq);cache=caches[0]
    def packboth():
        for n in ('k','v'):cache[n].store_quantized(0,48,fake_pack(cache[n].read(0,48)))
    obj.quantize(0,48,48,packboth)
    before={n:snapshot(cache[n]) for n in ('k','v')};front=obj.frontier_tokens
    errors['partial_packed_overwrite'] = expect_reject('partial_packed_overwrite_rejected',lambda:call_attend(obj,cache,3,0,rn,freq,AttentionSpy()))
    check('partial_overwrite_rejection_preserves_all_state',obj.frontier_tokens==front and all(same_snapshot(before[n],snapshot(cache[n])) for n in before))
    # Original store actually does corrupt the span if called without the guard.
    original = new_store(cls);original.write(0,48,torch.cat([values(i*3,4,0) for i in range(8)],1));original.store_quantized(0,48,fake_pack(original.read(0,48)))
    original.write(6,12,values(3,0,0))
    errors['upstream_partial_write'] = expect_reject('original_partial_write_breaks_other_quantized_frames',lambda:original.read(0,6))
    # Both no-commit and partially committed callbacks must preserve frontier,
    # poison the generation, and forbid continuation; rollback is not promised.
    for partial in (False,True):
        obj,caches=initialize(mod,cls,21,layers=2);fill_first_eight(obj,caches,rn,freq)
        front=obj.frontier_tokens
        def failed():
            if partial:
                cache=caches[0];cache['k'].store_quantized(0,48,fake_pack(cache['k'].read(0,48)))
            raise RuntimeError('synthetic original quantizer failure')
        errors[f'quantizer_failure_{partial}']=expect_reject('quantizer_error_propagates',lambda:obj.quantize(0,48,48,failed))
        check('failed_quantizer_frontier_unchanged',obj.frontier_tokens==front)
        check('failed_quantizer_poisons',obj.poisoned)
        errors[f'poison_readwrite_{partial}']=expect_reject('poison_blocks_readwrite',lambda:call_attend(obj,caches[0],24,0,rn,freq,AttentionSpy()))
        errors[f'poison_quantize_{partial}']=expect_reject('poison_blocks_quantizer_retry',lambda:obj.quantize(0,48,48,lambda:None))
    return errors

def keyboard_interrupt_case(mod, cls, rn):
    obj,caches=initialize(mod,cls,21);freq=frequencies();fill_first_eight(obj,caches,rn,freq)
    old_caches=caches;front=obj.frontier_tokens
    def interrupted():
        cache=caches[0]
        cache['k'].store_quantized(0,48,fake_pack(cache['k'].read(0,48)))
        raise KeyboardInterrupt('synthetic interruption after K commit')
    try:
        obj.quantize(0,48,48,interrupted)
    except KeyboardInterrupt:
        check('KeyboardInterrupt_propagates',True)
    else:
        raise AssertionError('interrupt swallowed')
    check('KeyboardInterrupt_frontier_unchanged',obj.frontier_tokens==front)
    check('KeyboardInterrupt_poisoned',obj.poisoned)
    expect_reject('KeyboardInterrupt_blocks_continuation',lambda:call_attend(obj,caches[0],24,0,rn,freq,AttentionSpy()))
    obj.reset()
    check('reset_clears_old_spans',all(not c[n].quantized_spans for c in old_caches for n in ('k','v')))
    check('reset_clears_old_raw',all(all(x is None for x in c[n].chunks) for c in old_caches for n in ('k','v')))
    fresh=obj.initialize(cls,batch_size=1,frame_seq_length=2,num_heads=2,head_dim=12,num_layers=1,dtype=torch.bfloat16,device='cpu')
    check('reset_fresh_not_poisoned',not obj.poisoned and obj.frontier_tokens==0 and fresh is not old_caches)
    call_attend(obj,fresh[0],0,0,rn,freq,AttentionSpy())
    return {'interruption_after_partial_K_commit':True,'frontier_preserved':True,'resume_without_reset_rejected':True,'fresh_generation_succeeds':True,'rollback_claim':False}


def pipeline_fixture(mod, cls, rn):
    from types import SimpleNamespace
    tree=ast.parse(PIPELINE.read_text())
    init_nodes=[n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_initialize_kv_cache']
    check('original_initializer_AST_found',len(init_nodes)==1)
    ns={'torch':torch,'ChunkedKVCache':cls}
    exec(compile(ast.Module(body=init_nodes,type_ignores=[]),str(PIPELINE),'exec'),ns)
    original_ns=SimpleNamespace(local_attn_size=21,frame_seq_length=2,num_transformer_blocks=1)
    ns['_initialize_kv_cache'](original_ns,1,torch.bfloat16,'cpu',target_frames=126)
    check('original_finite_initializer_ignores_target_frames',original_ns.kv_cache1[0]['k'].max_num_chunks==21)
    class FakeAttention:
        def __init__(self,w):
            self.sink_size=0;self.local_attn_size=w;self.num_heads=2;self.head_dim=12
        def attn_kv_cache_prerope(self,*args,**kwargs):
            raise RuntimeError('Unpatched source attention must not be used in fake pipeline')
    class FakePipeline:
        def __init__(self,w=21):
            self.kv_cache1=None;self.num_transformer_blocks=2;self.num_frame_per_block=3
            self.independent_first_frame=False;self.denoising_step_list=[1,2,3,4]
            self.quant_config=SimpleNamespace(quant_type='prq');self.local_attn_size=w;self.frame_seq_length=2
            self.generator=SimpleNamespace(model=SimpleNamespace(local_attn_size=w,blocks=[SimpleNamespace(self_attn=FakeAttention(w)) for _ in range(2)]))
            self.original_init_calls=0;self.original_quant_calls=[];self.original_inference_calls=0
            self.freq=frequencies();self.last_output=None
        def _initialize_kv_cache(self,batch_size,dtype,device,target_frames=None):
            self.original_init_calls+=1
            return ns['_initialize_kv_cache'](self,batch_size,dtype,device,target_frames)
        def quantize_kv_cache(self,start,end,max_tokens):
            self.original_quant_calls.append((start,end,max_tokens))
            for cache in self.kv_cache1:
                for side in ('k','v'):
                    cache[side].store_quantized(start,end,fake_pack(cache[side].read(start,end)))
        def inference(self,noise,text_prompts,initial_latent=None,return_latents=False,profile=False,low_memory=False):
            self.original_inference_calls+=1
            self._initialize_kv_cache(noise.shape[0],torch.bfloat16,'cpu',target_frames=126)
            outputs=[]
            for block in range(42):
                if block and block%8==0:self.quantize_kv_cache((block-8)*6,block*6,block*6)
                for step in range(5):
                    for layer,blk in enumerate(self.generator.model.blocks):
                        q,k,v=values(block*3,step,2),values(block*3,step,0),values(block*3,step,1)
                        out,cache=blk.self_attn.attn_kv_cache_prerope(self.kv_cache1[layer],block*6,q,k,v,None,torch.tensor([[3,1,2]]),self.freq,None,None)
                        outputs.append(out.clone())
            self.last_output=torch.stack(outputs)
            return self.last_output
    pipe=FakePipeline();old_class_methods={n:getattr(type(pipe),n) for n in ('_initialize_kv_cache','quantize_kv_cache','inference')}
    handle=mod.install_pipeline(pipe,cache_cls=cls,rope_fn=rn['causal_rope_apply'],attention_fn=AttentionSpy(),window_frames=21)
    noise=torch.zeros((1,126,1,1,1),dtype=torch.bfloat16)
    out1=pipe.inference(noise,['fixture']);old_caches=pipe.kv_cache1
    check('installed_initializer_absolute_capacity',all(c[n].max_num_chunks==126 for c in old_caches for n in ('k','v')))
    check('installed_initializer_did_not_delegate_broken_original',pipe.original_init_calls==0)
    check('original_quantizer_still_called_five_times',len(pipe.original_quant_calls)==5)
    check('installer_original_inference_called',pipe.original_inference_calls==1)
    before={n:snapshot(old_caches[0][n]) for n in ('k','v')};before_state=handle.controller.inspect()
    initial_error=expect_reject('initial_latent_rejected_before_clear',lambda:pipe.inference(noise,['fixture'],initial_latent=torch.ones(1)))
    check('unsupported_initial_preserves_existing_generation',pipe.kv_cache1 is old_caches and handle.controller.inspect()==before_state and all(same_snapshot(before[n],snapshot(old_caches[0][n])) for n in before))
    horizon_error=expect_reject('wrong_noise_horizon_rejected',lambda:pipe.inference(noise[:,:123],['fixture']))
    out2=pipe.inference(noise,['fixture'])
    check('second_generation_fresh_cache_identity',pipe.kv_cache1 is not old_caches)
    check('second_generation_clears_prior_spans',all(not c[n].quantized_spans for c in old_caches for n in ('k','v')))
    check('second_generation_clears_prior_chunks',all(all(v is None for v in c[n].chunks) for c in old_caches for n in ('k','v')))
    check('reused_pipeline_output_bit_equal',torch.equal(out1,out2))
    check('original_quantizer_second_five_calls',len(pipe.original_quant_calls)==10)
    handle.uninstall()
    check('uninstall_cache_cleared',pipe.kv_cache1 is None)
    check('uninstall_restores_inherited_methods',all(n not in vars(pipe) and getattr(type(pipe),n) is m for n,m in old_class_methods.items()))
    # Unsupported installation must not partially patch the pipeline.
    bad=FakePipeline();bad.generator.model.blocks[1].self_attn.sink_size=1
    before_keys=set(vars(bad))
    expect_reject('invalid_install_sink_rejected',lambda:mod.install_pipeline(bad,cache_cls=cls,rope_fn=rn['causal_rope_apply'],attention_fn=AttentionSpy(),window_frames=21))
    check('invalid_install_no_partial_method_patches',set(vars(bad))==before_keys and all(n not in vars(bad) for n in old_class_methods))
    return {'fake_pipeline_generation_runs':2,'fake_generation_blocks_per_run':42,'fake_layers':2,'actual_upstream_initializer_AST_tested':True,
            'original_full_inference_or_real_quantizer_executed':False,'initial_latent_error':initial_error,'horizon_error':horizon_error}


def main():
    global T0
    T0=time.monotonic();torch.set_num_threads(4);torch.set_grad_enabled(False);guard()
    paths=[CACHE,ROPE,PIPELINE,HERE/'adapter.py',Path(__file__),HERE/'REGISTRATION.json']
    hashes={str(p):sha(p) for p in paths}
    cls,states,rn=source_environment();mod=load_adapter()
    cases=[trace_case(mod,cls,states,rn,w) for w in (21,63,126)]
    negatives=negative_cases(mod,cls,states,rn)
    interrupted=keyboard_interrupt_case(mod,cls,rn)
    installed=pipeline_fixture(mod,cls,rn)
    guard()
    check('all_bound_sources_unchanged',all(sha(p)==h for p,h in hashes.items()))
    result={'status':'passed','created_utc':time.time(),'wall_seconds':time.monotonic()-T0,
            'shared_absolute_deadline_utc':REG['absolute_deadline_utc'],'checks_total':len(CHECKS),'checks':CHECKS,
            'cases':cases,'negative_controls':negatives,'interrupt_control':interrupted,'installer_fixture':installed,
            'source_sha256':hashes,'CPU_threads':torch.get_num_threads(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'cuda_initialized':torch.cuda.is_initialized(),'actual_PRQ_executions':0,'actual_model_loads':0,'actual_tensor_artifact_loads':0,'GPU_calls':0,
            'scope':'Actual new adapter with original cache and RoPE AST plus independent frame-list and all-history references. Fake codec and fake pipeline integration only; not actual full pipeline/PRQ/GPU/video quality evidence.',
            'independent_helper_scope':'window_adapter_reference read registration and original cache only; no adapter read, torch execution, or edits; absolute offset and pending counterexamples incorporated.'}
    out=HERE/'AUDIT_CODEX.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','checks_total','wall_seconds','peak_RSS_bytes','GPU_calls')},indent=2))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        failure={'status':'failed_or_incomplete','type':type(exc).__name__,'message':str(exc),'created_utc':time.time(),
                 'traceback':traceback.format_exc(),'checks_passed_before_failure':len(CHECKS),
                 'source_sha256':{str(p):sha(p) for p in (HERE/'adapter.py',Path(__file__)) if p.exists()},
                 'GPU_calls':0,'actual_PRQ_executions':0,'actual_model_loads':0}
        stamp=str(time.time_ns())
        with (HERE/('AUDIT_CODEX_FAILURE_'+stamp+'.json')).open('x') as f:json.dump(failure,f,indent=2);f.write('\n')
        raise
