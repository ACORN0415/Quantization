"""Prepared, singly supervised real-codec contract probe. No model imports.

Only `selftest` is authorized during preparation. `run` requires a root-written
execution registration and starts one worker under the total 600 second cap.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = Path('/home/acorn0415/sfq_data/qvg-reuse-reference')
DEPS = Path('/home/acorn0415/sfq_data/qvg-reuse-deps')
ADAPTER = HERE.parent / 'qvg_window_adapter_cpu_stage1/adapter.py'
ROPE = SOURCE / 'experiments/Self-Forcing/wan/modules/causal_model.py'
ATTENTION = SOURCE / 'experiments/Self-Forcing/wan/modules/attention.py'
MODEL_DEFINITION = SOURCE / 'experiments/Self-Forcing/wan/modules/model.py'
REVISION = '0601468f2dbba6a17ac7086faec6d41527cad188'
ADAPTER_SHA = 'cce683d4a2ec1db16dfe986d0202970d1ce3f83458427597f3145e55e8a46d09'
SEED = 20261106
FS, HEADS, DIM = 1560, 12, 128
CAP = 8 * 2**30
CHECKPOINTS = {21: {7:'precompression',8:'postcompression',9:'partial_span',41:'final'},
               63: {7:'precompression',8:'postcompression',21:'partial_span',41:'final'},
               126:{7:'precompression',8:'postcompression',21:'intermediate_full_horizon_not_partial',41:'final'}}


class ContractFailure(RuntimeError):
    pass


class Inconclusive(RuntimeError):
    pass


def require(x, message):
    if not x:
        raise ContractFailure(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_new(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def byte_view(x):
    import torch
    flat = x.detach().contiguous().reshape(-1)
    if flat.numel() and flat.stride(0) != 1:
        fixed = torch.empty(flat.numel(), dtype=flat.dtype, device=flat.device)
        fixed.copy_(flat)
        flat = fixed
    return flat.view(torch.uint8)


def exact(a, b):
    import torch
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(byte_view(a), byte_view(b))


def primitive(obj):
    """Preserve info/config metadata while making a model-free CPU artifact."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {str(k): primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [primitive(v) for v in obj]
    if isinstance(obj, torch.dtype):
        return str(obj)
    if hasattr(obj, '__dict__'):
        return primitive(vars(obj))
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(f'Unrecognized packet metadata {type(obj)}')


def content_digest(obj):
    import torch
    h = hashlib.sha256()
    def walk(x):
        if isinstance(x, torch.Tensor):
            h.update(b'tensor:')
            h.update(json.dumps([str(x.dtype), list(x.shape)], separators=(',', ':')).encode())
            h.update(byte_view(x).cpu().numpy().tobytes())
        elif isinstance(x, dict):
            h.update(b'dict:')
            for key in sorted(x):
                h.update(json.dumps(str(key)).encode());walk(x[key])
            h.update(b'enddict')
        elif isinstance(x, (list, tuple)):
            h.update(b'list:');h.update(str(len(x)).encode())
            for value in x:walk(value)
            h.update(b'endlist')
        elif hasattr(x, '__dict__'):
            walk(vars(x))
        elif isinstance(x, torch.dtype):
            walk(str(x))
        else:
            h.update(json.dumps(x, sort_keys=True, allow_nan=False).encode())
    walk(obj)
    return h.hexdigest()


def tensor_manifest(obj, prefix=''):
    import torch
    rows = []
    if isinstance(obj, torch.Tensor):
        rows.append({'name':prefix,'dtype':str(obj.dtype),'shape':list(obj.shape),
                     'logical_bytes':obj.numel()*obj.element_size(),
                     'owned_storage_bytes':obj.untyped_storage().nbytes()})
    elif isinstance(obj, dict):
        for k,v in obj.items():rows += tensor_manifest(v,prefix+'/'+str(k))
    elif isinstance(obj,(list,tuple)):
        for k,v in enumerate(obj):rows += tensor_manifest(v,prefix+'/'+str(k))
    return rows


def independent_unpack_cpu(state, shape):
    """Existing kv_reuse_preflight INT2/one-stage algebra, explicit CPU tensors."""
    import torch
    require(set(state) == {'centroids_list','cluster_ids_list','residual_quant','scales','zeros','info'},
            'Unexpected packet keys or clipping state')
    cfg = state['info']['quant_config']
    cfg = cfg if isinstance(cfg,dict) else vars(cfg)
    require(cfg['quant_type']=='triton-nstages-kmeans-int2' and cfg['quant_block_size']==64
            and cfg['num_prq_stages']==1 and not cfg['asymmetric'], 'Unregistered unpack format')
    require(state['zeros'] is None and len(state['centroids_list'])==1
            and len(state['cluster_ids_list'])==1, 'Unexpected asymmetric/stage state')
    require(state['residual_quant'].dtype == torch.uint8
            and tuple(state['residual_quant'].shape)==(*shape[:-1],shape[-1]//4), 'Packed INT2 shape/dtype')
    require(state['scales'].dtype==torch.float8_e4m3fn
            and tuple(state['scales'].shape)==(*shape[:-1],shape[-1]//64), 'Scale must be native FP8 D64')
    packed=state['residual_quant'].cpu().to(torch.int32)
    codes=torch.stack([(packed>>shift)&3 for shift in (6,4,2,0)],-1).reshape(shape)-1
    del packed
    scale=state['scales'].cpu().float().repeat_interleave(64,-1)
    centroids=state['centroids_list'][0].cpu().float()
    indices=state['cluster_ids_list'][0].cpu().long()
    gathered=torch.gather(centroids,2,indices[...,None].expand(*shape))
    result=(codes.float()*scale+gathered).bfloat16()
    require(bool(torch.isfinite(result).all()), 'Nonfinite independent decode')
    return result


def load_file(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def rope_functions(torch):
    tree=ast.parse(ROPE.read_text())
    names={'causal_rope_apply','causal_rope_apply_long_input'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==names,'Original RoPE AST missing')
    ns={'torch':torch,'math':math}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROPE),'exec'),ns)
    mt=ast.parse(MODEL_DEFINITION.read_text())
    params=[n for n in mt.body if isinstance(n,ast.FunctionDef) and n.name=='rope_params']
    require(len(params)==1,'Original frequency constructor missing')
    exec(compile(ast.Module(body=params,type_ignores=[]),str(MODEL_DEFINITION),'exec'),ns)
    return ns


def verify_lock(path):
    lock=json.loads(Path(path).read_text())
    require(lock.get('status')=='approved_for_one_GPU_probe','Root execution approval missing')
    required=[Path(__file__).resolve(),ADAPTER,ROPE,ATTENTION,MODEL_DEFINITION,
              SOURCE/'quant_videogen/kv_cache.py',HERE/'PREPARATION.json']
    required+=sorted((SOURCE/'quant_videogen').rglob('*.py'))
    for p in required:require(str(p) in lock['sha256'],f'Unbound source {p}')
    for p,h in lock['sha256'].items():require(sha(p)==h,f'Lock mismatch {p}')
    require(sha(ADAPTER)==ADAPTER_SHA,'Frozen CPU adapter changed')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==REVISION,'Upstream revision mismatch')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip(),'Upstream dirty')
    return lock


def idle_gpu():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if text:
        raise Inconclusive(f'GPU occupied before probe: {text}')


def worker(lock_path, deadline):
    started=time.time()
    result={'status':'running','started_utc':started,'absolute_deadline_utc':deadline,
            'scope':'synthetic real PRQ/cache/RoPE/FA2 compatibility, no model or video',
            'seed':SEED,'source_revision':REVISION,'reads':[],'compressions':[],
            'attention_records':[],'cache_read_records':[], 'model_loads':0,'video_generations':0,
            'counters':{'compress':0,'prq_tensor_encode':0,'decode_total':0,'uncompress_total':0,
                        'rope_apply_total':0,'rope_long_reference':0,'fa2_kernel':0,'sdpa_fallback':0}}
    torch=None
    try:
        lock=verify_lock(lock_path);idle_gpu()
        claim=json.loads((HERE/'ATTEMPT_CLAIM.json').read_text())
        require(deadline==claim['absolute_deadline_utc'],'Worker deadline is not supervisor deadline')
        os.environ['TIME_BENCH']='0'
        sys.path[:0]=[str(DEPS),str(SOURCE)]
        import torch as _torch
        torch=_torch
        torch.set_num_threads(4);torch.set_grad_enabled(False)
        torch.cuda.set_per_process_memory_fraction(CAP/torch.cuda.get_device_properties(0).total_memory)
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        import flash_attn
        import triton
        cm=importlib.import_module('quant_videogen.compress')
        uc=importlib.import_module('quant_videogen.uncompress')
        kc=importlib.import_module('quant_videogen.kv_cache')
        from quant_videogen import QuantizeConfig
        native=load_file('probe_native_attention',ATTENTION)
        require(native.FLASH_ATTN_2_AVAILABLE,'Native FA2 unavailable; no fallback allowed')
        adapter=load_file('probe_frozen_adapter',ADAPTER)
        ropes=rope_functions(torch)
        originals={'compress':cm.compress_kv_cache,'prq':cm.triton_prq_quantize_tensor,
                   'decode':uc.triton_prq_dequantize_tensor,'uncompress':uc.uncompress_kv_cache,
                   'read':kc.ChunkedKVCache.read,'fa2':flash_attn.flash_attn_varlen_func,
                   'rope':ropes['causal_rope_apply'],'long_rope':ropes['causal_rope_apply_long_input']}
        count=result['counters'];phase={'name':'setup'};expected={};store_labels={}
        def guard():
            if time.time()>=deadline:raise Inconclusive('Total 600 second deadline')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>CAP:raise Inconclusive('RSS cap')
            if torch.cuda.max_memory_allocated()>CAP:raise Inconclusive('GPU allocator cap')
            if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())>2**30:raise Inconclusive('Disk cap')
        def observed_prq(*a,**kw):
            count['prq_tensor_encode']+=1;return originals['prq'](*a,**kw)
        def observed_decode(*a,**kw):
            count['decode_total']+=1;return originals['decode'](*a,**kw)
        def observed_uncompress(*a,**kw):
            count['uncompress_total']+=1;return originals['uncompress'](*a,**kw)
        def observed_fa2(*a,**kw):
            count['fa2_kernel']+=1;return originals['fa2'](*a,**kw)
        def forbidden_sdpa(*a,**kw):
            count['sdpa_fallback']+=1
            raise ContractFailure('Unregistered SDPA fallback')
        def observed_rope(*a,**kw):
            count['rope_apply_total']+=1;return originals['rope'](*a,**kw)
        def observed_read(store,start,end):
            label=store_labels[id(store)]
            spans=sum(s['start_chunk']<end//FS and s['end_chunk']>start//FS for s in store.quantized_spans)
            out=originals['read'](store,start,end)
            result['cache_read_records'].append({'phase':phase['name'],'store':label,'start':start,'end':end,'packed_spans':spans})
            if phase['name']=='adapter_attention':
                side=label.rsplit('_',1)[1]
                require(exact(out,expected['raw_'+side]),f'Raw {label} mismatch')
                expected['raw_'+side+'_equal']=True
            return out
        cm.triton_prq_quantize_tensor=observed_prq
        uc.triton_prq_dequantize_tensor=observed_decode
        uc.uncompress_kv_cache=observed_uncompress;kc.uncompress_kv_cache=observed_uncompress
        kc.ChunkedKVCache.read=observed_read
        flash_attn.flash_attn_varlen_func=observed_fa2
        torch.nn.functional.scaled_dot_product_attention=forbidden_sdpa
        cfg=QuantizeConfig(quant_type='triton-nstages-kmeans-int2',quant_block_size=64,
                           num_prq_stages=1,kmeans_max_iters=2,cache_num_k_centroids=256,
                           cache_num_v_centroids=256,asymmetric=False)
        result['codec_config']=primitive(cfg)
        result['effective_config']={'scale_precision':'torch.float8_e4m3fn',
                                    'source':'functions.triton_prq_quantize_tensor default; not a QuantizeConfig field',
                                    'use_percentile_clipping':False,'num_bits':2}
        result['format']={'residual':'unsigned packed shifts6,4,2,0; code minus1',
                          'scale_dtype':'torch.float8_e4m3fn','scale_axis':'per B/H/token and D64',
                          'clipping':False,'zeros':None,'stages':1}
        result['environment']={'torch':torch.__version__,'triton':triton.__version__,
                               'flash_attn':flash_attn.__version__,'cuda':torch.version.cuda,
                               'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability()}
        bindings=[Path(__file__),ADAPTER,ROPE,ATTENTION,MODEL_DEFINITION]+sorted((SOURCE/'quant_videogen').rglob('*.py'))
        bindings += [Path(m.__file__) for m in list(sys.modules.values())
                     if getattr(m,'__file__',None) and ('flash_attn' in str(m.__file__))
                     and Path(m.__file__).is_file()]
        result['source_sha256']={str(p.resolve()):sha(p) for p in set(bindings)}
        require(all(lock['sha256'].get(p)==h for p,h in result['source_sha256'].items()),
                'Loaded native Python/extension source was not bound by execution lock')
        result['binding']={'cache_class_module':kc.ChunkedKVCache.__module__,
                           'compress_module':originals['compress'].__module__,
                           'decoder_module':originals['decode'].__module__,
                           'fa2_function_module':originals['fa2'].__module__,
                           'rope':'unmodified function AST; no model-module import',
                           'attention':'native attention wrapper fa_version=2; actual varlen entry counted',
                           'no_surrogate_attention':'Non-checkpoint callback validates QKV and returns None; no attention value is computed there',
                           'packet_sharing':'same packed dict object on four stores, no offload/onload calls'}
        controls={};caches={}
        for w in (21,63,126):
            controls[w]=adapter.FiniteWindowAdapter(w)
            caches[w]=controls[w].initialize(kc.ChunkedKVCache,batch_size=1,frame_seq_length=FS,
                                             num_heads=HEADS,head_dim=DIM,num_layers=1,
                                             dtype=torch.bfloat16,device=torch.device('cuda'))[0]
        ref={s:kc.ChunkedKVCache(1,FS,HEADS,DIM,126,torch.bfloat16,torch.device('cuda'),layout='BSHD') for s in ('k','v')}
        for s in ('k','v'):store_labels[id(ref[s])]='reference_'+s
        for w in caches:
            for s in ('k','v'):store_labels[id(caches[w][s])]=str(w)+'_'+s
        generator=torch.Generator(device='cuda').manual_seed(SEED)
        grid=torch.tensor([[3,30,52]],dtype=torch.long,device='cpu')
        # Exactly the original model's constructor, without importing/creating the model.
        freqs=torch.cat([ropes['rope_params'](1024,44),ropes['rope_params'](1024,42),
                         ropes['rope_params'](1024,42)],dim=1).to('cuda')
        result['rope_input']={'grid_sizes':[[3,30,52]],'freqs_shape':list(freqs.shape),
                              'freqs_dtype':str(freqs.dtype),'frequency_recipe':'original rope_params(1024,44/42/42), concat dim1; theta10000',
                              'model_frequency_values':True,'absolute_frame_range':[0,126]}
        packets=[]
        def native_fa(q,k,v):
            return native.attention(q,k,v,fa_version=2,dropout_p=0.0,causal=False)
        for block in range(42):
            guard();start=3*block;end=start+3
            if block and block%8==0:
                phase['name']='quantization_source'
                a,b=start-24,start
                source_pair=tuple(ref[s].read(a*FS,b*FS) for s in ('k','v'))
                source_matches=[]
                for w in (21,63,126):
                    require(all(exact(caches[w][s].read(a*FS,b*FS),x) for s,x in zip(('k','v'),source_pair)),f'Pending source mismatch W{w}')
                    source_matches.append(w)
                src_hash=[content_digest(x) for x in source_pair]
                codec_args=tuple(x.permute(0,2,1,3).contiguous() for x in source_pair)
                args_hash=[content_digest(x) for x in codec_args]
                count['compress']+=1
                pair=originals['compress'](*codec_args,cfg.quant_type,cfg,cm.get_quantize_fn(cfg.quant_type,cfg))
                require(args_hash==[content_digest(x) for x in codec_args],'Encoder mutated actual contiguous arguments')
                del codec_args
                require(src_hash==[content_digest(x) for x in source_pair],'Encoder mutated source')
                del source_pair
                for state in pair:state['info']={'quant_config':cfg,'output_dtype':torch.bfloat16}
                before=content_digest(pair)
                # Keep the exact one-stage preflight algebra, enlarged to 24frames.
                phase['name']='independent_decoder'
                decoded=observed_uncompress(*pair)
                dec_equal={}
                for s,state,out in zip(('k','v'),pair,decoded):
                    cpu_expected=independent_unpack_cpu(state,(1,HEADS,24*FS,DIM))
                    dec_equal[s]=exact(out.cpu(),cpu_expected)
                    require(dec_equal[s],f'Independent native INT2 decode mismatch {s}')
                    del cpu_expected
                del decoded,out
                artifact=HERE/f'packet_{len(packets):02d}.pt'
                cpu_packet=primitive(pair)
                require(content_digest(cpu_packet)==before,'CPU artifact metadata/tensor mismatch')
                torch.save(cpu_packet,artifact);del cpu_packet
                record={'start_frame':a,'end_frame':b,'source_matches_windows':source_matches,
                        'source_sha256':src_hash,'call_serial':count['compress'],
                        'actual_codec_argument_sha256':args_hash,'codec_arguments_unchanged':True,
                        'independent_decode_equal':dec_equal,'packet_before_sha256':before,
                        'packet_after_sha256':None,'tensor_manifest':tensor_manifest(pair),
                        'artifact':{'path':str(artifact),'sha256':sha(artifact),'size':artifact.stat().st_size}}
                for w in (21,63,126):
                    def store_callback(w=w):
                        for s,state in zip(('k','v'),pair):caches[w][s].store_quantized(a*FS,b*FS,state)
                    controls[w].quantize(a*FS,b*FS,b*FS,store_callback)
                for s,state in zip(('k','v'),pair):ref[s].store_quantized(a*FS,b*FS,state)
                require(all(caches[w][s].quantized_spans[-1]['quant_data'] is state
                            for w in caches for s,state in zip(('k','v'),pair)),'Packet sharing not exact')
                packets.append(pair);result['compressions'].append(record);guard()
            for step in range(5):
                guard()
                q,k,v=[torch.randn((1,3*FS,HEADS,DIM),generator=generator,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
                ref['k'].write(start*FS,end*FS,k);ref['v'].write(start*FS,end*FS,v)
                phase['name']='reference'
                rk=ref['k'].read(0,end*FS)
                # Avoid retaining raw V while native full-prefix FP64 RoPE peaks.
                torch.cuda.empty_cache()
                count['rope_long_reference']+=1
                rkr=originals['long_rope'](rk,grid,freqs).type_as(v)
                rv=ref['v'].read(0,end*FS)
                rq=observed_rope(q,grid,freqs,start_frame=start).type_as(v)
                for w in (21,63,126):
                    left=max(0,end-w)*FS
                    expected.update(raw_k=rk[:,left:],raw_v=rv[:,left:],
                                    rope_k=rkr[:,left:],rope_q=rq,
                                    raw_k_equal=False,raw_v_equal=False)
                    checkpoint=step==4 and block in CHECKPOINTS[w]
                    def checked_attention(aq,ak,av):
                        qe=exact(aq,expected['rope_q']);ke=exact(ak,expected['rope_k'])
                        require(qe and ke,'Absolute RoPE mismatch')
                        require(exact(av,expected['raw_v']),'Attention V mismatch')
                        row={'window_frames':w,'block':block,'step':step,
                             'start_token':left,'end_token':end*FS,
                             'frontier_tokens':controls[w].frontier_tokens,
                             'capacity_tokens':caches[w]['k'].kv_cache_size,
                             'raw_k_equal':expected['raw_k_equal'],'raw_v_equal':expected['raw_v_equal'],
                             'q_rope_equal':qe,'k_rope_equal':ke}
                        require(row['raw_k_equal'] and row['raw_v_equal'],'Read observer was bypassed')
                        result['reads'].append(row)
                        if checkpoint:
                            actual=native_fa(aq,ak,av)
                            baseline=native_fa(expected['rope_q'],expected['rope_k'],expected['raw_v'])
                            repeat=native_fa(expected['rope_q'],expected['rope_k'],expected['raw_v'])
                            stable=exact(baseline,repeat);equal=exact(actual,baseline)
                            result['attention_records'].append({'window_frames':w,'block':block,'step':step,
                                                                'label':CHECKPOINTS[w][block],
                                                                'repeat_equal':stable,'adapter_reference_equal':equal})
                            if not stable:raise Inconclusive('Native FA2 same-input repeat differs')
                            require(equal,'Native FA2 adapter/reference mismatch despite identical QKV')
                            del actual,baseline,repeat
                        return None  # No surrogate output and no unregistered FA2 timing at other reads.
                    phase['name']='adapter_attention'
                    controls[w].attend(0,caches[w],current_start=start*FS,q=q,k=k,v=v,grid_sizes=grid,
                                       freqs=freqs,rope_fn=observed_rope,attention_fn=checked_attention)
                    result['reads'][-1].update(global_end_index=int(caches[w]['global_end_index'].item()),
                                              local_end_index=int(caches[w]['local_end_index'].item()))
                    expected.clear()
                del rk,rkr,rv,rq,q,k,v
            for pair,record in zip(packets,result['compressions']):
                after=content_digest(pair)
                require(after==record['packet_before_sha256'],'Packed tensor or metadata mutation')
                record['packet_after_sha256']=after
            guard()
        require(len(result['reads'])==630 and len(result['attention_records'])==12,'Incomplete comparison grid')
        require(count['compress']==5 and count['prq_tensor_encode']==10 and count['fa2_kernel']==36
                and count['sdpa_fallback']==0,'Actual call counts differ')
        expected_uc=sum(r['packed_spans'] for r in result['cache_read_records'])+5
        require(count['uncompress_total']==expected_uc and count['decode_total']==2*expected_uc,'Native read/decoder call path mismatch')
        result['expected_uncompress_calls']=expected_uc
        result['controls_final']={str(w):controls[w].inspect() for w in controls}
        result['source_hashes_unchanged']=all(sha(p)==h for p,h in result['source_sha256'].items())
        require(result['source_hashes_unchanged'],'Source changed during probe')
        result['status']='passed'
    except BaseException as exc:
        result['status']='inconclusive' if isinstance(exc,(Inconclusive,TimeoutError,MemoryError)) or 'out of memory' in str(exc).lower() else 'failed'
        result['error']=repr(exc);result['traceback']=traceback.format_exc()
    finally:
        result['completed_utc']=time.time();result['wall_seconds']=time.time()-started
        result['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        result['CPU_threads']=4
        result['output_disk_bytes']=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
        if torch is not None and torch.cuda.is_initialized():
            result['GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            result['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        save_new(HERE/'RESULT.json',result)
    return 0 if result['status']=='passed' else 2


def process_group_rss(pgid):
    total=0;members={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            tail=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(tail[2])!=pgid:continue
            rss=next((int(line.split()[1])*1024 for line in (p/'status').read_text().splitlines() if line.startswith('VmRSS:')),0)
            total+=rss;members[int(p.name)]=int(tail[19])
        except (OSError,ValueError,StopIteration):pass
    return total,members


def launch_guarded(command, env, deadline, log_path, status, _after_spawn=None):
    """Same supervisor core for real worker and three CPU-only signal fixtures."""
    proc=None
    known={}
    old_term=signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise InterruptedError(f'Supervisor signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with Path(log_path).open('x') as log:
            # Defer Python signal exceptions until Popen ownership is registered.
            # Thread-local signal masks are insufficient with torch helper threads.
            pending=[]
            old_int=signal.getsignal(signal.SIGINT)
            def defer(signum,frame):pending.append(signum)
            signal.signal(signal.SIGINT,defer);signal.signal(signal.SIGTERM,defer)
            try:
                proc=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,
                                      env=env,start_new_session=True)
                if _after_spawn is not None:_after_spawn()
                _,initial=process_group_rss(proc.pid);known.update(initial)
                status['worker_pid']=proc.pid
            finally:
                signal.signal(signal.SIGINT,old_int);signal.signal(signal.SIGTERM,interrupted)
            if pending:
                if signal.SIGINT in pending:raise KeyboardInterrupt('Deferred supervisor SIGINT')
                raise InterruptedError('Deferred supervisor SIGTERM')
            status['worker_pid']=proc.pid
            while proc.poll() is None:
                rss,members=process_group_rss(proc.pid)
                known.update(members)
                status['peak_group_RSS_bytes']=max(status['peak_group_RSS_bytes'],rss)
                # Reserve cleanup time inside the same total cap.
                if time.time()>=deadline-1.5:raise Inconclusive('External total wall cap including cleanup')
                if rss>CAP:raise Inconclusive('External process group RSS cap')
                if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())>2**30:raise Inconclusive('External disk cap')
                time.sleep(.2)
            status['worker_exit_code']=proc.returncode
    finally:
        if proc is not None:
            _,members=process_group_rss(proc.pid)
            if proc.poll() is None and proc.pid in members:
                # A live, unreaped direct Popen child cannot be a recycled PID.
                known[proc.pid]=members[proc.pid]
            owned=any(known.get(pid)==start for pid,start in members.items())
            # Do not signal a re-used PID/PGID after the owned group disappeared.
            if owned:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            if proc.poll() is None:
                try:proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    _,current=process_group_rss(proc.pid)
                    if any(known.get(pid)==start for pid,start in current.items()):
                        os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait(timeout=.2)
            _,survivors=process_group_rss(proc.pid)
            if any(known.get(pid)==start for pid,start in survivors.items()):
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                time.sleep(.05)
            _,left=process_group_rss(proc.pid)
            status['remaining_group_members']=left
            status['worker_exit_code']=proc.returncode
            status['owned_cleanup_verified']=not left
        signal.signal(signal.SIGTERM,old_term)


def supervisor(lock_path):
    started=time.time();deadline=started+600
    verify_lock(lock_path)
    require(not (HERE/'RESULT.json').exists(),'Previous result exists; no retry')
    save_new(HERE/'ATTEMPT_CLAIM.json',{'started_utc':started,'absolute_deadline_utc':deadline,'attempt':1,'supervisor_pid':os.getpid()})
    status={'started_utc':started,'absolute_deadline_utc':deadline,'status':'running','peak_group_RSS_bytes':0}
    try:
        idle_gpu()
        env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        launch_guarded([sys.executable,str(Path(__file__).resolve()),'worker','--lock',str(Path(lock_path).resolve()),'--deadline',str(deadline)],
                       env,deadline,HERE/'WORKER.log',status)
        result=json.loads((HERE/'RESULT.json').read_text())
        if time.time()>deadline:
            status['status']='inconclusive';status['error']='Completion after absolute deadline'
        else:
            status['status']='completed' if status['worker_exit_code']==0 and result['status']=='passed' and status['owned_cleanup_verified'] else 'inconclusive'
            if result['status']=='failed':status['status']='failed'
        status['result_sha256']=sha(HERE/'RESULT.json')
    except BaseException as exc:
        status['status']='inconclusive';status['error']=repr(exc)
    finally:
        status['completed_utc']=time.time();status['wall_seconds']=time.time()-started
        save_new(HERE/'SUPERVISOR.json',status)
    return 0 if status['status']=='completed' else 2


def selftest():
    import torch
    torch.set_num_threads(4)
    require(not torch.cuda.is_initialized(),'Selftest cannot initialize CUDA')
    from types import SimpleNamespace
    cfg=SimpleNamespace(quant_type='triton-nstages-kmeans-int2',quant_block_size=64,num_prq_stages=1,asymmetric=False)
    shape=(1,2,4,128)
    codes=torch.arange(math.prod(shape),dtype=torch.int32).reshape(shape)%3
    chunks=codes.reshape(*shape[:-1],32,4)
    packed=((chunks[...,0]<<6)|(chunks[...,1]<<4)|(chunks[...,2]<<2)|chunks[...,3]).to(torch.uint8)
    centers=torch.arange(1*2*3*128).reshape(1,2,3,128).float()/32
    ids=torch.arange(8).reshape(1,2,4).remainder(3).to(torch.uint8)
    state={'centroids_list':[centers.bfloat16()],'cluster_ids_list':[ids],
           'residual_quant':packed,'scales':torch.full((1,2,4,2),.5).to(torch.float8_e4m3fn),
           'zeros':None,'info':{'quant_config':cfg,'output_dtype':torch.bfloat16}}
    expected=((codes-1).float()*.5+torch.gather(centers.bfloat16().float(),2,ids.long()[...,None].expand(shape))).bfloat16()
    require(exact(independent_unpack_cpu(state,shape),expected),'Synthetic independent unpack algebra')
    before=content_digest(state);require(content_digest(primitive(state))==before,'Metadata primitive equivalence')
    synthetic_packet=HERE/'SYNTHETIC_PACKET_V5.pt'
    require(not synthetic_packet.exists(),'Preserve prior synthetic artifact')
    torch.save(primitive(state),synthetic_packet)
    loaded=torch.load(synthetic_packet,map_location='cpu',weights_only=True)
    require(content_digest(loaded)==before,'CPU FP8 packet serialization changed content')
    cfg.asymmetric=True;require(content_digest(state)!=before,'Info mutation was missed');cfg.asymmetric=False
    require(not exact(torch.tensor([0.]),torch.tensor([-0.])),'Signed zero must differ')
    base=torch.arange(4,dtype=torch.bfloat16);single=base.as_strided((1,),(3,))
    require(exact(single,base[:1]),'Singleton nonunit stride hashing')
    ropes=rope_functions(torch)
    f=torch.cat([ropes['rope_params'](1024,44),ropes['rope_params'](1024,42),ropes['rope_params'](1024,42)],1)
    require(f.shape==(1024,64) and f.dtype==torch.complex128,'Original model frequency construction')
    import threading
    dummy=[]
    for mode in ('normal','interrupt','sigterm','early_interrupt','early_sigterm'):
        status={'peak_group_RSS_bytes':0};timer=None;caught=None
        if mode in ('interrupt','sigterm'):
            signum=signal.SIGINT if mode=='interrupt' else signal.SIGTERM
            timer=threading.Timer(.5,lambda signum=signum:os.kill(os.getpid(),signum));timer.start()
        try:
            launch_guarded([sys.executable,'-c','import time; time.sleep('+('0.05' if mode=='normal' else '30')+')'],
                           dict(os.environ),time.time()+5,HERE/f'DUMMY_{mode}_V5.log',status,
                           _after_spawn=(lambda:os.kill(os.getpid(),signal.SIGINT if mode=='early_interrupt' else signal.SIGTERM))
                           if mode.startswith('early_') else None)
        except BaseException as exc:caught=type(exc).__name__
        finally:
            if timer is not None:timer.cancel();timer.join()
        require(status.get('owned_cleanup_verified') and not status.get('remaining_group_members'),f'Dummy {mode} cleanup')
        require((mode=='normal' and caught is None and status['worker_exit_code']==0)
                or (mode in ('interrupt','early_interrupt') and caught=='KeyboardInterrupt')
                or (mode in ('sigterm','early_sigterm') and caught=='InterruptedError'),f'Dummy {mode} signal contract')
        dummy.append({'mode':mode,'caught':caught,**status})
    require(not torch.cuda.is_initialized(),'Selftest initialized CUDA')
    save_new(HERE/'SELFTEST_CPU_V5.json',{'status':'passed','checks':13,'CPU_threads':4,'GPU_calls':0,
                                     'actual_PRQ_calls':0,'model_loads':0,'source_sha256':sha(__file__),
                                     'supervisor_dummy':dummy,'freqs_shape':list(f.shape),'freqs_dtype':str(f.dtype),
                                     'scope':'small synthetic INT2 unpack/metadata/hash contract, not actual PRQ'})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['selftest','run','worker']);p.add_argument('--lock',type=Path);p.add_argument('--deadline',type=float)
    a=p.parse_args()
    if a.command=='selftest':selftest();return 0
    if a.lock is None:p.error('--lock is required')
    if a.command=='run':return supervisor(a.lock)
    if a.deadline is None:p.error('--deadline is required for supervised worker')
    return worker(a.lock,a.deadline)


if __name__=='__main__':raise SystemExit(main())
