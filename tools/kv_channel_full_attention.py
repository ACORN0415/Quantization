"""B-2: capture the unchanged teacher, then replay frozen packed K on full attention."""
import argparse
import gc
import hashlib
import json
import math
import resource
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'src'), str(ROOT/'src/scripts')]
import torch
import torch.nn.functional as F
import kv_channel_budget_core as codec

OUT = ROOT/'src/experiments/kv_channel_budget_full_attention'
DATA = Path('/home/acorn0415/sfq_data/kv_channel_budget_full_attention')
B1 = ROOT/'src/experiments/kv_channel_budget_stage1'
DEADLINE = 1793812560.0  # 2026-11-04 17:16 UTC
LAYERS = (0, 14, 29)
ARMS = ('U120', 'U156', 'S120', 'R120')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*2**20), b''): h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def check_time():
    if time.time() >= DEADLINE: raise TimeoutError('B-2 90-minute limit')


def free_gpu():
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if any(s.strip().isdigit() for s in pids.splitlines()): raise RuntimeError('GPU occupied before start')


def record(path):
    return {'path': str(path), 'bytes': Path(path).stat().st_size, 'sha256': sha(path)}


def crop_blob(blob, drop=4680):
    """Evict whole pages without changing any surviving codes/scales/masks."""
    magic, n, h, d, g, bits, boost = codec.HEADER.unpack_from(blob)
    assert magic == b'KCB1' and drop % g == 0 and 0 < drop < n
    p, removed = n//g, drop//g
    sizes = [h*d*2, h*d*2]
    if boost: sizes += [h*d//8, h*g*d*2//8, h*g*boost*2//8]
    else: sizes += [h*g*d*bits//8]
    pos = codec.HEADER.size; parts = [codec.HEADER.pack(magic, n-drop, h, d, g, bits, boost)]
    for page_bytes in sizes:
        end = pos+p*page_bytes
        parts.append(blob[pos+removed*page_bytes:end]); pos = end
    assert pos == len(blob)
    return b''.join(parts)


def error(a, ref):
    diff = a.double()-ref.double()
    return {'mse': float(diff.square().mean()), 'normalized_squared_error': float(diff.square().sum()/ref.double().square().sum()),
            'max_abs': float(diff.abs().max())}


def fp32_attention(q, k, v, w, b):
    logits = q[0].permute(1,0,2).float() @ k[0].permute(1,2,0).float()/math.sqrt(k.shape[-1])
    probs = logits.softmax(-1)
    av = (probs @ v[0].permute(1,0,2).float()).permute(1,0,2).unsqueeze(0).contiguous()
    post = F.linear(av.flatten(2), w.float(), b.float())
    return probs, av, post


def decide(cases):
    decisions = []
    for prompt in (0, 4):
        rows = [c for c in cases if c['prompt'] == prompt]
        means = {a: sum(c['arms'][a]['native_post']['normalized_squared_error'] for c in rows)/3 for a in ARMS}
        fp = {a: sum(c['arms'][a]['fp32_post']['normalized_squared_error'] for c in rows)/3 for a in ARMS}
        gains = {a: 100*(1-means['S120']/means[a]) for a in ('U120','U156','R120')}
        wins = sum(c['arms']['S120']['native_post']['normalized_squared_error'] < min(c['arms'][u]['native_post']['normalized_squared_error'] for u in ('U120','U156')) for c in rows)
        ok = all(gains[u] >= 10 for u in ('U120','U156')) and gains['R120'] > 0 and wins >= 2 and all(fp['S120'] < fp[u] for u in ('U120','U156'))
        decisions.append({'prompt': prompt, 'native_mean_nse': means, 'fp32_mean_nse': fp, 'native_reduction_percent': gains, 'layers_beating_both_uniform': wins, 'passes': ok})
    return decisions


def capture(prompt):
    import p3_tf_fr as pilot
    import wan.modules.causal_model as model
    free_gpu(); check_time()
    folder = DATA/f'p{prompt:02d}_s101'
    if folder.exists(): raise RuntimeError('Refuse overwrite capture')
    folder.mkdir(parents=True)
    r = {'status': 'running', 'prompt': prompt, 'seed': 101, 'started_utc': time.time(), 'model_forwards': 0,
         'source_sha256': pilot.source_hash(), 'code_sha256': sha(__file__), 'registration_sha256': sha(OUT/'REGISTRATION_KO.txt'), 'artifacts': [], 'chunks': []}
    save(folder/'manifest.json', r)
    assert r['source_sha256'] == json.loads((B1/'source_audit.json').read_text())['model_source_sha256']
    old_hashes = {x['path']:x['sha256'] for c in json.loads((B1/'results.json').read_text())['cases'] for x in c['inputs']}
    pipe = None; handles = []; original = model.attention
    try:
        pipe = pilot.build(101)
        text = (ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines()[prompt]
        cond = pilot.encode(pipe, [text])[0]
        cache = pilot.bf16_cache(pipe, pilot.gpu); xc = pilot.cross_cache(pipe, pilot.gpu)
        noise = pilot.chunk_noise(pipe, 126, 101, pilot.gpu)
        reference = Path(f'/home/acorn0415/sfq_data/policy_study/reference/p{prompt:02d}_s101/bf16.pt')
        expected = torch.load(reference, map_location='cpu', weights_only=True)['latent']
        r['reference'] = record(reference)
        modules = [m for m in pipe.generator.modules() if isinstance(m, model.CausalWanSelfAttention)]
        assert len(modules) == 30
        counts = {i: 0 for i in LAYERS}; active = {}; pending = {}
        def pre(layer):
            def hook(module, args):
                active.clear()
                if layer in LAYERS:
                    count = counts[layer]; counts[layer] += 1
                    active.update(layer=layer, chunk=count//5, call=count%5)
            return hook
        def post(layer):
            def hook(module, args, output):
                if layer in pending:
                    payload = pending.pop(layer); payload['native_post'] = output.detach().cpu()
                    path = folder/f'l{layer:02d}_c7.pt'; torch.save(payload, path)
                    r['artifacts'].append(record(path)); save(folder/'manifest.json', r)
            return hook
        for layer, module in enumerate(modules):
            handles.append(module.register_forward_pre_hook(pre(layer)))
            if layer in LAYERS: handles.append(module.register_forward_hook(post(layer)))
        def attention(q, k, v, *args, **kwargs):
            context = dict(active); active.clear()
            y = original(q, k, v, *args, **kwargs)
            if context.get('chunk') == 7 and context.get('call') == 0:
                layer = context['layer']; assert not args and not kwargs
                assert tuple(k.shape) == (1,32760,12,128) and q.shape[1] == 4680
                indices = torch.linspace(0, q.shape[1]-1, 32, device=q.device).long()
                payload = {'q':q.detach().cpu(), 'k':k.detach().cpu(), 'v':v.detach().cpu(), 'native_av':y.detach().cpu(),
                           'wo': modules[layer].o.weight.detach().cpu(), 'bias': modules[layer].o.bias.detach().cpu(),
                           'indices': indices.cpu(), 'layer': layer, 'prompt': prompt, 'chunk':7, 'call':0}
                old = Path(f'/home/acorn0415/sfq_data/postwo_rounding/p{prompt:02d}_s101')
                for filename in (old/f'l{layer:02d}_c6.pt',old/f'l{layer:02d}_c7.pt'):
                    assert sha(filename) == old_hashes[str(filename)]
                c6 = torch.load(old/f'l{layer:02d}_c6.pt', map_location='cpu', weights_only=True)
                c7 = torch.load(old/f'l{layer:02d}_c7.pt', map_location='cpu', weights_only=True)
                assert same(payload['k'][:,:28080], c6['key'][:,4680:]), 'Historical prefix differs'
                assert same(payload['q'][:,payload['indices']], c7['query']), 'Teacher query differs'
                assert same(payload['wo'], c6['wo']), 'Output weight differs'
                pending[layer] = payload
            return y
        model.attention = attention
        def count_model(module, args):
            check_time(); r['model_forwards'] += 1
            if r['model_forwards'] > 40: raise RuntimeError('Forward cap')
        handles.append(pipe.generator.register_forward_pre_hook(count_model))
        for chunk in range(8):
            check_time(); y = pilot.run_chunk(pipe, cache, xc, cond, noise, chunk, 101).cpu()
            old = expected[:,chunk*3:(chunk+1)*3]
            old_cast = old.to(y.dtype)
            assert same(old_cast.to(old.dtype), old), 'Reference cast loses information'
            assert same(y, old_cast), f'Teacher byte identity failed chunk {chunk}'
            r['chunks'].append({'chunk': chunk, 'reference_dtype': str(old.dtype), 'output_dtype': str(y.dtype), 'byte_equal_after_lossless_cast': True})
            save(folder/'manifest.json', r); print('capture',prompt,chunk+1,'/8',flush=True)
        assert not pending and len(r['artifacts']) == 3 and all(n==40 for n in counts.values())
        assert pilot.source_hash() == r['source_sha256']
        r.update(status='completed', selected_layer_calls=counts)
    except BaseException as exc:
        r.update(status='failed', error=repr(exc), traceback=traceback.format_exc()); raise
    finally:
        model.attention = original
        for handle in handles: handle.remove()
        r.update(finished_utc=time.time(), max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(folder/'manifest.json', r)


def evaluate():
    from wan.modules.attention import attention
    free_gpu(); check_time()
    path = OUT/'results.json'
    if path.exists(): raise RuntimeError('Refuse overwrite evaluation')
    r = {'status':'running','started_utc':time.time(),'code_sha256':sha(__file__),'registration_sha256':sha(OUT/'REGISTRATION_KO.txt'),
         'model_forwards':0,'attention_replays':0,'cases':[]}
    b1 = json.loads((B1/'results.json').read_text())
    try:
        for prompt in (0,4):
            folder = DATA/f'p{prompt:02d}_s101'
            manifest = json.loads((folder/'manifest.json').read_text()); assert manifest['status']=='completed'
            for layer in LAYERS:
                check_time(); source = folder/f'l{layer:02d}_c7.pt'
                item = next(x for x in manifest['artifacts'] if x['path']==str(source)); assert sha(source)==item['sha256']
                d = torch.load(source,map_location='cpu',weights_only=True)
                q,k,v,w,b = (d[n].cuda() for n in ('q','k','v','wo','bias'))
                y = attention(q,k,v); r['attention_replays']+=1
                z = F.linear(y.flatten(2),w,b)
                assert same(y.cpu(),d['native_av']) and same(z.cpu(),d['native_post']), 'Native no-intervention replay differs'
                small = d['q'][:,d['indices']]; pr,av,post = fp32_attention(small,d['k'],d['v'],d['wo'],d['bias'])
                case = {'prompt':prompt,'layer':layer,'capture':item,'native_identity_passed':True,'arms':{},
                        'common_current_k_bytes':4680*12*128*2,'common_all_v_bytes':32760*12*128*2}
                r['cases'].append(case); save(path,r)
                raw = {'reference_av':d['native_av'],'reference_post':d['native_post'],'fp32_reference_av':av,'fp32_reference_post':post}
                previous = next(x for x in b1['cases'] if x['prompt']==prompt and x['layer']==layer)
                for arm in ARMS:
                    check_time(); item1 = next(x for x in previous['arms'] if x['arm']==arm)['payload']
                    assert sha(item1['path'])==item1['sha256']
                    blob = Path(item1['path']).read_bytes(); cropped = crop_blob(blob)
                    restored = codec.decode(cropped)
                    assert same(restored,codec.decode(blob)[:,4680:]), 'Eviction changed retained representation'
                    packed = DATA/f'p{prompt:02d}_l{layer:02d}_{arm}_active.bin'; packed.write_bytes(cropped)
                    assert len(cropped) <= 17611804
                    full = torch.cat((restored,d['k'][:,28080:]),dim=1)
                    kq = full.cuda(); ya = attention(q,kq,v); r['attention_replays']+=1
                    za = F.linear(ya.flatten(2),w,b); ya,za = ya.cpu(),za.cpu()
                    pa,ava,posta = fp32_attention(small,full,d['v'],d['wo'],d['bias'])
                    tv = .5*(pa-pr).abs().sum(-1)
                    row = {'payload':record(packed),'full_retained_decode_byte_equal':True,'native_av':error(ya,d['native_av']),
                           'native_post':error(za,d['native_post']),'fp32_av':error(ava,av),'fp32_post':error(posta,post),
                           'full_support_tv':float(tv.double().mean()),'full_support_tv_per_head_query':tv.tolist()}
                    case['arms'][arm]=row
                    raw[arm]={'av':ya,'post':za,'fp32_av':ava,'fp32_post':posta}
                    print('evaluate',prompt,layer,arm,row['native_post']['normalized_squared_error'],flush=True)
                    save(path,r); del blob,cropped,restored,full,kq,pa; gc.collect()
                result_file = DATA/f'p{prompt:02d}_l{layer:02d}_outputs.pt'; torch.save(raw,result_file)
                case['outputs']=record(result_file); save(path,r)
                assert sha(source)==case['capture']['sha256']
                del d,q,k,v,w,b,y,z,small,pr,av,post,raw; gc.collect();torch.cuda.empty_cache()
        assert len(r['cases'])==6 and r['attention_replays']==30
        r['decisions']=decide(r['cases']);r['advance']=all(x['passes'] for x in r['decisions']);r['status']='completed'
    except BaseException as exc:
        r.update(status='failed',error=repr(exc),traceback=traceback.format_exc()); raise
    finally:
        r.update(finished_utc=time.time(),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(path,r)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['capture','evaluate']);parser.add_argument('--prompt',type=int,choices=[0,4]);args=parser.parse_args()
    torch.set_num_threads(4);torch.set_grad_enabled(False);DATA.mkdir(parents=True,exist_ok=True)
    if args.mode=='capture':
        if args.prompt is None:parser.error('--prompt required')
        capture(args.prompt)
    else:evaluate()
