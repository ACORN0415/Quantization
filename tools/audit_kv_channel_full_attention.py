"""Read-only audit: independent bit-vector decoder and saved-output FP64 scores."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/experiments/kv_channel_budget_full_attention'
B1=ROOT/'src/experiments/kv_channel_budget_stage1'
ARMS=('U120','U156','S120','R120')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()

def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

def unpack(blob):
    magic,n,h,d,g,bits,boost=struct.unpack_from('<4s6I',blob)
    assert magic==b'KCB1' and n%g==0 and h==12 and d==128
    p=n//g;pos=28;count=p*h*d
    def take(size):
        nonlocal pos
        a=blob[pos:pos+size];assert len(a)==size;pos+=size;return a
    def field(bits,size):
        a=np.frombuffer(take((bits*size+7)//8),dtype=np.uint8)
        b=np.unpackbits(a,bitorder='little')[:bits*size].reshape(size,bits)
        return (b*(1<<np.arange(bits,dtype=np.uint8))).sum(1,dtype=np.uint8)
    lo=np.frombuffer(take(count*2),dtype='<f2').astype(np.float32).reshape(p,h,1,d)
    scale=np.frombuffer(take(count*2),dtype='<f2').astype(np.float32).reshape(p,h,1,d)
    if boost:
        assert bits==0 and boost==63
        mask=field(1,count).reshape(p*h,d).astype(bool);assert np.all(mask.sum(1)==63)
        codes=field(2,n*h*d).reshape(p*h,g,d)
        high=field(2,p*h*g*boost).reshape(p*h,g,boost)
        for i in range(p*h):codes[i][:,mask[i]] |= high[i]<<2
        codes=codes.reshape(p,h,g,d)
    else:
        assert bits==3
        codes=field(3,n*h*d).reshape(p,h,g,d)
    assert pos==len(blob)
    a=(codes.astype(np.float32)*scale+lo).transpose(0,2,1,3).reshape(1,n,h,d)
    return torch.from_numpy(a.copy()).bfloat16()

def score(a,b):
    x=a.float().numpy().astype(np.float64);y=b.float().numpy().astype(np.float64)
    delta=x-y
    return {'mse':float(np.mean(delta**2)), 'normalized_squared_error':float(np.sum(delta**2)/np.sum(y**2)), 'max_abs':float(np.max(np.abs(delta)))}

def main():
    torch.set_num_threads(4)
    r=json.loads((OUT/'results.json').read_text());assert r['status']=='completed'
    assert r['finished_utc']<1793812560 and r['model_forwards']==0 and r['attention_replays']==30
    assert sha(ROOT/'tools/kv_channel_full_attention.py')==r['code_sha256']
    assert sha(OUT/'REGISTRATION_KO.txt')==r['registration_sha256']
    old=json.loads((B1/'results.json').read_text());scores={};fp_scores={};checked=0;max_delta=0.
    captures=[]
    for c in r['cases']:
        for item in (c['capture'],c['outputs']):assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
        x=torch.load(c['capture']['path'],map_location='cpu',weights_only=True)
        y=torch.load(c['outputs']['path'],map_location='cpu',weights_only=True)
        for name in ('av','post'):exact(x['native_'+name],y['reference_'+name])
        first=next(z for z in old['cases'] if z['prompt']==c['prompt'] and z['layer']==c['layer'])
        scores[c['prompt'],c['layer']]={}
        fp_scores[c['prompt'],c['layer']]={}
        for a in ARMS:
            row=c['arms'][a];item=row['payload'];assert sha(item['path'])==item['sha256']
            blob=Path(item['path']).read_bytes();assert len(blob)==item['bytes']<=17611804
            prior=next(z for z in first['arms'] if z['arm']==a)['payload'];assert sha(prior['path'])==prior['sha256']
            exact(unpack(blob),unpack(Path(prior['path']).read_bytes())[:,4680:])
            for typ,prefix in (('native',''),('fp32','fp32_')):
                for component in ('av','post'):
                    value=score(y[a][prefix+component],y[prefix+'reference_'+component])
                    for key in value:
                        ref=row[typ+'_'+component][key]
                        max_delta=max(max_delta,abs(value[key]-ref))
                        assert np.isclose(value[key],ref,rtol=2e-12,atol=1e-15)
                    checked+=1
            scores[c['prompt'],c['layer']][a]=score(y[a]['post'],y['reference_post'])['normalized_squared_error']
            fp_scores[c['prompt'],c['layer']][a]=score(y[a]['fp32_post'],y['fp32_reference_post'])['normalized_squared_error']
        manifest=Path(c['capture']['path']).parent/'manifest.json'
        if str(manifest) not in captures:
            m=json.loads(manifest.read_text());assert m['status']=='completed' and m['model_forwards']==40 and len(m['chunks'])==8
            assert m['finished_utc']<1793812560 and m['code_sha256']==r['code_sha256']
            assert sha(m['reference']['path'])==m['reference']['sha256']
            assert all(z['byte_equal_after_lossless_cast'] for z in m['chunks'])
            captures.append(str(manifest))
    for decision in r['decisions']:
        p=decision['prompt'];means={a:sum(scores[p,l][a] for l in (0,14,29))/3 for a in ARMS}
        assert all(np.isclose(means[a],decision['native_mean_nse'][a],rtol=2e-12,atol=1e-15) for a in ARMS)
        gains={a:100*(1-means['S120']/means[a]) for a in ('U120','U156','R120')}
        wins=sum(scores[p,l]['S120']<min(scores[p,l]['U120'],scores[p,l]['U156']) for l in (0,14,29))
        fp={a:sum(fp_scores[p,l][a] for l in (0,14,29))/3 for a in ARMS}
        assert all(np.isclose(fp[a],decision['fp32_mean_nse'][a],rtol=2e-12,atol=1e-15) for a in ARMS)
        ok=all(gains[a]>=10 for a in ('U120','U156')) and gains['R120']>0 and wins>=2 and all(fp['S120']<fp[a] for a in ('U120','U156'))
        assert ok==decision['passes'] and wins==decision['layers_beating_both_uniform']
    assert r['advance']==all(d['passes'] for d in r['decisions'])
    print(json.dumps({'status':'passed','gpu_calls':0,'audit_code_sha256':sha(__file__),'payloads_independently_decoded':24,
                      'saved_output_score_sets_recomputed':checked,'max_score_absolute_difference':max_delta,
                      'capture_manifests_checked':2,'reported_model_forwards':80,'attention_replays':30,
                      'decisions_recomputed':True,'advance':r['advance']},indent=2))

if __name__=='__main__':main()
