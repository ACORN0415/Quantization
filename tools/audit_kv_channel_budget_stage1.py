"""Independent NumPy bit-stream decode plus CPU attention-score audit of B-1."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/experiments/kv_channel_budget_stage1'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def independent_decode(blob,key,mode,seed):
    magic,n,h,d,g,bits,boost=struct.unpack_from('<4s6I',blob)
    assert magic==b'KCB1' and n==32760 and h==12 and d==128 and n%g==0
    pages=n//g;size=pages*h*d;pos=28
    def take(length):
        nonlocal pos
        x=blob[pos:pos+length];assert len(x)==length;pos+=length;return x
    def get_codes(bits,count):
        raw=np.frombuffer(take((count*bits+7)//8),dtype=np.uint8)
        # A bit-vector decoder independent of the measured uint32-word decoder.
        fields=np.unpackbits(raw,bitorder='little')[:count*bits].reshape(count,bits)
        return np.sum(fields*(1 << np.arange(bits,dtype=np.uint8)),axis=1,dtype=np.uint8)
    lo=np.frombuffer(take(size*2),dtype='<f2').astype(np.float32).reshape(pages,h,1,d)
    scale=np.frombuffer(take(size*2),dtype='<f2').astype(np.float32).reshape(pages,h,1,d)
    if boost:
        assert boost==63 and bits==0
        mask=get_codes(1,size).astype(bool).reshape(pages,h,d)
        assert (mask.sum(-1)==63).all()
        shape=(pages,h,g,d)
        codes=get_codes(2,n*h*d).reshape(shape)
        hi=get_codes(2,pages*h*g*boost).reshape(pages*h,g,boost)
        ids=np.nonzero(mask.reshape(-1,d))[1].reshape(-1,boost)
        # Explicit page/head loop and per-channel scatter, without codec.decode.
        view=codes.reshape(pages*h,g,d)
        for i in range(pages*h):view[i][:,ids[i]] |= hi[i] << 2
        x=key[0].float().reshape(pages,g,h,d).permute(0,2,1,3)
        score=x.abs().mean(2) if mode=='selected' else torch.rand((pages,h,d),generator=torch.Generator().manual_seed(seed))
        expected=torch.zeros((pages,h,d),dtype=torch.bool)
        expected.scatter_(-1,torch.argsort(score,dim=-1,descending=True,stable=True)[...,:63],True)
        assert np.array_equal(mask,expected.numpy())
    else:
        assert bits==3 and mode=='uniform'
        codes=get_codes(3,n*h*d).reshape(pages,h,g,d)
    assert pos==len(blob)
    restored=(codes.astype(np.float32)*scale+lo).transpose(0,2,1,3).reshape(1,n,h,d)
    return torch.from_numpy(restored.copy()).bfloat16()


def tv(q,k,reference):
    logits=q[0].permute(1,0,2).float()@k[0].permute(1,2,0).float()/np.sqrt(128)
    probs=logits.softmax(-1)
    return .5*(probs-reference).abs().sum(-1)


def main():
    torch.set_num_threads(4)
    r=json.loads((OUT/'results.json').read_text());assert r['status']=='completed' and r['gpu_calls']==0
    assert r['finished_utc']<r['deadline_utc']
    for f,k in [('tools/kv_channel_budget_stage1.py','code_sha256'),('tools/kv_channel_budget_core.py','codec_sha256'),('src/experiments/kv_channel_budget_stage1/REGISTRATION_KO.txt','registration_sha256')]:assert sha(ROOT/f)==r[k]
    rows=[]
    for case in r['cases']:
        for e in case['inputs']:assert sha(e['path'])==e['sha256']
        c6,c7=[torch.load(e['path'],map_location='cpu',weights_only=True) for e in case['inputs']]
        key=c6['key'];qs=[c6['query'],c7['query']]
        refs=[(q[0].permute(1,0,2).float()@key[0].permute(1,2,0).float()/np.sqrt(128)).softmax(-1) for q in qs]
        scores={};counts={}
        for arm in case['arms']:
            e=arm['payload'];assert sha(e['path'])==e['sha256'];blob=Path(e['path']).read_bytes()
            assert len(blob)==arm['payload_bytes']<=20547100
            y=independent_decode(blob,key,arm['mode'],20261104+case['prompt']*100+case['layer'])
            assert torch.isfinite(y).all()
            for c,q,ref in zip((6,7),qs,refs):
                t=tv(q,y,ref);saved=torch.tensor(arm[f'q{c}']['tv_per_head_query'])
                assert torch.equal(t,saved)
                assert float(t.double().mean())==arm[f'q{c}']['tv_mean']
            scores[arm['arm']]=float(t.double().mean());counts[arm['arm']]=len(blob)
        assert counts=={'U120':20547100,'U156':20160028,'S120':20501236,'R120':20501236}
        rows.append({'prompt':case['prompt'],'layer':case['layer'],'q7_tv':scores,'payload_bytes':counts,'independent_decode_and_both_query_scores_equal':True})
    decisions=[]
    for p in (0,4):
        rr=[x for x in rows if x['prompt']==p]
        means={a:float(np.mean([x['q7_tv'][a] for x in rr])) for a in ('U120','U156','S120','R120')}
        improvements={a:100*(1-means['S120']/means[a]) for a in ('U120','U156','R120')}
        wins=sum(x['q7_tv']['S120']<min(x['q7_tv']['U120'],x['q7_tv']['U156']) for x in rr)
        d={'prompt':p,'mean_tv':means,'selected_improvement_percent':improvements,'layers_beating_both_uniform':wins,
           'passes':improvements['U120']>=10 and improvements['U156']>=10 and improvements['R120']>0 and wins>=2}
        decisions.append(d)
    assert decisions==r['decisions'] and all(x['passes'] for x in decisions)==r['advance']
    print(json.dumps({'status':'passed','gpu_calls':0,'independent_numpy_bit_decoder':True,'validated_payloads':24,'validated_query_distributions':48,'rows':rows,'decisions':decisions,'advance':r['advance'],'audit_code_sha256':sha(__file__)},indent=2))


if __name__=='__main__':main()
