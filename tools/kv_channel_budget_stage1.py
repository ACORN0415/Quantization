"""Bounded CPU past-support attention diagnostic. No model or GPU invocation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch
import kv_channel_budget_core as codec

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/experiments/kv_channel_budget_stage1'
DATA=Path('/home/acorn0415/sfq_data/kv_channel_budget_stage1')
START=1793803620.
DEADLINE=START+2700
ARMS=[('U120',120,'uniform'),('U156',156,'uniform'),('S120',120,'selected'),('R120',120,'random')]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(p)


def attention(q,k):
    logits=q[0].permute(1,0,2).float() @ k[0].permute(1,2,0).float()/np.sqrt(k.shape[-1])
    return logits,logits.softmax(-1)


def compare(q,k,ref):
    logits,probs=attention(q,k);lr,pr=ref
    delta=logits-lr;centered=delta-delta.mean(-1,keepdim=True);lrc=lr-lr.mean(-1,keepdim=True)
    tv=.5*(probs-pr).abs().sum(-1)
    return {'tv_mean':float(tv.double().mean()),'tv_per_head_query':tv.tolist(),
            'logit_relative_l2':float(delta.double().norm()/lr.double().norm()),
            'centered_logit_relative_l2':float(centered.double().norm()/lrc.double().norm())}


def main():
    torch.set_num_threads(4);DATA.mkdir(parents=True,exist_ok=True)
    path=OUT/'results.json'
    if path.exists():raise RuntimeError('Refusing overwrite')
    r={'status':'running','registered_start_utc':START,'deadline_utc':DEADLINE,'gpu_calls':0,'model_forwards':0,
       'code_sha256':sha(__file__),'codec_sha256':sha(codec.__file__),'registration_sha256':sha(OUT/'REGISTRATION_KO.txt'),'cases':[]}
    try:
        r['unit_checks']=codec.unit_checks();save(path,r)
        manifest=json.loads((ROOT/'src/experiments/mixed_precision_world_review_20261104/input_availability.json').read_text())
        hashes={e['path']:e['sha256'] for e in manifest['files']}
        for prompt in (0,4):
            for layer in (0,14,29):
                if time.time()>=DEADLINE:raise TimeoutError('45-minute limit')
                files=[Path(f'/home/acorn0415/sfq_data/postwo_rounding/p{prompt:02d}_s101/l{layer:02d}_c{c}.pt') for c in (6,7)]
                for f in files:assert sha(f)==hashes[str(f)]
                c6,c7=[torch.load(f,map_location='cpu',weights_only=True) for f in files]
                key=c6['key'];assert list(key.shape)==[1,32760,12,128] and key.dtype==torch.bfloat16
                refs={str(c):attention(q,key) for c,q in ((6,c6['query']),(7,c7['query']))}
                case={'prompt':prompt,'layer':layer,'inputs':[{'path':str(f),'sha256':hashes[str(f)]} for f in files],
                      'support':'32760 historical keys only; q7 distribution renormalized on this prefix','arms':[]}
                r['cases'].append(case);save(path,r)
                for name,group,mode in ARMS:
                    if time.time()>=DEADLINE:raise TimeoutError('45-minute limit')
                    start=time.perf_counter();blob,restored=codec.encode(key,group,mode,20261104+prompt*100+layer)
                    prep=time.perf_counter()-start
                    assert len(blob)<=20547100
                    file=DATA/f'p{prompt:02d}_l{layer:02d}_{name}.bin';file.write_bytes(blob)
                    row={'arm':name,'group':group,'mode':mode,'payload_bytes':len(blob),'payload':{'path':str(file),'sha256':sha(file)},
                         'encode_validate_decode_cpu_s':prep,'full_pack_roundtrip_valid':True,
                         'key_relative_l2':float((restored.float()-key.float()).double().norm()/key.double().norm())}
                    for c,q in ((6,c6['query']),(7,c7['query'])):row[f'q{c}']=compare(q,restored,refs[str(c)])
                    case['arms'].append(row);save(path,r)
                    print(prompt,layer,name,len(blob),row['q7']['tv_mean'],flush=True)
                    del blob,restored;gc.collect()
                for f in files:assert sha(f)==hashes[str(f)]
                del refs,key,c6,c7;gc.collect()
        decisions=[]
        for p in (0,4):
            cases=[c for c in r['cases'] if c['prompt']==p]
            means={a:float(np.mean([next(x for x in c['arms'] if x['arm']==a)['q7']['tv_mean'] for c in cases])) for a,_,_ in ARMS}
            improvements={u:100*(1-means['S120']/means[u]) for u in ('U120','U156','R120')}
            win=sum(next(x for x in c['arms'] if x['arm']=='S120')['q7']['tv_mean'] < min(x['q7']['tv_mean'] for x in c['arms'] if x['arm'] in ('U120','U156')) for c in cases)
            passes=improvements['U120']>=10 and improvements['U156']>=10 and improvements['R120']>0 and win>=2
            decisions.append({'prompt':p,'mean_tv':means,'selected_improvement_percent':improvements,'layers_beating_both_uniform':win,'passes':passes})
        r['decisions']=decisions;r['advance']=all(x['passes'] for x in decisions);r['status']='completed'
    except BaseException as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        r['finished_utc']=time.time();r['max_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;save(path,r)


if __name__=='__main__':main()
