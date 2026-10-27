"""Development-only model gates and cost measurements, with durable progress."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import p3_tf_fr as pilot
from anchor_identity_gate import IdentityQuantizer
from kvq import Quantizer
from kvq.lockstep import anchor_store, relative
from kvq.replay import ReferenceReplay, StorageNoiseQuantizer, make_store, store_cache


OUT = Path.home()/'sfq_data/feasibility'
STATUS = {}


def dump(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)


def report(phase, state, detail):
    STATUS[phase]=dict(state=state,detail=detail,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    dump(OUT/'status.json',STATUS)
    print(time.strftime('%H:%M:%S'),phase,state,detail,flush=True)
    (OUT.parent/'WORK_STATUS.md').write_text(
        'IN PROGRESS: P1/P3 development feasibility; main study not started.\n'+
        '\n'.join(f'{k}: {v["state"]} — {v["detail"]}' for k,v in STATUS.items())+
        '\nLive log: tail -f ~/sfq_data/feasibility.log\n'
        'CLI: cd ~/Quantization/src && ~/sfq_data/py scripts/progress.py --watch\n')


def fingerprint():
    h=hashlib.sha256()
    root=Path(__file__).resolve().parents[1]
    for folder in ('kvq','wan','pipeline','utils','scripts'):
        for p in sorted((root/folder).rglob('*.py')):
            h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def clean():
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()


def replay_cache(pipe, policy, quantizer):
    cap=pipe.local_attn_size*pipe.frame_seq_length
    sink=int(pipe.args.model_kwargs.sink_size)*pipe.frame_seq_length
    return [ReferenceReplay(make_store(policy,quantizer,cap,sink))
            for _ in range(pipe.num_transformer_blocks)]


def sampling(pipe, cond, policy, prompt, identity=False, diagnostic=False):
    phase=f'{policy}_'+('identity' if identity else 'rtn4')+f'_p{prompt}'+('_diagnostic' if diagnostic else '')
    report(phase,'running','allocating reference/replay/working states')
    clean();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    ref=pilot.bf16_cache(pipe,pilot.gpu)
    q=IdentityQuantizer() if identity else Quantizer('rtn',4,4)
    replay=replay_cache(pipe,policy,q)
    for r,teacher in zip(ref,replay):r['write_observer']=teacher
    fr=None if identity else store_cache([make_store(policy,Quantizer(),pipe.local_attn_size*pipe.frame_seq_length)
                                         for _ in ref],pilot.gpu)
    rx,tx,fx=[pilot.cross_cache(pipe,pilot.gpu) for _ in range(3)]
    noise=pilot.chunk_noise(pipe,63,0,pilot.gpu)
    outputs={'bf16':[],'tf':[]}
    if fr is not None:outputs['fr']=[]
    rows=[];history=[];work=None
    for chunk in range(21):
        work=None
        work=store_cache([r.snapshot() for r in replay],pilot.gpu)
        if policy=='chunk':
            # Compare actual attention-visible next-chunk histories one layer at
            # a time; final pending chunk remains BF16 in both constructions.
            for layer,(r,c) in enumerate(zip(ref,work)):
                old=c['store'];anchor=anchor_store(r,q,old.capacity,3*pipe.frame_seq_length)
                assert [s['start'] for s in old.segments]==[s['start'] for s in anchor.segments]
                for a,b in zip(old.segments,anchor.segments):
                    for x,y in zip(a['state'],b['state']):assert torch.equal(x.restore(),y.restore())
                if old.pending is not None:
                    assert all(torch.equal(x,y) for x,y in zip(old.pending,anchor.pending))
                del anchor
        r=pilot.run_chunk(pipe,ref,rx,cond,noise,chunk,0)
        if diagnostic:
            for layer,(w,reference) in enumerate(zip(work,ref)):
                def observe(k,v,layer=layer,reference=reference):
                    n=k.shape[1]
                    history.append(dict(chunk=chunk,layer=layer,call=len(history)//len(ref)%5,
                        tokens=n,k_rel=relative(k,reference['k'][:,:n]),
                        v_rel=relative(v,reference['v'][:,:n])))
                w['store'].history_observer=observe
        t=pilot.run_chunk(pipe,work,tx,cond,noise,chunk,0)
        archived=torch.load(Path.home()/f'sfq_data/verify/g3_base/latents/prompt{prompt:03d}/chunk_{chunk:04d}.pt',
                            map_location='cpu',weights_only=True)['latent']
        assert torch.equal(r.float().cpu(),archived),'reference differs from g3_base'
        if identity:assert torch.equal(r,t),'replay TF identity failed'
        assert all(x.calls==(chunk+1)*5 for x in replay),'reference write count differs'
        row=dict(chunk=chunk,tf_rel=relative(t,r),reference_bitexact=True)
        outputs['bf16'].append(r.cpu());outputs['tf'].append(t.cpu())
        if fr is not None:
            f=pilot.run_chunk(pipe,fr,fx,cond,noise,chunk,0)
            outputs['fr'].append(f.cpu());row['fr_rel']=relative(f,r)
        rows.append(row)
        report(phase,'running',f'chunk {chunk+1}/21; 5 reference writes/layer/chunk verified')
    torch.cuda.synchronize()
    meta=dict(policy=policy,prompt=prompt,identity=identity,diagnostic=diagnostic,
              seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,rows=rows,
              writes_per_layer=[r.calls for r in replay],history=history)
    folder=OUT/phase;folder.mkdir(exist_ok=True)
    # All final research tensors are retained. Equality is checked on tensors,
    # not torch.save container bytes, which may encode different filenames.
    for name,values in outputs.items():
        latent=torch.cat(values,1)
        if diagnostic:
            previous=torch.load(OUT/f'{policy}_rtn4_p{prompt}'/f'{name}.pt',weights_only=True)['latent']
            assert torch.equal(latent,previous),f'diagnostic changed {name}'
        torch.save(dict(latent=latent),folder/f'{name}.pt')
    if diagnostic:meta['all_outputs_bitexact_without_logging']=True
    dump(folder/'result.json',meta)
    report(phase,'pass',f'21/21; {meta["seconds"]:.1f}s; allocated peak {meta["peak_allocated_gib"]:.2f} GiB')
    return meta


def standalone(pipe,cond,policy,control=False):
    phase=policy+('_storage_noise' if control else '_fr_diagnostic')
    report(phase,'running','single trajectory; no reference/replay state on GPU')
    clean();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    quantizers=[StorageNoiseQuantizer(Quantizer(),987654+i) if control else Quantizer()
                for i in range(pipe.num_transformer_blocks)]
    cache=store_cache([make_store(policy,q,pipe.local_attn_size*pipe.frame_seq_length)
                       for q in quantizers],pilot.gpu)
    records=[]
    if not control:
        for c in cache:c['records']=records
    xc=pilot.cross_cache(pipe,pilot.gpu);noise=pilot.chunk_noise(pipe,63,0,pilot.gpu)
    values=[]
    for chunk in range(21):
        for c in cache:c['call_meta']=dict(chunk=chunk)
        values.append(pilot.run_chunk(pipe,cache,xc,cond,noise,chunk,0).cpu())
        report(phase,'running',f'chunk {chunk+1}/21')
    torch.cuda.synchronize()
    latent=torch.cat(values,1)
    if not control:
        old=torch.load(OUT/f'{policy}_rtn4_p0/fr.pt',weights_only=True)['latent']
        assert torch.equal(old,latent),'FR diagnostic changed output'
    norms=[r for q in quantizers for r in getattr(q,'records',[])]
    meta=dict(seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
              all_outputs_bitexact_without_logging=not control,records=records,
              noise_norms=norms,control_definition='per-encode innovation norm on own inputs; not read-error matched' if control else None)
    folder=OUT/phase;folder.mkdir(exist_ok=True)
    torch.save(dict(latent=latent),folder/'latent.pt');dump(folder/'result.json',meta)
    report(phase,'pass',f'{meta["seconds"]:.1f}s; allocated peak {meta["peak_allocated_gib"]:.2f} GiB')
    return meta


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.out;OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'status.json').exists():STATUS.update(json.loads((OUT/'status.json').read_text()))
    frozen=fingerprint();dump(OUT/'run.json',dict(source_sha256=frozen,pid=os.getpid(),purpose='development feasibility'))
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    try:
        report('setup','running','loading checkpoint and two development prompt embeddings')
        pipe=pilot.build(0)
        texts=(Path.home()/'sfq_data/verify/prompts2.txt').read_text().splitlines()
        conds=pilot.encode(pipe,texts);report('setup','pass','Wan 1.3B; 21-frame cache; seed 0')
        results={}
        for policy in ('rolling','chunk'):
            for prompt,cond in enumerate(conds):
                results[f'{policy}_identity_p{prompt}']=sampling(pipe,cond,policy,prompt,identity=True)
                clean()
        # Measurements start only after all identity and anchor/replay gates.
        report('gates','pass','84/84 TF chunks and archived BF16 references; actual replay path')
        for policy in ('rolling','chunk'):
            results[policy+'_rtn4']=sampling(pipe,conds[0],policy,0);clean()
            results[policy+'_tf_diagnostic']=sampling(pipe,conds[0],policy,0,diagnostic=True);clean()
            results[policy+'_fr_diagnostic']=standalone(pipe,conds[0],policy);clean()
            results[policy+'_noise']=standalone(pipe,conds[0],policy,control=True);clean()
        assert fingerprint()==frozen,'source changed during model feasibility'
        dump(OUT/'result.json',dict(passed=True,source_sha256=frozen,results=results))
        report('overall','pass','feasibility complete; main study has not started')
    except Exception as exc:
        report('overall','failed',f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':main()
