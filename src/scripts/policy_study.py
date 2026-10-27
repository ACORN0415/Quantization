"""Frozen P1/P3 study runner: gates, resumable measurements, final analysis."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import p3_tf_fr as pilot
from anchor_identity_gate import IdentityQuantizer
from kvq import Quantizer
from kvq.lockstep import relative
from kvq.memory import memory_report
from kvq.replay import ReferenceReplay,StorageNoiseQuantizer,RecordingQuantizer,NonfiniteControlError,RawState,make_store,store_cache

PLAN_PATH=ROOT/'experiments/policy_study_plan.json'
PROMPTS_PATH=ROOT/'experiments/policy_study_prompts.txt'
PLAN=json.loads(PLAN_PATH.read_text())
DATA=Path.home()/'sfq_data/policy_study'
FROZEN=''
QUEUE={}


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def tensor_hash(x):
    x=x.detach().cpu().contiguous()
    h=hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode())
    h.update(x.view(torch.uint8).numpy().tobytes());return h.hexdigest()


def source_hash():
    files=[]
    for folder in ('kvq','wan','pipeline','utils'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.extend(ROOT/p for p in ('scripts/policy_study.py','scripts/analyze_policy_study.py',
        'scripts/p3_tf_fr.py','scripts/anchor_identity_gate.py','configs/a1.yaml','configs/default_config.yaml',
        'experiments/policy_study_plan.json','experiments/policy_study_prompts.txt','experiments/P1_P3_MAIN_STUDY.md',
        'experiments/P1_P3_CONTROL_AMENDMENT.md'))
    h=hashlib.sha256()
    for p in sorted(files):h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def check_frozen():
    if source_hash()!=FROZEN:raise RuntimeError('frozen study source or protocol changed')


def status(case,stage,detail,state='running',**extra):
    utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    value=dict(state=state,detail=f'{case}: {stage}: {detail}',case=case,stage=stage,
               pid=os.getpid(),updated_utc=utc,**QUEUE,**extra)
    save_json(DATA/'status.json',{'study':value})
    print(time.strftime('%Y-%m-%d %H:%M:%S'),state,case,stage,detail,flush=True)
    (DATA.parent/'WORK_STATUS.md').write_text(
        f'POLICY STUDY: {state.upper()}\n{case}: {stage}: {detail}\n'+
        f'Cases complete: {QUEUE.get("completed_cases",0)}/{QUEUE.get("total_cases",80)}\n'+
        f'Updated UTC: {utc}; PID: {os.getpid()}\n'+
        'CLI: bash ~/Quantization/tools/watch_progress.sh\n'+
        'Log: tail -f ~/sfq_data/policy_study.log\n'+
        'Final analysis runs automatically after all registered cases.\n')


def release():
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()


def cache_size(cache):
    # Exclude diagnostic source shadows and Python callbacks. This is resident
    # simulation state, with hypothetical packing identified separately.
    stores=[c['store'] for c in cache]
    state=[dict(state=getattr(s,'state',None),residual=getattr(s,'residual',None),
                segments=getattr(s,'segments',None),pending=getattr(s,'pending',None)) for s in stores]
    def unwrap(x):
        if isinstance(x,RawState):return x.tensor
        if isinstance(x,dict):return {k:unwrap(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [unwrap(v) for v in x]
        return x
    result=memory_report(unwrap(state))
    result['diagnostic_source_bytes']=memory_report([getattr(s,'source',None) for s in stores])['resident_tensor_bytes']
    return result


def load_stage(folder,name):
    meta=folder/f'{name}_sampling.json';tensor=folder/f'{name}.pt'
    if not meta.exists():return None
    r=json.loads(meta.read_text())
    if r['source_sha256']!=FROZEN:raise RuntimeError(f'{meta}: source mismatch')
    if not tensor.exists() or file_hash(tensor)!=r['latent_file_sha256']:
        raise RuntimeError(f'{tensor}: missing or altered completed tensor')
    return r


def persist_stage(folder,name,latent,meta):
    check_frozen();folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{name}.pt';tmp=path.with_suffix('.tmp')
    torch.save({'latent':latent},tmp);tmp.replace(path)
    meta.update(source_sha256=FROZEN,latent_sha256=tensor_hash(latent),
                latent_file_sha256=file_hash(path))
    save_json(folder/f'{name}_sampling.json',meta)


def save_reference(folder,latent):
    folder.mkdir(parents=True,exist_ok=True);path=folder/'bf16.pt';digest=tensor_hash(latent)
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True)['latent']
        if not torch.equal(old,latent):raise RuntimeError('recomputed BF16 reference differs')
        saved=json.loads((folder/'bf16_sampling.json').read_text())
        if saved['source_sha256']!=FROZEN:raise RuntimeError('reference source mismatch')
    else:
        persist_stage(folder,'bf16',latent,dict(reference_bitexact=True))
    return digest


def paired_tf(pipe,cond,folder,reference_folder,policy,bits,seed,frames,case,archived=False):
    cached=load_stage(folder,'tf')
    if cached is not None:
        if not (reference_folder/'bf16.pt').exists():raise RuntimeError('cached TF missing BF16')
        return cached
    release();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    ref=pilot.bf16_cache(pipe,pilot.gpu)
    q=IdentityQuantizer() if bits==0 else Quantizer('rtn',bits,bits,PLAN['group'])
    capacity=pipe.local_attn_size*pipe.frame_seq_length
    replay=[ReferenceReplay(make_store(policy,q,capacity)) for _ in ref]
    for reference,teacher in zip(ref,replay):reference['write_observer']=teacher
    rx,tx=pilot.cross_cache(pipe,pilot.gpu),pilot.cross_cache(pipe,pilot.gpu)
    noise=pilot.chunk_noise(pipe,frames,seed,pilot.gpu)
    original=[];outputs=[];history=[];rows=[];work=None
    for chunk in range(frames//3):
        work=None;work=store_cache([r.snapshot() for r in replay],pilot.gpu)
        r=pilot.run_chunk(pipe,ref,rx,cond,noise,chunk,seed)
        if archived and chunk<21:
            old=torch.load(Path.home()/f'sfq_data/verify/g3_base/latents/prompt000/chunk_{chunk:04d}.pt',map_location='cpu',weights_only=True)['latent']
            if not torch.equal(r.float().cpu(),old):raise RuntimeError('development BF16 archived gate failed')
        for layer,(w,reference) in enumerate(zip(work,ref)):
            def observe(k,v,layer=layer,reference=reference):
                n=k.shape[1]
                history.append(dict(chunk=chunk,layer=layer,call=len(history)//len(ref)%5,
                    read_history_tokens=n,read_history_k_rel=relative(k,reference['k'][:,:n]),
                    read_history_v_rel=relative(v,reference['v'][:,:n]),
                    kv_error_phase='pre_attention_history',kv_error_reference='insertion_bf16_history'))
            w['store'].history_observer=observe
        t=pilot.run_chunk(pipe,work,tx,cond,noise,chunk,seed)
        if not bits and not torch.equal(t,r):raise RuntimeError('replay identity gate failed')
        if not all(x.calls==(chunk+1)*5 for x in replay):raise RuntimeError('reference write replay count failed')
        if not torch.isfinite(t).all() or not torch.isfinite(r).all():raise RuntimeError('nonfinite TF/reference')
        original.append(r.cpu());outputs.append(t.cpu());rows.append(dict(chunk=chunk,tf_rel=relative(t,r)))
        status(case,'TF',f'chunk {chunk+1}/{frames//3}')
    torch.cuda.synchronize()
    original=torch.cat(original,1);latent=torch.cat(outputs,1)
    meta=dict(policy=policy,bits=bits,seed=seed,frames=frames,rows=rows,history=history,
              seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
              cache_state=cache_size(work),writes_per_layer=[r.calls for r in replay],
              reference_latent_sha256=save_reference(reference_folder,original))
    persist_stage(folder,'tf',latent,meta);return meta


def standalone(pipe,cond,folder,reference_folder,policy,bits,seed,frames,prompt,case,control=False):
    name='noise' if control else 'fr'
    cached=load_stage(folder,name)
    if cached is not None:return cached
    release();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    quantizers=[]
    schedules=None
    if control:
        calibration=load_stage(folder,'fr')
        if calibration is None:raise RuntimeError('quantized FR calibration must precede noise control')
        schedules=[[r['norm'] for r in calibration['innovation_norms'] if r['layer']==i]
                   for i in range(pipe.num_transformer_blocks)]
    for layer in range(pipe.num_transformer_blocks):
        q=IdentityQuantizer() if bits==0 else Quantizer('rtn',bits,bits,PLAN['group'])
        if control:q=StorageNoiseQuantizer(q,PLAN['noise_seed_base']+prompt*10000+seed*100+layer,
                                           support=PLAN['noise_support'],target_norms=schedules[layer])
        else:q=RecordingQuantizer(q)
        quantizers.append(q)
    cache=store_cache([make_store(policy,q,pipe.local_attn_size*pipe.frame_seq_length) for q in quantizers],pilot.gpu)
    records=[]
    for c in cache:c['records']=records
    xc=pilot.cross_cache(pipe,pilot.gpu);noise=pilot.chunk_noise(pipe,frames,seed,pilot.gpu)
    reference=torch.load(reference_folder/'bf16.pt',map_location='cpu',weights_only=True)['latent']
    outputs=[];rows=[];numerical_failure=None
    for chunk in range(frames//3):
        for c in cache:c['call_meta']=dict(chunk=chunk)
        try:
            value=pilot.run_chunk(pipe,cache,xc,cond,noise,chunk,seed)
        except NonfiniteControlError as exc:
            if not control or bits==0:raise
            numerical_failure=dict(chunk=chunk,reason=str(exc));break
        if not torch.isfinite(value).all():
            if not control or bits==0:raise RuntimeError(f'nonfinite {name} at chunk {chunk}')
            numerical_failure=dict(chunk=chunk,reason='nonfinite noise latent')
            torch.save({'latent':value.cpu()},folder/f'noise_nonfinite_chunk_{chunk:04d}.pt')
            break
        cpu=value.cpu();ref=reference[:,chunk*3:(chunk+1)*3]
        if not bits and not torch.equal(cpu,ref):raise RuntimeError('zero-noise/FR identity gate failed')
        outputs.append(cpu);rows.append(dict(chunk=chunk,rel=relative(cpu,ref)))
        status(case,name.upper(),f'chunk {chunk+1}/{frames//3}')
    torch.cuda.synchronize()
    observations=[dict(layer=i,**r) for i,q in enumerate(quantizers) for r in q.records]
    norms=observations if control else []
    if control and numerical_failure is None and any(len(q.records)!=len(schedules[i]) for i,q in enumerate(quantizers)):
        raise RuntimeError('noise calibration event count mismatch')
    ratios=[abs(r['actual_norm']/r['target_norm']-1) for r in norms if r['target_norm']]
    meta=dict(policy=policy,bits=bits,seed=seed,frames=frames,rows=rows,history=records,noise_norms=norms,
              state='nonfinite' if numerical_failure else 'finite',numerical_failure=numerical_failure,
              innovation_norms=observations if not control else [],
              max_noise_norm_relative_mismatch=max(ratios,default=0),seconds=time.monotonic()-started,
              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,cache_state=cache_size(cache),
              reference_latent_sha256=tensor_hash(reference))
    if numerical_failure:
        status(case,'CONTROL_INSTABILITY',f'chunk {numerical_failure["chunk"]}: {numerical_failure["reason"]}; retained, no replacement')
    partial=torch.cat(outputs,1) if outputs else reference[:,:0].clone()
    persist_stage(folder,name,partial,meta);return meta


def quality(pipe,folder,name,case):
    out=folder/f'{name}_quality.json'
    if out.exists():
        value=json.loads(out.read_text())
        if value['source_sha256']!=FROZEN:raise RuntimeError('quality source mismatch')
        if value['latent_file_sha256']!=file_hash(folder/f'{name}.pt'):raise RuntimeError('quality tensor mismatch')
        return value
    import pyiqa
    from torchvision.io import write_video
    release();started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    status(case,'QUALITY',name+' decoding raw frames')
    pipe.vae.to(pilot.gpu);metric=pyiqa.create_metric('musiq',device=pilot.gpu)
    latent=torch.load(folder/f'{name}.pt',map_location='cpu',weights_only=True)['latent']
    video=(pipe.vae.decode_to_pixel(latent.to(pilot.gpu),use_cache=False)*.5+.5).clamp(0,1)
    if not torch.isfinite(video).all():raise RuntimeError('nonfinite decoded video')
    scores=[]
    for chunk in range(latent.shape[1]//3):
        frame=min(video.shape[1]-1,round((chunk*3+1.5-1)*4+1))
        score=float(metric(video[:,frame].float()).item())
        if not __import__('math').isfinite(score):raise RuntimeError('nonfinite MUSIQ')
        scores.append(dict(chunk=chunk,pixel_frame=frame,musiq=score))
        if chunk%7==0:status(case,'QUALITY',f'{name}: MUSIQ {chunk+1}/{latent.shape[1]//3}')
    pixels=(video[0].permute(0,2,3,1)*255).to(torch.uint8).cpu()
    motion=[float((pixels[i].float()-pixels[i-1].float()).abs().mean().item()/255) for i in range(1,len(pixels))]
    write_video(str(folder/f'{name}.mp4'),pixels,fps=16)
    torch.cuda.synchronize()
    result=dict(source_sha256=FROZEN,latent_file_sha256=file_hash(folder/f'{name}.pt'),
                per_chunk=scores,musiq_mean=sum(s['musiq'] for s in scores)/len(scores),
                motion_absdiff_mean=sum(motion)/len(motion),pixel_frames=len(pixels),
                seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                metric_input='raw decoded frame before codec')
    save_json(out,result)
    pipe.vae.model.clear_cache();pipe.vae.to('cpu');del metric,video,pixels,latent;release()
    return result


def run_case(pipe,cond,folder,ref_folder,policy,bits,seed,frames,prompt,case,decode=True,archived=False):
    done=folder/'complete.json'
    if done.exists():
        old=json.loads(done.read_text())
        if old['source_sha256']!=FROZEN:raise RuntimeError('completed case source mismatch')
        for name in ('tf','fr','noise'):
            if load_stage(folder,name) is None:raise RuntimeError('completed case missing stage')
        return old
    started=time.monotonic();folder.mkdir(parents=True,exist_ok=True)
    tf=paired_tf(pipe,cond,folder,ref_folder,policy,bits,seed,frames,case,archived=archived);release()
    fr=standalone(pipe,cond,folder,ref_folder,policy,bits,seed,frames,prompt,case);release()
    noise=standalone(pipe,cond,folder,ref_folder,policy,bits,seed,frames,prompt,case,control=True);release()
    if decode:
        quality(pipe,ref_folder,'bf16',case)
        for name in ('tf','fr','noise'):
            if name=='noise' and noise['state']=='nonfinite':
                save_json(folder/'noise_quality.json',dict(source_sha256=FROZEN,
                    latent_file_sha256=noise['latent_file_sha256'],state='nonfinite',
                    numerical_failure=noise['numerical_failure'],per_chunk=[]))
            else:quality(pipe,folder,name,case)
    check_frozen()
    meta=dict(source_sha256=FROZEN,case=case,prompt_index=prompt,seed=seed,policy=policy,bits=bits,
              frames=frames,seconds=time.monotonic()-started,decoded=decode,
              noise_nonfinite=noise['state']=='nonfinite',
              tf_seconds=tf['seconds'],fr_seconds=fr['seconds'],noise_seconds=noise['seconds'],
              max_noise_norm_relative_mismatch=noise['max_noise_norm_relative_mismatch'])
    save_json(folder/'latent_manifest.json',{'retained':True,'artifacts':[
        dict(path=f'{n}.pt',bytes=(folder/f'{n}.pt').stat().st_size,sha256=file_hash(folder/f'{n}.pt')) for n in ('tf','fr','noise')]})
    detail=f'{meta["seconds"]:.1f}s'+('; secondary noise numerically unstable, retained' if meta['noise_nonfinite'] else '')
    save_json(done,meta);status(case,'COMPLETE',detail,state='pass');return meta


def gates(pipe):
    status('gates','SETUP','development prompts only; held-out sampling disabled')
    text=(Path.home()/'sfq_data/verify/prompts2.txt').read_text().splitlines()[0]
    cond=pilot.encode(pipe,[text])[0];records=[]
    for policy in PLAN['policies']:
        for bits,frames,decode in ((0,63,False),(4,63,False),(2,126,True)):
            name=f'{policy}_int{bits}_f{frames}'
            folder=DATA/'gates'/name;reference=DATA/'gates'/f'reference_f{frames}'
            row=run_case(pipe,cond,folder,reference,policy,bits,0,frames,0,name,decode=decode,archived=True)
            if bits==4:
                for arm in ('bf16','tf','fr'):
                    ours=torch.load((reference if arm=='bf16' else folder)/f'{arm}.pt',map_location='cpu',weights_only=True)['latent']
                    old=torch.load(Path.home()/f'sfq_data/feasibility/{policy}_rtn4_p0/{arm}.pt',map_location='cpu',weights_only=True)['latent']
                    if not torch.equal(ours,old):raise RuntimeError('INT4 compatibility gate failed: '+policy+'/'+arm)
                row['compatibility_with_feasibility']=True
            if row['max_noise_norm_relative_mismatch']>.05:raise RuntimeError('rounded noise norm mismatch exceeds 5% development bound')
            records.append(row)
    check_frozen()
    value=dict(passed=True,source_sha256=FROZEN,plan_sha256=file_hash(PLAN_PATH),
               prompt_sha256=file_hash(PROMPTS_PATH),records=records,
               git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save_json(DATA/'gates.json',value);status('gates','COMPLETE','all prerequisites passed; registration seal required before study',state='pass')


def study(pipe):
    gate=json.loads((DATA/'gates.json').read_text())
    seal_path=ROOT/'experiments/policy_study_registration.json'
    seal=json.loads(seal_path.read_text())
    if not gate['passed'] or gate['source_sha256']!=FROZEN:raise RuntimeError('matching-source model gates required')
    for key,expected in [('source_sha256',FROZEN),('plan_sha256',file_hash(PLAN_PATH)),
                         ('prompt_sha256',file_hash(PROMPTS_PATH)),('gate_sha256',file_hash(DATA/'gates.json'))]:
        if seal[key]!=expected:raise RuntimeError('registration seal mismatch: '+key)
    committed=subprocess.check_output(['git','show','HEAD:src/experiments/policy_study_registration.json'],cwd=ROOT,text=True)
    if json.loads(committed)!=seal:raise RuntimeError('registration must be committed before held-out sampling')
    texts=PROMPTS_PATH.read_text().splitlines()
    if len(texts)!=PLAN['prompts'] or len(set(texts))!=len(texts):raise RuntimeError('prompt count or duplicates')
    dev=[]
    for p in (ROOT/'experiments/p3_pilot_prompts.txt',Path.home()/'sfq_data/verify/prompts2.txt'):
        dev.extend(p.read_text().splitlines())
    if set(texts)&set(dev):raise RuntimeError('development prompt in study set')
    conds=pilot.encode(pipe,texts)
    completed=[];total=len(texts)*len(PLAN['seeds'])*len(PLAN['bits'])*len(PLAN['policies'])
    QUEUE.update(completed_cases=0,total_cases=total)
    for prompt,cond in enumerate(conds):
        for seed in PLAN['seeds']:
            for bits in PLAN['bits']:
                for policy in PLAN['policies']:
                    check_frozen();name=f'p{prompt:02d}_s{seed}_{policy}_int{bits}'
                    folder=DATA/'cases'/name;reference=DATA/'reference'/f'p{prompt:02d}_s{seed}'
                    status(name,'START',f'{len(completed)}/{total} cases complete')
                    for attempt in range(PLAN['retry_infrastructure']+1):
                        retry=False
                        try:
                            result=run_case(pipe,cond,folder,reference,policy,bits,seed,PLAN['frames'],prompt,name)
                        except (OSError,torch.cuda.OutOfMemoryError) as exc:
                            failure=dict(case=name,attempt=attempt+1,error=f'{type(exc).__name__}: {exc}',
                                         source_sha256=FROZEN,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                            save_json(DATA/'failures'/f'{name}_attempt{attempt+1}.json',failure)
                            if attempt==PLAN['retry_infrastructure']:raise
                            status(name,'RETRY',failure['error']);retry=True
                        if not retry:break
                        release()
                    completed.append(result);QUEUE['completed_cases']=len(completed)
                    durations=sorted(r['seconds'] for r in completed)
                    QUEUE['eta_hours_rough']=(total-len(completed))*durations[len(durations)//2]/3600
                    save_json(DATA/'queue.json',dict(source_sha256=FROZEN,complete=len(completed),total=total,cases=completed,**QUEUE))
    check_frozen();status('study','ANALYSIS','all 80 registered cases complete')
    from analyze_policy_study import analyze
    analyze(DATA,ROOT/'experiments/policy_study_results')
    status('study','COMPLETE','all measurements and final analysis complete',state='pass')


def main():
    global DATA,FROZEN
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['gates','study','analyze'],required=True)
    ap.add_argument('--data',type=Path,default=DATA);args=ap.parse_args();DATA=args.data
    DATA.mkdir(parents=True,exist_ok=True);FROZEN=source_hash()
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    try:
        if args.mode=='analyze':
            from analyze_policy_study import analyze
            analyze(DATA,ROOT/'experiments/policy_study_results');return
        status(args.mode,'MODEL','loading fixed checkpoint')
        pipe=pilot.build(0)
        if pipe.local_attn_size!=PLAN['context_frames']:raise RuntimeError('context mismatch')
        if args.mode=='gates':gates(pipe)
        else:study(pipe)
    except Exception as exc:
        status(args.mode,'FAILED',f'{type(exc).__name__}: {exc}',state='failed')
        raise


if __name__=='__main__':main()
