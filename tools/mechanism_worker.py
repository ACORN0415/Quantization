"""Separate GPU process for diagnostic-free timing, generation and raw scoring."""
import argparse
import atexit
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/scripts')]
import torch
import p3_tf_fr as pilot
import mechanism_run as diagnostic
from mechanism_core import make,state_bytes
from kvq.replay import store_cache
from kvq.memory import memory_report
from kvq.quant import Quantized

DATA=Path.home()/'sfq_data/mechanism'
FAILURE_PATH=None


def sha(path):return diagnostic.original.file_hash(Path(path))


def source_hash():
    h=hashlib.sha256(diagnostic.fingerprint().encode())
    for name in ('mechanism_worker.py','mechanism_followup.py','analyze_mechanism_fr.py'):
        p=ROOT/'tools'/name
        if not p.exists():raise RuntimeError(f'follow-up implementation incomplete: {name}')
        h.update(name.encode());h.update(p.read_bytes())
    h.update((ROOT/'src/experiments/mechanism_prompts_v0.txt').read_bytes())
    h.update((ROOT/'tools/evaluate_policy_supplement.py').read_bytes())
    for path in [ROOT/'src/checkpoints/self_forcing_dmd.pt',ROOT/'src/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',
                 Path(torch.hub.get_dir())/'checkpoints/dino_vitbase16_pretrain.pth',
                 Path(torch.hub.get_dir())/'checkpoints/raft_small_C_T_V2-01064c6d.pth']:
        h.update(sha(path).encode())
    return h.hexdigest()


def strict_mean(values):
    return None if any(v is None for v in values) else sum(values)/len(values)


@torch.inference_mode()
def raw_auxiliary(pixels):
    import cv2
    import numpy as np
    from evaluate_policy_supplement import Evaluator,subject_values,temporal
    evaluator=Evaluator();images=pixels.numpy();features=[]
    for lo in range(0,len(images),16):features.append(evaluator.features(list(images[lo:lo+16])))
    scores=subject_values(torch.cat(features)).numpy()
    pairs=[dict(kind='inside',chunk=c,i=12*c+o,j=12*c+o+1) for c in range(42) for o in (2,3)]
    pairs += [dict(kind='boundary',chunk=c,i=12*c-4,j=12*c-3) for c in range(1,42)]
    rows=[]
    for lo in range(0,len(pairs),4):
        group=pairs[lo:lo+4]
        a=torch.from_numpy(np.stack([cv2.resize(images[r['i']],(448,256)) for r in group])).permute(0,3,1,2).cuda().float()/255
        b=torch.from_numpy(np.stack([cv2.resize(images[r['j']],(448,256)) for r in group])).permute(0,3,1,2).cuda().float()/255
        n=len(group);flows=evaluator.raft(torch.cat((a,b))*2-1,torch.cat((b,a))*2-1,num_flow_updates=12)[-1]
        values=temporal(a,b,flows[:n],flows[n:])
        for index,r in enumerate(group):
            row=dict(r)
            for name,value in values.items():
                number=float(value[index])
                row[name]=None if name=='warp_error' and float(values['coverage'][index])==0 else number
                if row[name] is not None and not np.isfinite(row[name]):raise RuntimeError('nonfinite raw auxiliary metric')
            rows.append(row)
    windows={}
    for kind in ('boundary','inside'):
        selected=[r for r in rows if r['kind']==kind and r['chunk']>=21]
        windows['late_'+kind]={key:strict_mean([r[key] for r in selected]) for key in ('warp_error','coverage','flow_pixels','raw_difference')}
    b=windows['late_boundary']['warp_error'];i=windows['late_inside']['warp_error']
    return dict(subject_whole=float(scores.mean()),subject_late=float(scores[248:500].mean()),
                windows=windows,boundary_minus_inside_warp_error=None if b is None or i is None else b-i,
                pairs=rows,empty_pairs=sum(r['coverage']==0 for r in rows),
                metric_input='raw RGB uint8 before codec; DINO whole-frame proxy and sampled RAFT residual, exploratory')


class Telemetry:
    def __init__(self):self.rows=[];self.proc=None
    def start(self):
        fields='memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw'
        self.proc=subprocess.Popen(['nvidia-smi',f'--query-gpu={fields}','--format=csv,noheader,nounits','--loop-ms=100'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
        def loop():
            for line in self.proc.stdout:
                try:
                    values=[float(v.strip()) for v in line.split(',')]
                    rss=next(int(l.split()[1])*1024 for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:'))
                    self.rows.append(dict(wall=time.time(),gpu_memory_mib=values[0],gpu_utilization=values[1],temperature_c=values[2],sm_clock_mhz=values[3],power_w=values[4],rss_bytes=rss))
                except (ValueError,StopIteration):pass
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
        atexit.register(self.stop)
    def stop(self):
        if self.proc is not None:
            if self.proc.poll() is None:self.proc.terminate()
            self.proc.wait();self.thread.join(timeout=2);self.proc=None
            atexit.unregister(self.stop)


def memory():return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())


class OperatorTiming:
    """Profile-only CUDA intervals; nested intervals are not additive."""
    def __init__(self):self.events=[];self.logical_eviction_payload_bytes=0
    def timed(self,name,fn,*args,**kwargs):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record();value=fn(*args,**kwargs);end.record();self.events.append((name,begin,end));return value
    def install(self,cache):
        timer=self;self.restore=Quantized.restore;self.cat=torch.cat
        def restore(state):return timer.timed('restore',timer.restore,state)
        Quantized.restore=restore
        for c in cache:
            if 'store' not in c:continue
            store=c['store'];old_encode=store.quantizer.encode_pair
            store.quantizer.encode_pair=lambda k,v,fn=old_encode:timer.timed('encode',fn,k,v)
            old_attend=store.attend
            def attend(query,key,value,start,attention,local=True,record=None,fn=old_attend):
                def cat(*a,**kw):return timer.timed('storage_concat',timer.cat,*a,**kw)
                def actual_attention(*a,**kw):
                    torch.cat=timer.cat
                    try:return timer.timed('attention',attention,*a,**kw)
                    finally:torch.cat=cat
                torch.cat=cat
                try:return fn(query,key,value,start,actual_attention,local=local,record=record)
                finally:torch.cat=timer.cat
            store.attend=attend
            if hasattr(store,'write'):
                old_write=store.write
                def write(key,value,start,fn=old_write,s=store,**kw):
                    eviction=(start+key.shape[1]>s.global_end and key.shape[1]+s.local_end>s.capacity)
                    if eviction:
                        remaining=s.capacity-key.shape[1]
                        timer.logical_eviction_payload_bytes+=remaining*key.shape[0]*key.shape[2]*key.shape[3]*key.element_size()*2
                    return timer.timed('write_with_eviction' if eviction else 'write_without_eviction',fn,key,value,start,**kw)
                store.write=write
    def finish(self):
        Quantized.restore=self.restore;torch.cat=self.cat;torch.cuda.synchronize();values={}
        for name,begin,end in self.events:values.setdefault(name,[]).append(begin.elapsed_time(end))
        return dict(intervals={k:dict(count=len(v),sum_ms=sum(v),mean_ms=sum(v)/len(v),max_ms=max(v)) for k,v in values.items()},
                    logical_eviction_payload_bytes=self.logical_eviction_payload_bytes,
                    note='Profile-only nested stream intervals include host enqueue gaps; sums overlap and are not end-to-end wall time or DRAM bandwidth')


def execute(pipe,cond,arm,group,seed,frames,observe=False,profile=False):
    baseline=memory();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
    events=[]
    if profile:
        event=torch.cuda.Event(enable_timing=True);event.record();events.append(event)
    cache=pilot.bf16_cache(pipe,pilot.gpu) if arm=='BF16' else store_cache([make(arm,group,observe=observe) for _ in range(30)],pilot.gpu)
    xc=pilot.cross_cache(pipe,pilot.gpu);noise=pilot.chunk_noise(pipe,frames,seed,pilot.gpu);outputs=[]
    operators=OperatorTiming() if profile else None
    if operators:operators.install(cache)
    for chunk in range(frames//3):
        pred=pilot.run_chunk(pipe,cache,xc,cond,noise,chunk,seed);outputs.append(pred)
        if profile:
            event=torch.cuda.Event(enable_timing=True);event.record();events.append(event)
    latent=torch.cat(outputs,1);torch.cuda.synchronize();elapsed=time.monotonic()-started
    peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
    if not torch.isfinite(latent).all():raise RuntimeError('nonfinite primary generation')
    if arm=='BF16':payload=memory_report([c['k'] for c in cache]+[c['v'] for c in cache])
    else:
        per=[state_bytes(c['store']) for c in cache]
        payload={k:sum(v[k] for v in per) for k in per[0]}
    stats=dict(generation_s=elapsed,baseline=baseline,peak=peak,cache=payload,
               latent_frames=frames,pixel_frames=4*frames-3,latent_frames_per_s=frames/elapsed,chunks_per_s=(frames//3)/elapsed)
    if profile:
        times=[events[i].elapsed_time(events[i+1]) for i in range(len(events)-1)]
        stats['chunk_stream_intervals_ms']=times;stats['first_chunk_latent_ready_ms']=times[0]
        stats['operator_profile']=operators.finish()
    return latent,stats


def paths(args):
    if args.kind=='heldout':base=DATA/'fr'/f'p{args.prompt:02d}_s{args.seed}_{args.arm}'
    else:base=DATA/'performance'/f'{args.arm}_g{args.group}_r{args.rep}'
    return base


def generate(args):
    folder=paths(args);folder.mkdir(parents=True,exist_ok=True);source=source_hash()
    done=folder/'sampling.json'
    if done.exists():
        old=diagnostic.read(done)
        if old['source_sha256']!=source or sha(folder/'latent.pt')!=old['latent_file_sha256']:raise RuntimeError('generation resume mismatch')
        return
    if args.kind=='heldout':
        seal=diagnostic.read(ROOT/'src/experiments/mechanism_fr_registration.json')
        if seal['source_sha256']!=source or not seal['development_gates_passed']:raise RuntimeError('held-out registration mismatch')
        texts=(ROOT/'src/experiments/mechanism_prompts_v0.txt').read_text().splitlines()
    else:
        if not diagnostic.read(DATA/'decision.json')['advance']:raise RuntimeError('performance requires GO')
        texts=(ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines()
    other=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(v.strip().isdigit() and int(v.strip())!=os.getpid() for v in other.splitlines()):raise RuntimeError('GPU occupied by another compute process')
    monitor=Telemetry();monitor.start();start=time.monotonic();pipe=pilot.build(args.seed);load_s=time.monotonic()-start
    start=time.monotonic();cond=pilot.encode(pipe,[texts[args.prompt]])[0];text_s=time.monotonic()-start
    if args.warmup:
        warm,_=execute(pipe,cond,args.arm,args.group,args.seed,24)
        del warm;gc.collect();torch.cuda.synchronize()
    diagnostic.report(f'{args.kind.upper()} {args.arm} p{args.prompt} seed{args.seed} repetition{args.rep}')
    stage_start=time.time();latent,meta=execute(pipe,cond,args.arm,args.group,args.seed,126,observe=args.observe,profile=args.profile)
    stage_end=time.time();torch.cuda.synchronize();t=time.monotonic();cpu=latent.cpu();torch.cuda.synchronize();d2h=time.monotonic()-t
    t=time.monotonic();temp=folder/'latent.tmp';torch.save({'latent':cpu},temp);temp.replace(folder/'latent.pt');write_s=time.monotonic()-t
    monitor.stop();diagnostic.save(folder/'telemetry.json',monitor.rows)
    in_stage=[r for r in monitor.rows if stage_start<=r['wall']<=stage_end]
    meta.update(arm=args.arm,group=args.group,seed=args.seed,prompt=args.prompt,kind=args.kind,rep=args.rep,
                source_sha256=source,latent_sha256=diagnostic.original.tensor_hash(cpu),latent_file_sha256=sha(folder/'latent.pt'),
                model_load_s=load_s,text_encode_s=text_s,latent_d2h_s=d2h,artifact_write_s=write_s,
                gpu_sample_peak_mib=max((r['gpu_memory_mib'] for r in in_stage),default=None),
                cpu_phase_sample_max_bytes=max((r['rss_bytes'] for r in in_stage),default=None),
                cpu_lifetime_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                observe=args.observe,profile=args.profile,warmup_chunks=8 if args.warmup else 0,
                generation_started_wall=stage_start,generation_finished_wall=stage_end)
    if source_hash()!=source:raise RuntimeError('follow-up source changed during generation')
    diagnostic.save(done,meta)


def quality(args):
    import pyiqa
    from utils.wan_wrapper import WanVAEWrapper
    from torchvision.io import write_video
    folder=paths(args);source=source_hash();sampling=diagnostic.read(folder/'sampling.json')
    if source!=sampling['source_sha256'] or sha(folder/'latent.pt')!=sampling['latent_file_sha256']:raise RuntimeError('quality source/tensor mismatch')
    if (folder/'quality.json').exists():
        old=diagnostic.read(folder/'quality.json')
        if old['source_sha256']!=source:raise RuntimeError('quality resume mismatch')
        return
    vae=WanVAEWrapper().to(device=pilot.gpu,dtype=torch.bfloat16)
    latent=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True)['latent']
    torch.cuda.synchronize();start=time.monotonic()
    video=(vae.decode_to_pixel(latent.to(pilot.gpu,dtype=torch.bfloat16),use_cache=False)*.5+.5).clamp(0,1)
    torch.cuda.synchronize();decode_s=time.monotonic()-start
    if video.shape[1]!=501 or not torch.isfinite(video).all():raise RuntimeError('raw decode shape/nonfinite failure')
    start=time.monotonic();pixels=(video[0].permute(0,2,3,1)*255).to(torch.uint8).cpu();write_video(str(folder/'video.mp4'),pixels,fps=16)
    mp4_s=time.monotonic()-start;video_ready_wall=time.time()-sampling['generation_started_wall']
    metric=pyiqa.create_metric('musiq',device=pilot.gpu)
    start=time.monotonic();scores=[]
    for index in range(501):
        score=float(metric(video[:,index].float()))
        if not __import__('math').isfinite(score):raise RuntimeError('nonfinite MUSIQ')
        scores.append(score)
        if index%100==0:diagnostic.report(f'RAW QUALITY {args.arm} p{args.prompt} {index+1}/501')
    auxiliary=raw_auxiliary(pixels)
    torch.cuda.synchronize();quality_s=time.monotonic()-start
    boundary=[12*c-3+o for c in range(21,42) for o in range(-2,3)]
    inside=[12*c-3+o for c in range(21,42) for o in range(4,9)]
    value=dict(source_sha256=source,latent_file_sha256=sampling['latent_file_sha256'],per_frame_musiq=scores,
               late_raw_musiq=sum(scores[249:501])/252,boundary_raw_musiq=sum(scores[i] for i in boundary)/105,
               inside_raw_musiq=sum(scores[i] for i in inside)/105,vae_decode_s=decode_s,quality_eval_s=quality_s,mp4_encode_s=mp4_s,
               video_ready_wall_s=video_ready_wall,pixel_frames_per_video_ready_s=501/video_ready_wall,
               video_ready_time_note='observed elapsed from prepared-condition generation start through serial generation/IO/VAE-worker startup/video encoding, before metric model initialization; excludes generator load/text embedding/warmup',
               raw_auxiliary=auxiliary,metric_input='all raw float decoded frames before lossy encoding')
    if args.kind=='development' and args.arm in ('BF16','L','C_once'):
        oldpath=Path.home()/'sfq_data/policy_study'/('reference/p00_s101/bf16_quality.json' if args.arm=='BF16' else f'cases/p00_s101_{"rolling" if args.arm=="L" else "chunk"}_int4/fr_quality.json')
        if args.group==16:
            old=diagnostic.read(oldpath);diff=max(abs(scores[r['pixel_frame']]-r['musiq']) for r in old['per_chunk'])
            if diff>1e-4:raise RuntimeError(f'raw MUSIQ reproducibility failed {diff}')
            value['old_raw_musiq_max_abs_diff']=diff
    if source_hash()!=source:raise RuntimeError('quality source changed')
    diagnostic.save(folder/'quality.json',value)


def main():
    global FAILURE_PATH
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['generate','quality'],required=True)
    ap.add_argument('--kind',choices=['development','heldout'],default='development');ap.add_argument('--arm',required=True)
    ap.add_argument('--group',type=int,default=16);ap.add_argument('--prompt',type=int,default=0);ap.add_argument('--seed',type=int,default=101)
    ap.add_argument('--rep',default='0');ap.add_argument('--warmup',action='store_true');ap.add_argument('--observe',action='store_true');ap.add_argument('--profile',action='store_true')
    args=ap.parse_args();os.chdir(ROOT/'src');torch.set_grad_enabled(False);torch.set_num_threads(4)
    FAILURE_PATH=paths(args)/'last_failure.json'
    if args.task=='generate':generate(args)
    else:quality(args)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if FAILURE_PATH is not None:
            diagnostic.save(FAILURE_PATH,dict(type=type(exc).__name__,error=str(exc),
                 infrastructure=isinstance(exc,(OSError,torch.cuda.OutOfMemoryError)),wall=time.time()))
        raise
