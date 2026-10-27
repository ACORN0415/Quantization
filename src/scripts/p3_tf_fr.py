"""Preregistered P3 lockstep pilot; model gates precede measurement."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from demo_utils.memory import gpu,DynamicSwapInstaller
from kvq import Quantizer
from kvq.lockstep import bf16_cache,cross_cache,plain_snapshot,compressed_cache,chunk_noise,run_chunk,relative


def source_hash():
    h=hashlib.sha256()
    for folder in ('kvq','pipeline','wan','utils'):
        for p in sorted((ROOT/folder).rglob('*.py')):h.update(p.read_bytes())
    h.update(Path(__file__).read_bytes())
    h.update((ROOT/'experiments/P3_TF_FR_PILOT.md').read_bytes())
    return h.hexdigest()


def build(seed):
    set_seed(seed)
    config=OmegaConf.merge(OmegaConf.load(ROOT/'configs/default_config.yaml'),OmegaConf.load(ROOT/'configs/a1.yaml'))
    pipe=CausalInferencePipeline(config,device=gpu)
    weights=torch.load(ROOT/'checkpoints/self_forcing_dmd.pt',map_location='cpu')
    pipe.generator.load_state_dict(weights['generator_ema']);del weights
    pipe=pipe.to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(pipe.text_encoder,device=gpu)
    pipe.generator.to(gpu)
    return pipe


def encode(pipe,texts):
    conds=[pipe.text_encoder(text_prompts=[s]) for s in texts]
    DynamicSwapInstaller.uninstall_model(pipe.text_encoder)
    pipe.text_encoder.to('cpu');torch.cuda.empty_cache()
    return conds


def trajectory(pipe,cond,frames,seed,bits,folder,expected=None):
    started=time.time();torch.cuda.reset_peak_memory_stats()
    noise=chunk_noise(pipe,frames,seed,gpu)
    ref=bf16_cache(pipe,gpu)
    fr=compressed_cache(pipe,Quantizer('rtn',bits,bits),gpu) if bits else None
    rx,tx,fx=(cross_cache(pipe,gpu) for _ in range(3))
    quantizer=Quantizer('rtn',bits,bits) if bits else None
    outputs={k:[] for k in ('bf16','tf','fr') if k!='fr' or bits}
    rows=[];work=None;chunks=frames//pipe.num_frame_per_block
    for c in range(chunks):
        work=None  # free the previous snapshot before allocating another
        work=compressed_cache(pipe,quantizer,gpu,reference=ref) if bits else plain_snapshot(ref)
        r=run_chunk(pipe,ref,rx,cond,noise,c,seed)
        t=run_chunk(pipe,work,tx,cond,noise,c,seed)
        if not bits and not torch.equal(r,t):raise RuntimeError(f'no-quant TF gate failed at chunk {c}')
        if expected is not None:
            old=torch.load(expected/f'chunk_{c:04d}.pt',map_location='cpu',weights_only=True)['latent']
            if not torch.equal(r.float().cpu(),old):raise RuntimeError(f'pipeline comparison failed at chunk {c}')
        row={'chunk':c,'tf_rel':relative(t,r)}
        outputs['bf16'].append(r.float().cpu());outputs['tf'].append(t.float().cpu())
        if bits:
            f=run_chunk(pipe,fr,fx,cond,noise,c,seed)
            row.update(fr_rel=relative(f,r),fr_vs_tf_rel=relative(f,t))
            row['fr_minus_tf_rel']=row['fr_rel']-row['tf_rel']
            outputs['fr'].append(f.float().cpu())
        rows.append(row)
        if c%7==0 or c+1==chunks:print('CHUNK',c+1,chunks,'bits',bits,row,flush=True)
    meta={'frames':frames,'seed':seed,'bits':bits,'rows':rows,'seconds_sampling':time.time()-started,
          'peak_vram_gb':torch.cuda.max_memory_allocated()/1024**3,'source_sha256':source_hash()}
    folder.mkdir(parents=True,exist_ok=True)
    for name,values in outputs.items():torch.save({'latent':torch.cat(values,1)},folder/f'{name}.pt')
    (folder/'sampling.json').write_text(json.dumps(meta,indent=2))
    return meta


def quality(pipe,folder):
    import pyiqa
    from torchvision.io import write_video
    torch.cuda.empty_cache();pipe.vae.to(gpu)
    metric=pyiqa.create_metric('musiq',device=gpu);result={};cleanup=[]
    for name in ('bf16','tf','fr'):
        path=folder/f'{name}.pt'
        latent=torch.load(path,map_location='cpu',weights_only=True)['latent']
        video=(pipe.vae.decode_to_pixel(latent.to(gpu,dtype=torch.bfloat16),use_cache=False)*0.5+0.5).clamp(0,1)
        scores=[]
        for c in range(latent.shape[1]//3):
            fi=min(video.shape[1]-1,round((c*3+1.5-1)*4+1))
            scores.append({'chunk':c,'pixel_frame':fi,'musiq':float(metric(video[:,fi].float()).item())})
        pixels=(video[0].permute(0,2,3,1)*255).to(torch.uint8).cpu()
        motion=[float((pixels[i].float()-pixels[i-1].float()).abs().mean().item()/255) for i in range(1,len(pixels))]
        write_video(str(folder/f'{name}.mp4'),pixels,fps=16)
        result[name]={'per_chunk':scores,'musiq_mean':sum(s['musiq'] for s in scores)/len(scores),
                      'motion_absdiff_mean':sum(motion)/len(motion),'pixel_frames':len(pixels),
                      'metric_input':'raw decoded frames before codec'}
        cleanup.append({'path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        pipe.vae.model.clear_cache();del video,pixels,latent;torch.cuda.empty_cache()
        print('QUALITY',name,result[name]['musiq_mean'],flush=True)
    (folder/'quality.json').write_text(json.dumps(result,indent=2))
    (folder/'cleanup.json').write_text(json.dumps(cleanup,indent=2))
    for item in cleanup:(folder/item['path']).unlink()
    pipe.vae.to('cpu');del metric;torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['gate','pilot'],required=True)
    ap.add_argument('--out',type=Path,default=Path.home()/'sfq_data/p3_pilot')
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--bits',type=int,choices=[2,4])
    ap.add_argument('--prompt_index',type=int,default=0);args=ap.parse_args()
    torch.set_grad_enabled(False);torch.set_num_threads(4);args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='pilot':
        gate=args.out/'gates.json'
        if not gate.exists() or not json.loads(gate.read_text())['pass']:raise RuntimeError('passed model gates required')
        if json.loads(gate.read_text())['source_sha256']!=source_hash():raise RuntimeError('model gate source fingerprint differs')
        if args.bits is None:raise ValueError('--bits required')
    if args.mode=='pilot':
        folder=args.out/f'p{args.prompt_index}_s{args.seed}_int{args.bits}'
        done=folder/'complete.json'
        if done.exists() and json.loads(done.read_text())['source_sha256']==source_hash():
            print('REUSE COMPLETE',folder,flush=True);return
    pipe=build(args.seed)
    if args.mode=='gate':
        texts=(Path.home()/'sfq_data/verify/prompts2.txt').read_text().splitlines();conds=encode(pipe,texts);records=[]
        for i,cond in enumerate(conds):
            meta=trajectory(pipe,cond,63,args.seed,0,args.out/f'gate_prompt{i}',
                            expected=Path.home()/f'sfq_data/verify/g3_base/latents/prompt{i:03d}')
            records.append({'prompt':i,'chunks':len(meta['rows']),'max_tf_rel':max(r['tf_rel'] for r in meta['rows'])})
        (args.out/'gates.json').write_text(json.dumps({'pass':True,'records':records,'source_sha256':source_hash()},indent=2))
        print('GATES PASS',flush=True)
    else:
        texts=(ROOT/'experiments/p3_pilot_prompts.txt').read_text().splitlines();text=texts[args.prompt_index]
        cond=encode(pipe,[text])[0];folder=args.out/f'p{args.prompt_index}_s{args.seed}_int{args.bits}'
        previous=folder/'sampling.json'
        if previous.exists() and json.loads(previous.read_text())['source_sha256']==source_hash():
            meta=json.loads(previous.read_text());print('REUSE SAMPLING',folder,flush=True)
        else:
            meta=trajectory(pipe,cond,126,args.seed,args.bits,folder)
        quality(pipe,folder)
        meta.update(prompt=text,prompt_index=args.prompt_index,prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
        (folder/'complete.json').write_text(json.dumps(meta,indent=2));print('CASE COMPLETE',folder,flush=True)

if __name__=='__main__':main()
