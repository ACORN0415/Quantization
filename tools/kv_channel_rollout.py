"""B-3: fixed packed K codec in autoregressive generation, then raw RGB metrics."""
import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src'),str(ROOT/'src/scripts')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
import torch.nn.functional as F
import kv_channel_budget_core as codec
from kv_channel_full_attention import sha,save,record,same,free_gpu,error
OUT=ROOT/'src/experiments/kv_channel_budget_rollout'
DATA=Path('/home/acorn0415/sfq_data/kv_channel_budget_rollout')
B2=ROOT/'src/experiments/kv_channel_budget_full_attention'
ARMS=('BF16','U120','U156','S120')
DEADLINE=1793813640.0  # 2026-11-04 17:34 UTC

def limit():
    if time.time()>=DEADLINE:raise TimeoutError('B-3 90-minute limit')

def settings(arm):return (156 if arm=='U156' else 120,'selected' if arm=='S120' else 'uniform')

def provenance():
    return {'script':sha(__file__),'codec':sha(codec.__file__),'registration':sha(OUT/'REGISTRATION_KO.txt')}

def compatibility():
    path=OUT/'compatibility.json'
    if path.exists():raise RuntimeError('Refuse overwrite compatibility')
    r=json.loads((B2/'results.json').read_text());assert r['advance'] and r['status']=='completed'
    rows=[];calls=0
    for case in r['cases']:
        limit();assert sha(case['capture']['path'])==case['capture']['sha256']
        x=torch.load(case['capture']['path'],map_location='cpu',weights_only=True)['k'][:,:28080]
        for arm in ARMS[1:]:
            group,mode=settings(arm);parts=[];sizes=[]
            for i in range(6):
                calls+=1;blob,y=codec.encode(x[:,i*4680:(i+1)*4680],group,mode,0)
                parts.append(y);sizes.append(len(blob))
            old=case['arms'][arm]['payload'];assert sha(old['path'])==old['sha256']
            assert same(torch.cat(parts,1),codec.decode(Path(old['path']).read_bytes()))
            assert sum(sizes)==old['bytes']+140
            rows.append({'prompt':case['prompt'],'layer':case['layer'],'arm':arm,'byte_equal':True,'six_chunk_bytes':sum(sizes)})
    save(path,{'status':'passed','gpu_calls':0,'encode_calls':calls,'cases':rows,'provenance':provenance()})


class ReadController:
    def __init__(self,arm,model,modules,event_path):
        self.arm=arm;self.original=model.attention;self.counts=[0]*30;self.context={};self.stores=[{} for _ in modules]
        self.encodes=[0]*30;self.attention_calls=0;self.max_packed=0;self.max_restored=0;self.events=event_path.open('w');self.handles=[]
        self.packed_totals=[0]*30
        for layer,module in enumerate(modules):
            def pre(module,args,layer=layer):
                count=self.counts[layer];self.counts[layer]+=1
                self.context={'layer':layer,'chunk':count//5,'call':count%5}
            self.handles.append(module.register_forward_pre_hook(pre))

    def attend(self,q,k,v,*args,**kwargs):
        c=self.context;self.context={};assert c and not args and not kwargs
        self.attention_calls+=1
        if self.arm=='BF16' or c['chunk']==0:return self.original(q,k,v)
        layer,chunk=c['layer'],c['chunk'];limit()
        start=max(0,chunk-6);ids=list(range(start,chunk));past=len(ids)*4680
        assert k.shape[1]==past+4680
        store=self.stores[layer]
        for old in list(store):
            if old<start:del store[old]
        if chunk-1 not in store:
            assert c['call']==0
            begin=time.perf_counter();x=k[:,past-4680:past].detach().cpu()
            group,mode=settings(self.arm)
            self.encodes[layer]+=1;blob,y=codec.encode(x,group,mode,0)
            store[chunk-1]={'blob':blob,'restored':y.cuda()}
            event={'layer':layer,'encoded_chunk':chunk-1,'reader_chunk':chunk,'call':c['call'],
                   'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'roundtrip_byte_equal':True,
                   'cpu_encode_restore_transfer_s':time.perf_counter()-begin}
            self.events.write(json.dumps(event)+'\n');self.events.flush()
        assert sorted(store)==ids
        self.packed_totals[layer]=sum(len(z['blob']) for z in store.values())
        total=sum(self.packed_totals);self.max_packed=max(self.max_packed,total)
        self.max_restored=max(self.max_restored,sum(t['restored'].numel()*2 for s in self.stores for t in s.values()))
        restored=torch.cat([store[i]['restored'] for i in ids]+[k[:,past:]],dim=1)
        return self.original(q,restored,v)

    def close(self,folder):
        self.events.close()
        for handle in self.handles:handle.remove()
        examples=[]
        if self.arm!='BF16':
            for layer in (0,14,29):
                for c,entry in self.stores[layer].items():
                    path=folder/f'l{layer:02d}_c{c:02d}.bin';path.write_bytes(entry['blob']);examples.append(record(path))
        return {'layer_attention_calls':self.counts,'attention_calls':self.attention_calls,'layer_encode_calls':self.encodes,
                'encode_calls':sum(self.encodes),'max_cpu_packed_bytes':self.max_packed,'max_gpu_restored_k_bytes':self.max_restored,
                'representative_final_payloads':examples}


def generate(prompt):
    free_gpu();limit();os.chdir(ROOT/'src')
    gate=json.loads((OUT/'compatibility.json').read_text());assert gate['status']=='passed' and gate['provenance']==provenance()
    import p3_tf_fr as pilot
    import wan.modules.causal_model as model
    model_source=pilot.source_hash()
    assert model_source==json.loads((B2/'capture_p00_manifest.json').read_text())['source_sha256']
    pipe=pilot.build(101);text=(ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines()[prompt]
    cond=pilot.encode(pipe,[text])[0]
    modules=[m for m in pipe.generator.modules() if isinstance(m,model.CausalWanSelfAttention)];assert len(modules)==30
    original=model.attention
    ref_path=Path(f'/home/acorn0415/sfq_data/policy_study/reference/p{prompt:02d}_s101/bf16.pt')
    ref=torch.load(ref_path,map_location='cpu',weights_only=True)['latent'][:,:36]
    for arm in ARMS:
        limit();folder=DATA/f'p{prompt:02d}_{arm}'
        if folder.exists():raise RuntimeError('Refuse overwrite run')
        folder.mkdir(parents=True);r={'status':'running','prompt':prompt,'arm':arm,'seed':101,'started_utc':time.time(),
             'provenance':provenance(),'model_source_sha256':model_source,'reference':record(ref_path),'model_forwards':0,'chunks':[]}
        save(folder/'sampling.json',r);control=None;handle=None
        try:
            cache=pilot.bf16_cache(pipe,pilot.gpu);xc=pilot.cross_cache(pipe,pilot.gpu)
            noise=pilot.chunk_noise(pipe,126,101,pilot.gpu);values=[]
            control=ReadController(arm,model,modules,folder/'encode_events.jsonl');model.attention=control.attend
            def count(module,args):
                limit();r['model_forwards']+=1
                if r['model_forwards']>60:raise RuntimeError('Forward cap')
            handle=pipe.generator.register_forward_pre_hook(count)
            torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            for chunk in range(12):
                y=pilot.run_chunk(pipe,cache,xc,cond,noise,chunk,101).cpu()
                expected=ref[:,chunk*3:(chunk+1)*3]
                assert same(expected.to(y.dtype).to(expected.dtype),expected)
                if arm=='BF16' or chunk==0:assert same(y,expected.to(y.dtype)), 'No-intervention byte mismatch'
                assert torch.isfinite(y).all()
                values.append(y);r['chunks'].append({'chunk':chunk,'no_intervention_byte_equal':True if arm=='BF16' or chunk==0 else None,
                    'latent_error':error(y,expected.to(y.dtype))})
                save(folder/'sampling.json',r);print('generate',prompt,arm,chunk+1,'/12',flush=True)
            latent=torch.cat(values,dim=1);torch.save({'latent':latent},folder/'latent.pt')
            r.update(status='completed',latent=record(folder/'latent.pt'),latent_error=error(latent,ref.to(latent.dtype)),generation_s=time.perf_counter()-started)
        except BaseException as exc:
            r.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
        finally:
            model.attention=original
            if handle:handle.remove()
            if control:r.update(control.close(folder));r['events']=record(folder/'encode_events.jsonl')
            r.update(finished_utc=time.time(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            save(folder/'sampling.json',r);save(OUT/f'p{prompt:02d}_{arm}_sampling.json',r)
        assert r['model_forwards']==60 and r['attention_calls']==1800 and all(n==60 for n in r['layer_attention_calls'])
        assert all(n==(0 if arm=='BF16' else 11) for n in r['layer_encode_calls'])
        assert pilot.source_hash()==model_source
        del control,cache,xc,noise,values,y,latent;gc.collect();torch.cuda.empty_cache()


def windows(musiq,features,reference):
    norm=F.normalize(features.double(),dim=1);ref=F.normalize(reference.double(),dim=1)
    similarity=(norm*ref).sum(1).clamp(-1,1)
    adjacent=(norm[1:]*norm[:-1]).sum(1).clamp(-1,1)
    result={}
    for name,start in (('whole',0),('late',69)):
        result[name]={'musiq':float(np.mean(musiq[start:])),'dino_reference':float(similarity[start:].mean()),
                      'dino_adjacent':float(adjacent[max(0,start-1):].mean()),'frames':141-start}
    return result,similarity.tolist(),adjacent.tolist()


def decisions(rows):
    rr={(r['prompt'],r['arm']):r['windows'] for r in rows};details=[]
    simple=[]
    for u in ('U120','U156'):
        if all(rr[p,u]['late']['musiq']>=rr[p,'BF16']['late']['musiq']-1 and rr[p,u]['late']['dino_reference']>=.99 and rr[p,u]['late']['dino_adjacent']>=rr[p,'BF16']['late']['dino_adjacent']-.005 for p in (0,4)):simple.append(u)
    for p in (0,4):
        s=rr[p,'S120'];us=[rr[p,u] for u in ('U120','U156')]
        dm=s['late']['musiq']-max(u['late']['musiq'] for u in us)
        dd=s['late']['dino_reference']-max(u['late']['dino_reference'] for u in us)
        da=s['late']['dino_adjacent']-max(u['late']['dino_adjacent'] for u in us)
        dw=s['whole']['musiq']-max(u['whole']['musiq'] for u in us)
        guard=da>=-.005 and dw>=-1
        details.append({'prompt':p,'late_musiq_gain_vs_best_uniform':dm,'late_dino_reference_gain_vs_best_uniform':dd,
                        'late_dino_adjacent_difference_vs_best_uniform':da,'whole_musiq_difference_vs_best_uniform':dw,
                        'quality_path':guard and dm>=1 and dd>=-.005,'reference_path':guard and dd>=.01 and dm>=-1})
    advance=not simple and (all(d['quality_path'] for d in details) or all(d['reference_path'] for d in details))
    return {'simple_uniform_sufficient':simple,'details':details,'advance':advance}


def quality():
    free_gpu();limit();os.chdir(ROOT/'src')
    from utils.wan_wrapper import WanVAEWrapper
    from mechanism_worker import auxiliary_evaluator
    from torchvision.io import write_video
    import pyiqa
    weights={'vae':ROOT/'src/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',
             'musiq':Path(torch.hub.get_dir())/'pyiqa/musiq_koniq_ckpt-e95806b9.pth',
             'dino':Path(torch.hub.get_dir())/'checkpoints/dino_vitbase16_pretrain.pth',
             'raft_loaded_but_unused':Path(torch.hub.get_dir())/'checkpoints/raft_small_C_T_V2-01064c6d.pth'}
    files={name:record(path) for name,path in weights.items()}
    path=OUT/'quality.json'
    if path.exists():raise RuntimeError('Refuse overwrite quality')
    r={'status':'running','started_utc':time.time(),'provenance':provenance(),'weights':files,'vae_decodes':0,'rows':[]}
    save(path,r)
    try:
        vae=WanVAEWrapper().to(device='cuda',dtype=torch.bfloat16).eval()
        metric=pyiqa.create_metric('musiq',device='cuda');evaluator=auxiliary_evaluator()
        for prompt in (0,4):
            ref_features=None;ref_pixels=None
            for arm in ARMS:
                limit();folder=DATA/f'p{prompt:02d}_{arm}'
                sample=json.loads((folder/'sampling.json').read_text());assert sample['status']=='completed' and sample['provenance']==provenance()
                assert sha(sample['latent']['path'])==sample['latent']['sha256']
                latent=torch.load(sample['latent']['path'],map_location='cpu',weights_only=True)['latent']
                assert latent.shape==(1,36,16,60,104)
                decoded=vae.decode_to_pixel(latent.cuda().bfloat16(),use_cache=False);r['vae_decodes']+=1
                assert decoded.shape==(1,141,3,480,832) and torch.isfinite(decoded).all()
                pixels=((decoded[0]*.5+.5).clamp(0,1)*255).byte().permute(0,2,3,1).contiguous().cpu()
                vae.model.clear_cache();del decoded,latent;torch.cuda.empty_cache()
                torch.save({'pixels':pixels},folder/'pixels.pt')
                musiq=[];features=[]
                for start in range(0,141,4):
                    limit();batch=pixels[start:start+4].permute(0,3,1,2).cuda().float()/255
                    musiq.extend(metric(batch).reshape(-1).cpu().tolist())
                for start in range(0,141,16):
                    limit();features.append(evaluator.features(list(pixels[start:start+16].numpy())))
                features=torch.cat(features);assert len(musiq)==141 and features.shape[0]==141
                assert np.isfinite(musiq).all() and torch.isfinite(features).all()
                if arm=='BF16':ref_features=features;ref_pixels=pixels
                win,sim,adj=windows(musiq,features,ref_features)
                mse=[float((pixels[i].double()-ref_pixels[i].double()).square().mean()/255**2) for i in range(141)]
                torch.save({'features':features},folder/'features.pt')
                write_video(str(folder/'preview.mp4'),pixels,fps=16,video_codec='libx264',options={'crf':'18'})
                row={'prompt':prompt,'arm':arm,'windows':win,'musiq_per_frame':musiq,'dino_reference_per_frame':sim,'dino_adjacent':adj,
                     'pixel_mse_per_frame':mse,'latent':sample['latent'],'pixels':record(folder/'pixels.pt'),
                     'features':record(folder/'features.pt'),'preview':record(folder/'preview.mp4'),'metric_input':'raw decoded RGB8 before MP4'}
                r['rows'].append(row);save(path,r);print('quality',prompt,arm,json.dumps(win['late']),flush=True)
                del pixels,features,batch;gc.collect();torch.cuda.empty_cache()
        assert r['vae_decodes']==8 and len(r['rows'])==8
        r['decision']=decisions(r['rows']);r['status']='completed'
    except BaseException as exc:
        r.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        r.update(finished_utc=time.time(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save(path,r)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['compatibility','generate','quality']);parser.add_argument('--prompt',type=int,choices=[0,4]);args=parser.parse_args()
    torch.set_num_threads(4);torch.set_grad_enabled(False);DATA.mkdir(parents=True,exist_ok=True)
    if args.mode=='generate':
        if args.prompt is None:parser.error('--prompt required')
        generate(args.prompt)
    elif args.mode=='compatibility':compatibility()
    else:quality()
