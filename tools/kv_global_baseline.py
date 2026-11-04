"""B-4: existing RTN K8V8 under the total KV budget, with identity-store controls."""
import argparse,gc,hashlib,json,os,struct,sys,time,traceback,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src'),str(ROOT/'src/scripts')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
import torch.nn.functional as F
from kv_channel_full_attention import sha,save,record,same,free_gpu,error
from kv_channel_rollout import windows
from kvq.quant import Quantizer
from anchor_identity_gate import IdentityQuantizer
OUT=ROOT/'src/experiments/kv_channel_budget_global_baseline'
DATA=Path('/home/acorn0415/sfq_data/kv_channel_budget_global_baseline')
B3=ROOT/'src/experiments/kv_channel_budget_rollout'
OLD=Path('/home/acorn0415/sfq_data/kv_channel_budget_rollout')
DEADLINE=1793813220.  # 17:27 UTC

def limit():
    if time.time()>=DEADLINE:raise TimeoutError('B-4 35-minute limit')

def provenance():
    return {p:sha(ROOT/p) for p in ('tools/kv_global_baseline.py','tools/kv_channel_rollout.py','tools/kv_channel_full_attention.py',
        'src/kvq/quant.py','src/kvq/store.py','src/kvq/lockstep.py','src/scripts/anchor_identity_gate.py',
        'src/experiments/kv_channel_budget_global_baseline/REGISTRATION_KO.txt')}

def state_bytes(s):
    if hasattr(s,'tensor'):return s.tensor.numel()*s.tensor.element_size()
    assert s.bits==8 and s.codes.dtype==torch.int8 and s.scale.dtype==torch.float16 and s.offset is None
    return s.codes.numel()+s.scale.numel()*2

class Observed:
    def __init__(self,arm,layer,ctx,events):
        self.arm,self.layer,self.ctx,self.events=arm,layer,ctx,events
        self.base=IdentityQuantizer() if arm=='Identity' else Quantizer('rtn',8,8,group=16)
        self.pairs=0;self.k_calls=0;self.v_calls=0
    def encode_pair(self,k,v):
        limit();self.pairs+=1;assert self.pairs==self.ctx['chunk']
        if self.arm=='Identity':states=self.base.encode_pair(k,v)
        else:
            self.k_calls+=1;ks=self.base.encode(k,value=False)
            self.v_calls+=1;vs=self.base.encode(v,value=True);states=(ks,vs)
        row={'layer':self.layer,'reader_chunk':self.ctx['chunk'],'encoded_chunk':self.ctx['chunk']-1,
             'pair_ordinal':self.pairs,'kind':'identity_clone' if self.arm=='Identity' else 'rtn8',
             'k_bytes':state_bytes(states[0]),'v_bytes':state_bytes(states[1])}
        self.events.write(json.dumps(row)+'\n');self.events.flush();return states

def serialize(s,value):
    n,h,d=s.shape[1:];header=struct.pack('<4s6I',b'RTN8',n,h,d,16,8,int(value))
    codes=s.codes.cpu().contiguous().numpy();scale=s.scale.cpu().contiguous().numpy().astype('<f2')
    blob=header+codes.tobytes()+scale.tobytes()
    assert len(blob)==28+state_bytes(s)==8100892
    decoded=np.frombuffer(blob[28:28+codes.size],dtype=np.int8).reshape(codes.shape)
    scales=np.frombuffer(blob[28+codes.size:],dtype='<f2').reshape(scale.shape)
    assert np.array_equal(codes,decoded) and np.array_equal(scale.view(np.uint16),scales.view(np.uint16))
    return blob

def resident(cache):
    rows=[]
    for c in cache:
        store=c['store'];total=0
        for seg in store.segments:
            if seg['state'] is None:total+=sum(t.numel()*t.element_size() for t in seg['raw'])
            else:total+=sum(state_bytes(s) for s in seg['state'])
        if store.pending:total+=sum(t.numel()*t.element_size() for t in store.pending)
        rows.append(total)
    return rows

def generate(prompt):
    free_gpu();limit();os.chdir(ROOT/'src')
    import p3_tf_fr as pilot
    model_source=pilot.source_hash();pipe=pilot.build(101)
    text=(ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines()[prompt]
    cond=pilot.encode(pipe,[text])[0]
    reference=OLD/f'p{prompt:02d}_BF16/latent.pt'
    old_sampling=json.loads((B3/f'p{prompt:02d}_BF16_sampling.json').read_text())
    assert sha(reference)==old_sampling['latent']['sha256'] and model_source==old_sampling['model_source_sha256']
    ref=torch.load(reference,map_location='cpu',weights_only=True)['latent']
    for arm in ('Identity','K8V8'):
        limit();folder=DATA/f'p{prompt:02d}_{arm}'
        if folder.exists():raise RuntimeError('Refuse overwrite run')
        folder.mkdir(parents=True);ctx={'chunk':0};events=(folder/'encode_events.jsonl').open('w')
        encoders=[Observed(arm,l,ctx,events) for l in range(30)]
        r={'status':'running','prompt':prompt,'arm':arm,'started_utc':time.time(),'model_forwards':0,'chunks':[],
           'reference':record(reference),'provenance':provenance(),'model_source_sha256':model_source}
        save(folder/'sampling.json',r);handle=None
        try:
            torch.cuda.reset_peak_memory_stats()
            cache=pilot.compressed_cache(pipe,None,pilot.gpu)
            for c,e in zip(cache,encoders):c['store'].quantizer=e
            del c,e
            xc=pilot.cross_cache(pipe,pilot.gpu);noise=pilot.chunk_noise(pipe,126,101,pilot.gpu);values=[]
            def count(module,args):
                limit();r['model_forwards']+=1
                if r['model_forwards']>210:raise RuntimeError('Forward limit')
            handle=pipe.generator.register_forward_pre_hook(count)
            for chunk in range(42):
                ctx['chunk']=chunk;y=pilot.run_chunk(pipe,cache,xc,cond,noise,chunk,101).cpu()
                expected=ref[:,chunk*3:(chunk+1)*3]
                if arm=='Identity' or chunk==0:assert same(y,expected),'Identity/first chunk byte mismatch'
                assert torch.isfinite(y).all();values.append(y)
                starts=[[s['start']//4680 for s in c['store'].segments] for c in cache]
                assert all(v==list(range(max(0,chunk-6),chunk)) for v in starts)
                assert all(c['store'].pending_start==chunk*4680 for c in cache)
                r['chunks'].append({'chunk':chunk,'latent_error':error(y,expected),'kv_tensor_bytes_per_layer':resident(cache),
                    'actual_segment_starts_per_layer':starts,
                    'byte_equal_control':True if arm=='Identity' or chunk==0 else None})
                save(folder/'sampling.json',r);print('generate',prompt,arm,chunk+1,'/42',flush=True)
            assert all(e.pairs==41 for e in encoders) and r['model_forwards']==210
            latent=torch.cat(values,1);torch.save({'latent':latent},folder/'latent.pt')
            r['latent']=record(folder/'latent.pt');r['latent_error']=error(latent,ref);r['payloads']=[]
            if arm=='K8V8':
                assert resident(cache)==[125964288]*30
                for layer in (0,14,29):
                    for seg in cache[layer]['store'].segments:
                        for value,s in enumerate(seg['state']):
                            name=f'l{layer:02d}_c{seg["start"]//4680:02d}_{"V" if value else "K"}.bin'
                            path=folder/name;path.write_bytes(serialize(s,value));r['payloads'].append(record(path))
                r['kv_with_headers_per_layer']=125964288+6*2*28
                assert r['kv_with_headers_per_layer']<=132588312
            else:assert resident(cache)==[201277440]*30
            assert pilot.source_hash()==model_source;r['status']='completed'
        except BaseException as exc:r.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
        finally:
            if handle:handle.remove()
            events.close();r['events']=record(folder/'encode_events.jsonl')
            r['layer_calls']=[{'layer':e.layer,'pairs':e.pairs,'key_quantizer_calls':e.k_calls,'value_quantizer_calls':e.v_calls} for e in encoders]
            r.update(finished_utc=time.time(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            save(folder/'sampling.json',r);save(OUT/f'p{prompt:02d}_{arm}_sampling.json',r)
        del cache,xc,noise,values,latent,y;gc.collect();torch.cuda.empty_cache()

def quality():
    free_gpu();limit();os.chdir(ROOT/'src')
    from utils.wan_wrapper import WanVAEWrapper
    from mechanism_worker import auxiliary_evaluator
    from torchvision.io import write_video
    import pyiqa
    prior=json.loads((B3/'quality.json').read_text());assert prior['status']=='completed' and prior['decision']['advance']
    for item in prior['weights'].values():assert sha(item['path'])==item['sha256']
    path=OUT/'quality.json'
    if path.exists():raise RuntimeError('Refuse overwrite evaluation')
    r={'status':'running','started_utc':time.time(),'provenance':provenance(),'baseline_quality':record(B3/'quality.json'),
       'weights':prior['weights'],'vae_decodes':0,'rows':[]};save(path,r)
    try:
        vae=WanVAEWrapper().to(device='cuda',dtype=torch.bfloat16).eval()
        metric=pyiqa.create_metric('musiq',device='cuda');evaluator=auxiliary_evaluator()
        for prompt in (0,4):
            ref_features=None
            bf=next(z for z in prior['rows'] if z['prompt']==prompt and z['arm']=='BF16')
            for arm in ('Identity','K8V8'):
                limit();folder=DATA/f'p{prompt:02d}_{arm}'
                sample=json.loads((folder/'sampling.json').read_text());assert sample['status']=='completed' and sample['provenance']==provenance()
                assert sha(sample['latent']['path'])==sample['latent']['sha256']
                latent=torch.load(sample['latent']['path'],map_location='cpu',weights_only=True)['latent']
                decoded=vae.decode_to_pixel(latent.cuda(),use_cache=False);r['vae_decodes']+=1
                assert decoded.shape==(1,501,3,480,832) and torch.isfinite(decoded).all()
                pixels=((decoded[0]*.5+.5).clamp(0,1)*255).byte().permute(0,2,3,1).contiguous().cpu()
                vae.model.clear_cache();del decoded,latent;torch.cuda.empty_cache()
                if arm=='Identity':
                    assert sha(bf['pixels']['path'])==bf['pixels']['sha256']
                    old=torch.load(bf['pixels']['path'],map_location='cpu',weights_only=True)['pixels']
                    assert same(old,pixels),'Reused reference RGB differs';del old
                    pixel_file=bf['pixels'];preview=bf['preview']
                else:
                    torch.save({'pixels':pixels},folder/'pixels.pt');pixel_file=record(folder/'pixels.pt');preview=None
                musiq=[];features=[]
                for start in range(0,501,4):
                    limit();batch=pixels[start:start+4].permute(0,3,1,2).cuda().float()/255
                    musiq.extend(metric(batch).reshape(-1).cpu().tolist())
                for start in range(0,501,16):
                    limit();features.append(evaluator.features(list(pixels[start:start+16].numpy())))
                features=torch.cat(features);assert len(musiq)==501 and torch.isfinite(features).all() and np.isfinite(musiq).all()
                if arm=='Identity':
                    ref_features=features
                    old_features=torch.load(bf['features']['path'],map_location='cpu',weights_only=True)['features']
                    drift=float((features.double()-old_features.double()).norm()/old_features.double().norm())
                    md=float(np.max(np.abs(np.array(musiq)-np.array(bf['musiq_per_frame']))))
                    assert drift<=1e-5 and md<=1e-4,'Reference evaluator differs'
                    reference_gate={'pixels_byte_equal':True,'feature_relative_l2':drift,'max_musiq_difference':md}
                else:reference_gate=None
                win,sim,adj=windows(musiq,features,ref_features)
                torch.save({'features':features},folder/'features.pt')
                if preview is None:
                    write_video(str(folder/'preview.mp4'),pixels,fps=16,video_codec='libx264',options={'crf':'18'});preview=record(folder/'preview.mp4')
                row={'prompt':prompt,'arm':arm,'windows':win,'musiq_per_frame':musiq,'dino_reference_per_frame':sim,'dino_adjacent':adj,
                     'latent':sample['latent'],'pixels':pixel_file,'features':record(folder/'features.pt'),'preview':preview,
                     'reference_gate':reference_gate,'metric_input':'raw decoded RGB8 before MP4'}
                r['rows'].append(row);save(path,r);print('quality',prompt,arm,json.dumps(win['late']),flush=True)
                del pixels,features,batch;gc.collect();torch.cuda.empty_cache()
        decisions=[]
        for prompt in (0,4):
            k=next(z for z in r['rows'] if z['prompt']==prompt and z['arm']=='K8V8')['windows']
            s=next(z for z in prior['rows'] if z['prompt']==prompt and z['arm']=='S120')['windows']
            differences={metric:k['late'][metric]-s['late'][metric] for metric in ('musiq','dino_reference','dino_adjacent')}
            whole=k['whole']['musiq']-s['whole']['musiq']
            memory=json.loads((DATA/f'p{prompt:02d}_K8V8/sampling.json').read_text())['kv_with_headers_per_layer']
            ok=memory<=132588312 and differences['musiq']>=-1 and differences['dino_reference']>=-.005 and differences['dino_adjacent']>=-.005 and whole>=-1
            decisions.append({'prompt':prompt,'late_K8V8_minus_S120':differences,'whole_musiq_difference':whole,
                              'kv_with_headers_per_layer':memory,'existing_baseline_sufficient':ok})
        assert r['vae_decodes']==4 and len(r['rows'])==4
        r.update(status='completed',decisions=decisions,close_boost_development=all(z['existing_baseline_sufficient'] for z in decisions))
    except BaseException as exc:r.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        r.update(finished_utc=time.time(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);save(path,r)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['generate','quality']);p.add_argument('--prompt',type=int,choices=[0,4]);a=p.parse_args()
    torch.set_num_threads(4);torch.set_grad_enabled(False);DATA.mkdir(parents=True,exist_ok=True)
    if a.mode=='generate':
        if a.prompt is None:p.error('--prompt required')
        generate(a.prompt)
    else:quality()
