"""Read-only B-3 archive audit. Does not rerun generation, VAE, MUSIQ, or DINO."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from audit_kv_channel_full_attention import unpack
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/experiments/kv_channel_budget_rollout'
DATA=Path('/home/acorn0415/sfq_data/kv_channel_budget_rollout')
ARMS=('BF16','U120','U156','S120')
DEADLINE=1793813640.

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()

def checked(item):
    path=Path(item['path']);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    return path

def equal(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

def errors(a,b):
    a=a.float().numpy().astype(np.float64);b=b.float().numpy().astype(np.float64);e=(a-b)**2
    return {'mse':float(e.mean()),'normalized_squared_error':float(e.sum()/np.sum(b*b)),'max_abs':float(np.sqrt(e.max()))}

def close(a,b):
    assert np.isclose(a,b,rtol=2e-12,atol=1e-14),(a,b)

def main():
    torch.set_num_threads(4)
    q=json.loads((OUT/'quality.json').read_text());assert q['status']=='completed' and q['vae_decodes']==8 and q['finished_utc']<DEADLINE
    expected={'script':sha(ROOT/'tools/kv_channel_rollout.py'),'codec':sha(ROOT/'tools/kv_channel_budget_core.py'),'registration':sha(OUT/'REGISTRATION_KO.txt')}
    assert q['provenance']==expected
    deps=json.loads((OUT/'dependency_preflight.json').read_text())
    for name,h in deps['files'].items():assert sha(ROOT/name)==h
    for item in json.loads((OUT/'generation_provenance.json').read_text())['files']:checked(item)
    helper_record=json.loads((ROOT/'src/experiments/kv_channel_budget_full_attention/results.json').read_text())
    assert sha(ROOT/'tools/kv_channel_full_attention.py')==helper_record['code_sha256']
    for item in q['weights'].values():checked(item)
    compat=json.loads((OUT/'compatibility.json').read_text());assert compat['status']=='passed' and compat['provenance']==expected
    assert compat['encode_calls']==108 and len(compat['cases'])==18 and all(x['byte_equal'] for x in compat['cases'])
    before=json.loads((OUT/'compatibility_v1.json').read_text());assert before['encode_calls']==108 and before['status']=='passed'
    total_forwards=0;total_encodes=0;payloads=0;all_windows={};latent_checks=0;pixel_checks=0
    for p in (0,4):
        ref_features=None;ref_pixels=None
        for arm in ARMS:
            folder=DATA/f'p{p:02d}_{arm}';s=json.loads((folder/'sampling.json').read_text())
            assert s==json.loads((OUT/f'p{p:02d}_{arm}_sampling.json').read_text())
            assert s['status']=='completed' and s['provenance']==expected and s['finished_utc']<DEADLINE
            assert s['model_forwards']==210 and s['attention_calls']==6300 and s['layer_attention_calls']==[210]*30
            wanted=0 if arm=='BF16' else 41
            assert s['layer_encode_calls']==[wanted]*30 and s['encode_calls']==30*wanted
            total_forwards+=s['model_forwards'];total_encodes+=s['encode_calls']
            events=[json.loads(line) for line in checked(s['events']).read_text().splitlines()]
            assert len(events)==s['encode_calls']
            if arm=='BF16':assert not events
            else:
                assert {(e['layer'],e['encoded_chunk']) for e in events}=={(l,c) for l in range(30) for c in range(41)}
                nbytes={'U120':2935324,'U156':2880028,'S120':2928772}[arm]
                assert all(e['reader_chunk']==e['encoded_chunk']+1 and e['call']==0 and e['roundtrip_byte_equal'] and e['bytes']==nbytes for e in events)
                assert s['max_cpu_packed_bytes']==nbytes*6*30
                assert s['max_gpu_restored_k_bytes']==6*4680*12*128*2*30
                by_id={(e['layer'],e['encoded_chunk']):e for e in events}
                assert len(s['representative_final_payloads'])==18
                for item in s['representative_final_payloads']:
                    path=checked(item);tokens=path.stem.split('_');key=(int(tokens[0][1:]),int(tokens[1][1:]))
                    assert key[0] in (0,14,29) and key[1] in range(35,41)
                    assert item['sha256']==by_id[key]['sha256'] and item['bytes']==by_id[key]['bytes']
                    restored=unpack(path.read_bytes());assert restored.shape==(1,4680,12,128) and torch.isfinite(restored).all();payloads+=1
            latent=torch.load(checked(s['latent']),map_location='cpu',weights_only=True)['latent']
            ref=torch.load(checked(s['reference']),map_location='cpu',weights_only=True)['latent']
            assert latent.shape==(1,126,16,60,104) and len(s['chunks'])==42
            equal(ref.to(latent.dtype).to(ref.dtype),ref)
            ref=ref.to(latent.dtype)
            if arm=='BF16':equal(latent,ref)
            equal(latent[:,:3],ref[:,:3])
            for i,row in enumerate(s['chunks']):
                assert row['chunk']==i
                for key,v in errors(latent[:,i*3:(i+1)*3],ref[:,i*3:(i+1)*3]).items():close(v,row['latent_error'][key])
                latent_checks+=1
            for key,v in errors(latent,ref).items():close(v,s['latent_error'][key])
            row=next(r for r in q['rows'] if r['prompt']==p and r['arm']==arm)
            assert row['latent']==s['latent'] and row['metric_input']=='raw decoded RGB8 before MP4'
            checked(row['preview'])
            features=torch.load(checked(row['features']),map_location='cpu',weights_only=True)['features'].double().numpy()
            pixels=torch.load(checked(row['pixels']),map_location='cpu',weights_only=True)['pixels']
            assert features.shape[0]==501 and pixels.shape==(501,480,832,3) and pixels.dtype==torch.uint8
            norm=features/np.linalg.norm(features,axis=1,keepdims=True)
            if arm=='BF16':ref_features=norm;ref_pixels=pixels
            sim=np.clip(np.sum(norm*ref_features,axis=1),-1,1)
            adj=np.clip(np.sum(norm[1:]*norm[:-1],axis=1),-1,1)
            assert np.allclose(sim,row['dino_reference_per_frame'],rtol=2e-12,atol=1e-14)
            assert np.allclose(adj,row['dino_adjacent'],rtol=2e-12,atol=1e-14)
            musiq=np.asarray(row['musiq_per_frame']);assert musiq.shape==(501,) and np.isfinite(musiq).all()
            win={}
            for name,start in (('whole',0),('late',249)):
                win[name]={'musiq':float(musiq[start:].mean()),'dino_reference':float(sim[start:].mean()),
                           'dino_adjacent':float(adj[max(0,start-1):].mean()),'frames':501-start}
                for key,val in win[name].items():close(val,row['windows'][name][key])
            all_windows[p,arm]=win
            for i in range(501):
                x=pixels[i].numpy().astype(np.float64);y=ref_pixels[i].numpy().astype(np.float64)
                close(float(np.mean((x-y)**2)/255**2),row['pixel_mse_per_frame'][i]);pixel_checks+=1
    assert total_forwards==1680 and total_encodes==7380 and payloads==108
    sufficient=[]
    for arm in ('U120','U156'):
        if all(all_windows[p,arm]['late']['musiq']>=all_windows[p,'BF16']['late']['musiq']-1 and all_windows[p,arm]['late']['dino_reference']>=.99 and all_windows[p,arm]['late']['dino_adjacent']>=all_windows[p,'BF16']['late']['dino_adjacent']-.005 for p in (0,4)):sufficient.append(arm)
    paths=[]
    for p in (0,4):
        s=all_windows[p,'S120'];us=[all_windows[p,a] for a in ('U120','U156')]
        dm=s['late']['musiq']-max(u['late']['musiq'] for u in us);dd=s['late']['dino_reference']-max(u['late']['dino_reference'] for u in us)
        da=s['late']['dino_adjacent']-max(u['late']['dino_adjacent'] for u in us);dw=s['whole']['musiq']-max(u['whole']['musiq'] for u in us)
        paths.append((da>=-.005 and dw>=-1 and dm>=1 and dd>=-.005,da>=-.005 and dw>=-1 and dd>=.01 and dm>=-1))
    advance=not sufficient and (all(x[0] for x in paths) or all(x[1] for x in paths))
    assert sufficient==q['decision']['simple_uniform_sufficient'] and advance==q['decision']['advance']
    assert all(a==d['quality_path'] and b==d['reference_path'] for (a,b),d in zip(paths,q['decision']['details']))
    print(json.dumps({'status':'passed','gpu_calls':0,'model_forwards_checked':total_forwards,'encode_events_checked':total_encodes,
                      'compatibility_encodes':216,'payloads_independently_decoded':payloads,'latent_chunk_scores_recomputed':latent_checks,
                      'pixel_mse_frames_recomputed':pixel_checks,'dino_features_reaggregated':4008,'musiq_frame_scores_reaggregated':4008,
                      'musiq_dino_vae_inference_rerun':False,'decision_recomputed':True,'advance':advance,
                      'audit_code_sha256':sha(__file__)},indent=2))

if __name__=='__main__':main()
