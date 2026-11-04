"""Read-only B-4 audit. Reconstruct records and decisions, never rerun GPU inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,struct,hashlib
from pathlib import Path
import numpy as np
import torch
from audit_kv_channel_rollout import checked,equal,errors,close,sha
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/experiments/kv_channel_budget_global_baseline'
DATA=Path('/home/acorn0415/sfq_data/kv_channel_budget_global_baseline')
B3=ROOT/'src/experiments/kv_channel_budget_rollout'

def raw8(blob):
    magic,n,h,d,g,b,value=struct.unpack_from('<4s6I',blob)
    assert (magic,n,h,d,g,b)==(b'RTN8',4680,12,128,16,8) and value in (0,1)
    pages=(n+g-1)//g;count=pages*g*h*d
    assert len(blob)==28+count+pages*h*d*2==8100892
    codes=np.frombuffer(blob[28:28+count],dtype=np.int8).copy().reshape(1,pages,g,h,d)
    scales=np.frombuffer(blob[28+count:],dtype='<f2').copy().reshape(1,pages,1,h,d)
    assert np.isfinite(scales).all()
    # Exact stored FP16 metadata is rounded to BF16 by the existing restore method.
    y=(torch.from_numpy(codes).bfloat16()*torch.from_numpy(scales).bfloat16()).reshape(1,-1,h,d)[:,:n]
    assert torch.isfinite(y).all()
    return value

def main():
    torch.set_num_threads(4)
    q=json.loads((OUT/'quality.json').read_text());assert q['status']=='completed' and q['vae_decodes']==4 and q['finished_utc']<1793813220
    for rel,h in q['provenance'].items():assert sha(ROOT/rel)==h
    old=json.loads(checked(q['baseline_quality']).read_text())
    assert old['decision']['advance']
    for item in q['weights'].values():checked(item)
    stats={'model_forwards':0,'identity_pairs':0,'rtn_pairs':0,'key_encode_calls':0,'value_encode_calls':0,'payloads':0,'chunk_scores':0}
    win={}
    for p in (0,4):
        refnorm=None
        for arm in ('Identity','K8V8'):
            folder=DATA/f'p{p:02d}_{arm}'
            s=json.loads((folder/'sampling.json').read_text());assert s==json.loads((OUT/f'p{p:02d}_{arm}_sampling.json').read_text())
            assert s['status']=='completed' and s['provenance']==q['provenance'] and s['finished_utc']<1793813220
            assert s['model_forwards']==210 and len(s['chunks'])==42;stats['model_forwards']+=210
            want=0 if arm=='Identity' else 41
            assert len(s['layer_calls'])==30
            for l,c in enumerate(s['layer_calls']):
                assert c=={'layer':l,'pairs':41,'key_quantizer_calls':want,'value_quantizer_calls':want}
            stats['key_encode_calls']+=want*30;stats['value_encode_calls']+=want*30
            events=[json.loads(line) for line in checked(s['events']).read_text().splitlines()]
            assert len(events)==1230 and {(e['layer'],e['encoded_chunk']) for e in events}=={(l,c) for l in range(30) for c in range(41)}
            size=4680*1536*2 if arm=='Identity' else 8100864
            assert all(e['reader_chunk']==e['encoded_chunk']+1==e['pair_ordinal'] and e['k_bytes']==size and e['v_bytes']==size for e in events)
            stats['identity_pairs' if arm=='Identity' else 'rtn_pairs']+=len(events)
            latent=torch.load(checked(s['latent']),map_location='cpu',weights_only=True)['latent']
            ref=torch.load(checked(s['reference']),map_location='cpu',weights_only=True)['latent']
            if arm=='Identity':equal(latent,ref)
            equal(latent[:,:3],ref[:,:3])
            for c,row in enumerate(s['chunks']):
                assert row['chunk']==c
                expected=list(range(max(0,c-6),c));assert row['actual_segment_starts_per_layer']==[expected]*30
                bytes_per_layer=4680*1536*4+min(6,c)*(4680*1536*4 if arm=='Identity' else 16201728)
                assert row['kv_tensor_bytes_per_layer']==[bytes_per_layer]*30
                for key,val in errors(latent[:,c*3:(c+1)*3],ref[:,c*3:(c+1)*3]).items():close(val,row['latent_error'][key])
                stats['chunk_scores']+=1
            if arm=='K8V8':
                assert s['kv_with_headers_per_layer']==125964624<132588312 and len(s['payloads'])==36
                for item in s['payloads']:
                    file=checked(item);flag=raw8(file.read_bytes())
                    assert file.stem.split('_')[-1]==('V' if flag else 'K');stats['payloads']+=1
            row=next(x for x in q['rows'] if x['prompt']==p and x['arm']==arm)
            assert row['latent']==s['latent'] and row['metric_input']=='raw decoded RGB8 before MP4'
            checked(row['pixels']);checked(row['preview'])
            features=torch.load(checked(row['features']),map_location='cpu',weights_only=True)['features'].double().numpy()
            assert features.shape==(501,768)
            norm=features/np.linalg.norm(features,axis=1,keepdims=True)
            if arm=='Identity':
                refnorm=norm;previous=next(x for x in old['rows'] if x['prompt']==p and x['arm']=='BF16')
                assert row['pixels']==previous['pixels'] and row['reference_gate']['pixels_byte_equal']
                oldf=torch.load(checked(previous['features']),map_location='cpu',weights_only=True)['features'].double().numpy()
                drift=float(np.linalg.norm(features-oldf)/np.linalg.norm(oldf))
                md=float(np.max(np.abs(np.array(row['musiq_per_frame'])-np.array(previous['musiq_per_frame']))))
                assert drift<=1e-5 and md<=1e-4
                close(drift,row['reference_gate']['feature_relative_l2']);close(md,row['reference_gate']['max_musiq_difference'])
            sim=np.clip(np.sum(norm*refnorm,axis=1),-1,1);adj=np.clip(np.sum(norm[1:]*norm[:-1],axis=1),-1,1)
            assert np.allclose(sim,row['dino_reference_per_frame'],rtol=2e-12,atol=1e-14)
            assert np.allclose(adj,row['dino_adjacent'],rtol=2e-12,atol=1e-14)
            musiq=np.asarray(row['musiq_per_frame']);assert musiq.shape==(501,) and np.isfinite(musiq).all()
            w={}
            for name,start in (('whole',0),('late',249)):
                w[name]={'musiq':float(musiq[start:].mean()),'dino_reference':float(sim[start:].mean()),'dino_adjacent':float(adj[max(start-1,0):].mean()),'frames':501-start}
                for key,val in w[name].items():close(val,row['windows'][name][key])
            win[p,arm]=w
    assert stats=={'model_forwards':840,'identity_pairs':2460,'rtn_pairs':2460,'key_encode_calls':2460,'value_encode_calls':2460,'payloads':72,'chunk_scores':168}
    decisions=[]
    for p in (0,4):
        k=win[p,'K8V8'];s=next(x for x in old['rows'] if x['prompt']==p and x['arm']=='S120')['windows']
        dm=k['late']['musiq']-s['late']['musiq'];dd=k['late']['dino_reference']-s['late']['dino_reference']
        da=k['late']['dino_adjacent']-s['late']['dino_adjacent'];dw=k['whole']['musiq']-s['whole']['musiq']
        sufficient=dm>=-1 and dd>=-.005 and da>=-.005 and dw>=-1
        registered=next(x for x in q['decisions'] if x['prompt']==p)
        assert registered['kv_with_headers_per_layer']==125964624 and registered['existing_baseline_sufficient']==sufficient
        for key,val in [('musiq',dm),('dino_reference',dd),('dino_adjacent',da)]:close(val,registered['late_K8V8_minus_S120'][key])
        close(dw,registered['whole_musiq_difference']);decisions.append(sufficient)
    assert all(decisions)==q['close_boost_development']
    print(json.dumps({'status':'passed','gpu_calls':0,**stats,'musiq_frame_scores_reaggregated':2004,'dino_features_reaggregated':2004,
                      'vae_musiq_dino_inference_rerun':False,'decision_recomputed':True,'close_boost_development':all(decisions),'audit_code_sha256':sha(__file__)},indent=2))

if __name__=='__main__':main()
