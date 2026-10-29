"""Read-only arithmetic/artifact/optional CPU tensor audit for mixed precision.

No model generation or attention calculation is repeated here. This is same-author
verification, not independent external validation. --tensors reconstructs packed
states and validates fixed groups, masks and actual storage ownership on CPU.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
PROMPTS=(0,4)
LAYERS=(0,14,29)
SEEDS=(9101,9102,9103,9104,9105)
ARMS=('uniform2','uniform3','uniform4','change_high','change_low','magnitude_high')+tuple(f'random_{s}' for s in SEEDS)
FILES=('tools/change_mixed_probe.py','tools/change_mixed_core.py','tools/test_change_mixed_core.py',
       'tools/postwo_round_core.py','tools/postwo_round_probe.py','tools/transition_round_core.py',
       'src/experiments/CHANGE_MIXED_DEVELOPMENT.md')


def require(condition,message):
    if not condition:raise ValueError(message)


def close(actual,expected,message,rtol=1e-11,atol=1e-14):
    require(math.isfinite(actual) and math.isfinite(expected),f'nonfinite {message}')
    require(math.isclose(actual,expected,rel_tol=rtol,abs_tol=atol),f'{message}: {actual} != {expected}')


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def fingerprint():
    h=hashlib.sha256()
    for rel in FILES:h.update(rel.encode());h.update((ROOT/rel).read_bytes())
    return h.hexdigest()


def unique_bytes(states):
    from dataclasses import fields
    seen=set();total=0
    for state in states:
        for field in fields(state):
            value=getattr(state,field.name)
            if hasattr(value,'untyped_storage'):
                storage=value.untyped_storage();key=(str(value.device),storage.data_ptr(),storage.nbytes())
                if key not in seen:seen.add(key);total+=storage.nbytes()
    return total


def parse_rows(path,key_fields,int_fields):
    result={}
    for source in csv.DictReader(path.open()):
        row=dict(source)
        for key,value in list(row.items()):
            if key in int_fields:row[key]=int(value);require(row[key]>=0,f'negative {key}')
            elif key not in (*key_fields,'state_file','state_sha256'):
                row[key]=float(value);require(math.isfinite(row[key]) and row[key]>=0,f'invalid {key}')
        key=tuple(row[k] for k in key_fields)
        require(key not in result,f'duplicate {key}');result[key]=row
    return result


def audit(data,inputs,results,tensors=False):
    d=read(results/'decision.json');source=fingerprint()
    require(d['source_sha256']==source,'current diagnostic source differs')
    rows=parse_rows(results/'rows.csv',('prompt','layer','arm'),('prompt','layer','v_layer_bytes','kv_layer_bytes','mask_bytes'))
    costs=parse_rows(results/'costs.csv',('prompt','arm'),('prompt','v_actual_packed_bytes','kv_actual_packed_bytes','mask_bytes'))
    require(set(rows)==set(itertools.product(PROMPTS,LAYERS,ARMS)) and len(rows)==66,'expected66 complete rows')
    require(set(costs)==set(itertools.product(PROMPTS,ARMS)) and len(costs)==22,'expected22 costs')
    require(d['cuda_pack_bits']==[1,2,3,4],'CUDA bit roundtrip gate incomplete')
    gates={(g['prompt'],g['layer']):g for g in d['gates']}
    require(len(d['gates'])==6 and set(gates)==set(itertools.product(PROMPTS,LAYERS)),'expected6 gates')
    for gate in gates.values():require(all(gate[k] for k in ('v_bitexact','k_rope_bitexact','samebit_kv_identity')),'failed recorded gate')
    expected_inputs={(root,f'p{p:02d}_s101/'+name) for p,l in itertools.product(PROMPTS,LAYERS)
                     for root,name in [('motion_codec',f'c06_l{l:02d}.pt'),('postwo_rounding',f'l{l:02d}_c6.pt'),('postwo_rounding',f'l{l:02d}_c7.pt')]}
    verified=set()
    require(len(d['input_artifacts'])==18,'expected18 inputs')
    for item in d['input_artifacts']:
        key=(item['root'],item['file']);require(key in expected_inputs and key not in verified,'input identity/duplicate')
        path=inputs/item['root']/item['file']
        require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'input SHA/size mismatch')
        verified.add(key)
    require(verified==expected_inputs,'input set incomplete')
    masks={};scores={}
    expected_masks=set(itertools.product(PROMPTS,tuple(a for a in ARMS if not a.startswith('uniform'))))
    require(len(d['masks'])==16,'expected16 mask records')
    for item in d['masks']:
        key=(item['prompt'],item['arm']);require(key in expected_masks and key not in masks,'mask identity/duplicate')
        require(item['high_tokens']==2340 and item['mask_bytes']==585,'mask budget differs')
        require(math.isfinite(item['mask_construction_seconds']) and item['mask_construction_seconds']>=0,'invalid mask time')
        require(sha(data/item['file'])==item['sha256'],'mask SHA mismatch')
        require(sha(data/item['score_file'])==item['score_sha256'],'score SHA mismatch')
        prior=scores.setdefault(item['prompt'],(item['score_file'],item['score_sha256']))
        require(prior==(item['score_file'],item['score_sha256']),'different shared scores within prompt')
        masks[key]=item
    require(set(masks)==expected_masks,'mask set incomplete')
    base_bytes={2:2697216,3:3595776,4:4494336}
    state_files=set();recomputed=[]
    for prompt,arm in itertools.product(PROMPTS,ARMS):
        c=costs[prompt,arm];mixed=not arm.startswith('uniform');per=3595776 if mixed else base_bytes[int(arm[-1])]
        mask_bytes=585 if mixed else 0
        require(c['mask_bytes']==mask_bytes,'cost mask bytes')
        require(c['v_actual_packed_bytes']==3*per+mask_bytes,'shared V payload incorrect')
        require(c['kv_actual_packed_bytes']==6*per+mask_bytes,'shared KV payload incorrect')
        for layer in LAYERS:
            r=rows[prompt,layer,arm]
            require(r['mask_bytes']==mask_bytes and r['v_layer_bytes']==per+mask_bytes,'perlayer V payload incorrect')
            require(r['kv_layer_bytes']==2*per+mask_bytes,'perlayer KV payload incorrect')
            require((r['state_file'],r['state_sha256'])==(c['state_file'],c['state_sha256']),'row/state reference mismatch')
        close(c['encode_kv_seconds_median3_sum'],sum(rows[prompt,l,arm]['encode_kv_seconds_median3'] for l in LAYERS),'encode timing sum')
        close(c['decode_kv_seconds_median3_sum'],sum(rows[prompt,l,arm]['decode_kv_seconds_median3'] for l in LAYERS),'decode timing sum')
        require(c['state_file'] not in state_files,'duplicate state identity')
        require(sha(data/c['state_file'])==c['state_sha256'],'state SHA mismatch');state_files.add(c['state_file'])
    recorded={v['prompt']:v for v in d['decisions']}
    require(len(d['decisions'])==2 and set(recorded)==set(PROMPTS),'decision prompt set')
    for prompt in PROMPTS:
        means={arm:statistics.mean(rows[prompt,l,arm]['eval_d'] for l in LAYERS) for arm in ARMS}
        rr=[means[f'random_{s}'] for s in SEEDS];rand=statistics.mean(rr);high=means['change_high']
        random_gain=1-high/rand;inverse_gain=1-high/means['change_low'];uniform_ratio=high/means['uniform3']
        overhead=costs[prompt,'change_high']['v_actual_packed_bytes']/costs[prompt,'uniform3']['v_actual_packed_bytes']-1
        passed=random_gain>=.05 and inverse_gain>=.05 and uniform_ratio<=1.02 and overhead<=.01
        expected=dict(prompt=prompt,means=means,random_mean=rand,random_min=min(rr),random_max=max(rr),
                      high_vs_random_reduction=random_gain,high_vs_inverse_reduction=inverse_gain,
                      high_over_uniform3=uniform_ratio,v_payload_overhead=overhead,passes=passed)
        saved=recorded[prompt]
        for key,value in expected.items():
            if key=='means':
                require(set(saved[key])==set(ARMS),'mean arm set')
                for arm in ARMS:close(saved[key][arm],value[arm],f'prompt{prompt}/{arm}mean')
            elif type(value) is float:close(saved[key],value,f'prompt{prompt}/{key}')
            else:require(saved[key]==value,f'decision{key}')
        recomputed.append(expected)
    require(d['advance']==all(v['passes'] for v in recomputed),'advance decision differs')
    tensor_checks=None
    if tensors:
        import torch
        torch.set_num_threads(4)
        sys.path.insert(0,str(ROOT/'tools'))
        from change_mixed_core import PackedMixed,pack_bits,unpack_bits
        raw_cache={};mask_cache={}
        for prompt in PROMPTS:
            for layer in LAYERS:
                raw=torch.load(inputs/'motion_codec'/f'p{prompt:02d}_s101/c06_l{layer:02d}.pt',map_location='cpu',weights_only=True)
                post=torch.load(inputs/'postwo_rounding'/f'p{prompt:02d}_s101/l{layer:02d}_c6.pt',map_location='cpu',weights_only=True)
                require(raw['grid']==[3,30,52] and raw['start_frame']==18,'captured geometry/start')
                require(raw['k'].dtype==torch.bfloat16 and raw['v'].dtype==torch.bfloat16,'raw dtype')
                require(tuple(raw['k'].shape)==(1,4680,12,128) and raw['v'].shape==raw['k'].shape,'raw shape')
                require(torch.equal(raw['v'],post['new_value']),'crosscapture V differs')
                require(tuple(post['key'].shape)==(1,32760,12,128) and post['key'].dtype==torch.bfloat16,'post key shape/dtype')
                require(tuple(post['wo'].shape)==(1536,1536) and post['wo'].dtype==torch.bfloat16,'Wo shape/dtype')
                for chunk,call,payload in [(6,4,post),(7,0,torch.load(inputs/'postwo_rounding'/f'p{prompt:02d}_s101/l{layer:02d}_c7.pt',map_location='cpu',weights_only=True))]:
                    require(payload['chunk']==chunk and payload['call']==call,'query capture time')
                    require(tuple(payload['query'].shape)==(1,32,12,128) and payload['query'].dtype==torch.bfloat16,'query shape/dtype')
                    require(torch.equal(payload['query_indices'],torch.linspace(0,4679,32).long()),'query sample indices')
                raw_cache[prompt,layer]=raw
                del post
            score=torch.load(data/scores[prompt][0],map_location='cpu',weights_only=True)
            x=raw_cache[prompt,0]['v'].reshape(1,3,1560,12,128).float()
            change=(x[:,1:]-x[:,:-1]).square().mean((0,1,3,4)).sqrt()
            magnitude=x.square().mean((0,1,3,4)).sqrt();global_rms=x.square().mean().sqrt()
            normalized=change/magnitude.clamp_min(max(float(.1*global_rms),1e-12))
            for key,actual in [('change',change),('magnitude',magnitude),('global_rms',global_rms),('score',normalized)]:
                torch.testing.assert_close(score[key],actual,rtol=3e-6,atol=1e-8)
            high=torch.zeros(1560,dtype=torch.bool);high[torch.argsort(score['score'],descending=True,stable=True)[:780]]=True
            mag=torch.zeros_like(high);mag[torch.argsort(score['magnitude'],descending=True,stable=True)[:780]]=True
            expected={'change_high':high.repeat(3),'change_low':(~high).repeat(3),'magnitude_high':mag.repeat(3)}
            for seed in SEEDS:
                mask=torch.zeros(1560,dtype=torch.bool);mask[torch.randperm(1560,generator=torch.Generator().manual_seed(seed))[:780]]=True
                expected[f'random_{seed}']=mask.repeat(3)
            for arm,mask in expected.items():
                record=masks[prompt,arm];loaded=torch.load(data/record['file'],map_location='cpu',weights_only=True)
                require(torch.equal(loaded['mask'],mask),'mask selection differs')
                require(torch.equal(unpack_bits(loaded['packed_mask'],1,4680).bool(),mask),'stored packed mask differs')
                require(torch.equal(pack_bits(mask,1),loaded['packed_mask']),'packed mask bit order differs')
                mask_cache[prompt,arm]=loaded['packed_mask']
        restored_checks=0
        for prompt,arm in itertools.product(PROMPTS,ARMS):
            serial=torch.load(data/costs[prompt,arm]['state_file'],map_location='cpu',weights_only=True)
            require(set(serial)==set(map(str,LAYERS)),'serialized layer set')
            states=[];vstates=[];shared_masks=[]
            for layer in LAYERS:
                require(set(serial[str(layer)])=={'k','v'},'serialized KV set')
                pair={kind:PackedMixed(**record) for kind,record in serial[str(layer)].items()}
                for kind,state in pair.items():
                    raw=raw_cache[prompt,layer][kind]
                    require(state.shape==(1,4680,12,128) and state.group==16 and state.input_dtype==torch.bfloat16,'state geometry/dtype')
                    data32=torch.cat((raw.float(),torch.zeros(1,8,12,128)),1)
                    expected_max=data32.reshape(1,293,16,12,128).abs().amax(2,keepdim=True).clamp_min(2**-24).half()
                    require(state.group_max.dtype==torch.float16 and torch.equal(state.group_max,expected_max),'original group maxima changed')
                    if arm.startswith('uniform'):
                        bits=int(arm[-1]);require(state.low_bits==bits and state.high_bits is None and state.packed_mask is None and state.high_stream is None,'uniform format')
                        require(state.low_stream.dtype==torch.uint8 and state.low_stream.numel()==4680*12*128*bits//8,'uniform code payload')
                    else:
                        require(state.low_bits==2 and state.high_bits==4 and state.high_tokens==2340,'mixed bit/high counts')
                        require(torch.equal(state.packed_mask,mask_cache[prompt,arm]),'state mask differs')
                        require(state.low_stream.dtype==torch.uint8 and state.low_stream.numel()==898560,'low2 code payload')
                        require(state.high_stream.dtype==torch.uint8 and state.high_stream.numel()==1797120,'high4 code payload')
                        storage=state.packed_mask.untyped_storage();shared_masks.append((storage.data_ptr(),storage.nbytes()))
                    restored=state.restore()
                    require(restored.dtype==torch.float32 and tuple(restored.shape)==(1,4680,12,128) and bool(torch.isfinite(restored).all()),'restored shape/dtype/finite')
                    # CPU float32 norm over7m elements can accumulate visible error.
                    # Reconstruct in FP32, but independently accumulate norms in FP64.
                    value=float((restored.double()-raw.double()).norm()/raw.double().norm().clamp_min(1e-12))
                    close(value,rows[prompt,layer,arm]['v_rel' if kind=='v' else 'k_pre_rel'],'direct reconstruction relL2',rtol=4e-6,atol=1e-8)
                    restored_checks+=1
                states.extend(pair.values());vstates.append(pair['v'])
                require(pair['v'].bytes()['resident']==rows[prompt,layer,arm]['v_layer_bytes'],'tensor V perlayer bytes')
                require(unique_bytes(list(pair.values()))==rows[prompt,layer,arm]['kv_layer_bytes'],'tensor KV perlayer bytes')
            if not arm.startswith('uniform'):require(len(set(shared_masks))==1 and shared_masks[0][1]==585,'mask storage not shared across6states')
            require(unique_bytes(vstates)==costs[prompt,arm]['v_actual_packed_bytes'],'actual unique V payload')
            require(unique_bytes(states)==costs[prompt,arm]['kv_actual_packed_bytes'],'actual unique KV payload')
        tensor_checks=dict(packed_state_files=22,reconstructed_tensors=restored_checks,original_groups_preserved=True,
                           actual_shared_masks=True,score_and_masks_recomputed=True,crosscapture_v_equal=True)
    return dict(passed=True,rows=66,cost_rows=22,masks=16,input_artifacts_verified=18,packed_state_files_verified=22,
                source_sha256=source,advance=d['advance'],recomputed=recomputed,tensor_checks=tensor_checks,
                limits=['same-author verification, not external independent validation',
                        'attention D/J and K/RoPE/same-bit model gates are not independently re-executed by this audit',
                        'primary is V-only incremental FP32 post-Wo teacher-query error, not full KV/FR/video quality',
                        'actual packed payload excludes temporary reconstruction and full inference peak'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path.home()/'sfq_data/change_mixed')
    p.add_argument('--inputs',type=Path,default=Path.home()/'sfq_data')
    p.add_argument('--results',type=Path,default=ROOT/'src/experiments/change_mixed_development')
    p.add_argument('--tensors',action='store_true');a=p.parse_args()
    print(json.dumps(audit(a.data,a.inputs,a.results,a.tensors),ensure_ascii=False,indent=2))
