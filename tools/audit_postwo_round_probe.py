"""Read-only arithmetic, source, artifact and optional CPU-tensor audit.

This does not re-run model gates, attention objectives or optimization. It is
same-author verification, not external independent validation.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
PROMPTS=(0,4)
LAYERS=(0,14,29)
ARMS=('rtn','full_mean','full_transition','headlocal_transition')
FILES=('tools/postwo_round_probe.py','tools/postwo_round_core.py','tools/test_postwo_round_core.py',
       'tools/transition_round_core.py','tools/postwo_transition_gate.py','src/experiments/POSTWO_ROUNDING_DEVELOPMENT.md')


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def source_hash():
    h=hashlib.sha256()
    for rel in FILES:h.update(rel.encode());h.update((ROOT/rel).read_bytes())
    return h.hexdigest()


def model_source_hash():
    base=ROOT/'src';h=hashlib.sha256()
    for folder in ('kvq','pipeline','wan','utils'):
        for path in sorted((base/folder).rglob('*.py')):h.update(path.read_bytes())
    for rel in ('scripts/p3_tf_fr.py','scripts/anchor_identity_gate.py','experiments/P3_TF_FR_PILOT.md'):
        h.update((base/rel).read_bytes())
    return h.hexdigest()


def close(actual,expected,message,rtol=1e-11,atol=1e-14):
    require(math.isfinite(actual) and math.isfinite(expected),f'nonfinite {message}')
    require(math.isclose(actual,expected,rel_tol=rtol,abs_tol=atol),f'{message}: {actual} != {expected}')


def finite_tree(value,label):
    if isinstance(value,dict):
        for k,v in value.items():finite_tree(v,label+'.'+k)
    elif isinstance(value,list):
        for i,v in enumerate(value):finite_tree(v,f'{label}[{i}]')
    elif type(value) in (int,float):require(math.isfinite(value),f'nonfinite {label}')


def audit(data,results,tensors=False):
    d=read(results/'decision.json');finite_tree(d,'decision')
    source=source_hash();require(d['source_sha256']==source,'current diagnostic source hash differs')
    rows={};raw=list(csv.DictReader((results/'rows.csv').open()))
    expected=set(itertools.product(PROMPTS,LAYERS,ARMS));require(len(raw)==24,'expected24 rows')
    for original in raw:
        r=dict(original);key=(int(r['prompt']),int(r['layer']),r['arm'])
        require(key in expected and key not in rows,'row identity/duplicate')
        for k,v in list(r.items()):
            if k.startswith(('train_','eval_')) or k=='search_seconds':
                r[k]=float(v);require(math.isfinite(r[k]) and r[k]>=0,f'invalid {k}')
            elif k in ('prompt','layer','metric_denominator','changed_codes','resident_bytes','packed_estimate_bytes'):
                r[k]=int(v);require(r[k]>=0,f'negative {k}')
        require(r['metric_denominator']==32*1536,'metric must average32queries*1536outputchannels')
        for split in ('train','eval'):
            close(r[split+'_objective'],r[split+'_mean_error_sq']+r[split+'_transition_error_sq'],split+' D+J',rtol=2e-6)
            close(r[split+'_mean_output_error_rms']**2,r[split+'_mean_error_sq'],split+' mean RMS',rtol=2e-6)
            close(r[split+'_transition_error_rms']**2,r[split+'_transition_error_sq'],split+' jump RMS',rtol=2e-6)
        rows[key]=r
    require(set(rows)==expected,'missing rows')
    captures={};verified=[]
    require(len(d['captures'])==2,'expected2 capture manifests')
    for embedded in d['captures']:
        prompt=embedded['prompt'];require(prompt in PROMPTS and prompt not in captures,'capture IDs')
        recorded=read(data/f'p{prompt:02d}_s101/manifest.json')
        require(recorded==embedded,'embedded capture manifest mismatch')
        require(embedded['seed']==101 and embedded['teacher_bitexact_chunks']==8,'capture gate/seed')
        require(embedded['source_sha256']==source and embedded['model_source_sha256']==model_source_hash(),'capture source/model hash')
        expected_files={f'p{prompt:02d}_s101/l{layer:02d}_c{chunk}.pt':(layer,chunk,4 if chunk==6 else 0)
                        for layer,chunk in itertools.product(LAYERS,(6,7))}
        seen=set();require(len(embedded['artifacts'])==6,'expected6 artifacts/prompt')
        for item in embedded['artifacts']:
            name=item['file'];require(name in expected_files and name not in seen,'capture file identity')
            require((item['layer'],item['chunk'],item['call'])==expected_files[name],'capture call/layer')
            path=data/name;require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'capture SHA/size')
            seen.add(name);verified.append(name)
        require(seen==set(expected_files),'capture completeness');captures[prompt]=embedded
    preparation={(p['prompt'],p['layer']):p for p in d['preparation']}
    require(len(preparation)==6 and set(preparation)==set(itertools.product(PROMPTS,LAYERS)),'preparation identities')
    searches={};activity=[];torch=None
    if tensors:
        import torch
    for prompt,layer in itertools.product(PROMPTS,LAYERS):
        base=rows[prompt,layer,'rtn'];initial=None;capture=None
        if torch is not None:
            capture=torch.load(data/f'p{prompt:02d}_s101/l{layer:02d}_c6.pt',map_location='cpu',weights_only=True)
            future=torch.load(data/f'p{prompt:02d}_s101/l{layer:02d}_c7.pt',map_location='cpu',weights_only=True)
            require(tuple(capture['key'].shape)==(1,32760,12,128) and capture['key'].dtype==torch.bfloat16,'full BF16 key geometry')
            require(tuple(capture['new_value'].shape)==(1,4680,12,128) and capture['new_value'].dtype==torch.bfloat16,'new V geometry')
            require(tuple(capture['wo'].shape)==(1536,1536) and capture['wo'].dtype==torch.bfloat16,'real Wo geometry')
            indices=torch.linspace(0,4679,32).long()
            for sample,chunk,call in ((capture,6,4),(future,7,0)):
                require(sample['chunk']==chunk and sample['call']==call,'selection/evaluation capture split')
                require(torch.equal(sample['query_indices'],indices),'fixed query sampling')
                require(tuple(sample['query'].shape)==(1,32,12,128) and sample['query'].dtype==torch.bfloat16,'actual Q geometry')
                require(bool(torch.isfinite(sample['query']).all()),'nonfinite captured Q')
            for name in ('key','new_value','wo'):require(bool(torch.isfinite(capture[name]).all()),f'nonfinite {name}')
            initial=torch.load(data/base['code_file'],map_location='cpu',weights_only=True)
        for arm in ARMS:
            r=rows[prompt,layer,arm]
            require(r['resident_bytes']==base['resident_bytes'] and r['packed_estimate_bytes']==base['packed_estimate_bytes'],'equal payload across arms')
            require(r['search_file']==f'p{prompt:02d}_l{layer:02d}_{arm}_search.json','search file IDs')
            require(r['code_file']==f'p{prompt:02d}_l{layer:02d}_{arm}_codes.pt','code file IDs')
            require(sha(data/r['code_file'])==r['code_sha256'],'code SHA')
            require(sha(data/r['search_file'])==r['search_sha256'],'search SHA')
            meta=read(data/r['search_file']);finite_tree(meta,'search')
            if arm=='rtn':require(meta=={'candidates':1} and r['changed_codes']==0,'RTN metadata')
            else:
                require(meta['steps']==32 and meta['candidates_evaluated']==33,'fixed candidate budget')
                require(meta['head_local']==(arm=='headlocal_transition'),'headlocal selection mode')
                require(meta['transition_weight']==(0. if arm=='full_mean' else 1.),'lambda changed')
                history=meta['training_objective_history'];require(len(history)==33,'training history length')
                close(history[0],meta['initial']['objective'],'initial history objective')
                require(meta['final']['objective']<=meta['initial']['objective'],'best hardcandidate worse than RTN')
                require(meta['changed_codes']==r['changed_codes'],'changedcodes metadata/row')
                require(meta['accepted_candidate_updates']>=0 and meta['step_size']>0,'optimizer activity metadata')
                close(meta['wo_norm_sq'],preparation[prompt,layer]['wo_norm_sq'],'shared Wo spectral norm',rtol=1e-6)
                for metric in ('mean_error_sq','transition_error_sq'):
                    prefix='train_headlocal_' if arm=='headlocal_transition' else 'train_'
                    close(meta['initial'][metric]/r['metric_denominator'],base[prefix+metric],'shared RTN training initialization',rtol=2e-6)
                    close(meta['final'][metric]/r['metric_denominator'],r[prefix+metric],'stored training objective',rtol=2e-6)
                activity.append(dict(prompt=prompt,layer=layer,arm=arm,changed_codes=r['changed_codes'],
                    training_objective_fraction=meta['final']['objective']/meta['initial']['objective'] if meta['initial']['objective'] else None,
                    accepted_updates=meta['accepted_candidate_updates']))
            searches[prompt,layer,arm]=meta
            if torch is not None:
                state=torch.load(data/r['code_file'],map_location='cpu',weights_only=True)
                require(tuple(state['shape'])==(4680,12,128) and state['bits']==4 and state['group']==16,'state format')
                require(state['codes'].dtype==torch.uint8 and tuple(state['codes'].shape)==(4688,12,128),'uint8 padded code layout')
                require(state['scale'].dtype==torch.float16 and tuple(state['scale'].shape)==(293,1,12,128),'FP16 fixed scale layout')
                require(torch.equal(state['scale'],initial['scale']),'scale changed')
                require(bool(torch.isfinite(state['scale']).all()) and bool((state['scale']>0).all()),'invalid scales')
                require(bool((state['codes']<=14).all()),'symmetric code range')
                require(torch.equal(state['codes'][4680:],initial['codes'][4680:]),'padding changed')
                changed=int((state['codes'][:4680]!=initial['codes'][:4680]).sum())
                require(changed==r['changed_codes'],'actual changedcode count')
                scales=state['scale'][:,0].repeat_interleave(16,0)[:4680].float()
                values=capture['new_value'][0].float();low=(values/scales).floor().clamp(-7,7);high=(values/scales).ceil().clamp(-7,7)
                signed=state['codes'][:4680].float()-7
                require(bool(((signed==low)|(signed==high)).all()),'code not neighboring floor/ceil')
                codes_bytes=state['codes'].numel()*state['codes'].element_size();metadata=state['scale'].numel()*state['scale'].element_size()
                require(r['resident_bytes']==codes_bytes+metadata,'actual tensor byte count')
                require(r['packed_estimate_bytes']==math.ceil(state['codes'].numel()*4/8)+metadata,'hypothetical packed byte count')
        del capture,initial
    recomputed=[];require(len(d['decisions'])==2,'expected2 decisions')
    for prompt in PROMPTS:
        means={arm:{metric:statistics.mean(rows[prompt,l,arm]['eval_'+metric] for l in LAYERS)
                    for metric in ('mean_error_sq','transition_error_sq')} for arm in ARMS}
        base=means['full_mean'];candidate=means['full_transition']
        valid=base['transition_error_sq']>1e-12*max(base['mean_error_sq'],1e-30)
        reduction=1-candidate['transition_error_sq']/base['transition_error_sq'] if valid else None
        damage=candidate['mean_error_sq']/base['mean_error_sq'] if base['mean_error_sq'] else None
        passed=valid and reduction>=.1 and damage is not None and damage<=1.02
        recorded=next(v for v in d['decisions'] if v['prompt']==prompt)
        for arm in ARMS:
            for metric in means[arm]:close(recorded['means'][arm][metric],means[arm][metric],'3layer mean')
        require(recorded['baseline_jump_valid']==valid and recorded['passes']==passed,'fixed gate rule')
        for name,value in (('jump_reduction',reduction),('mean_damage_ratio',damage)):
            if value is None:require(recorded[name] is None,'undefined contrast must remain null')
            else:close(recorded[name],value,name)
        recomputed.append(dict(prompt=prompt,means=means,baseline_jump_valid=valid,jump_reduction=reduction,mean_damage_ratio=damage,passes=passed))
    advance=all(r['passes'] for r in recomputed);require(type(d['advance']) is bool and d['advance']==advance,'bothprompt advance rule')
    status=read(data/'status.json');require(status['state']=='pass' and status['detail']=='COMPLETE '+('GO' if advance else 'NO-GO'),'completion status')
    return dict(passed=True,rows=24,capture_artifacts_verified=len(verified),code_files_verified=24,search_files_verified=24,
        source_sha256=source,advance=advance,recomputed=recomputed,optimizer_activity=activity,tensors_checked=tensors,
        limits=['same-author read-only audit, not external independent validation',
                'model identity, attention probabilities, optimization and reported objectives not independently rerun',
                'new-V-only incremental error; no full old quantization error or FR claims',
                'same-query counterfactual deletion and teacher future-Q are not actual compressed rollout',
                'actual uint8 container and hypothetical packed bytes differ; no deployment efficiency claim'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path.home()/'sfq_data/postwo_rounding')
    p.add_argument('--results',type=Path,default=ROOT/'src/experiments/postwo_rounding_development')
    p.add_argument('--tensors',action='store_true',help='also check captured geometry and persisted states on CPU')
    a=p.parse_args();print(json.dumps(audit(a.data,a.results,a.tensors),ensure_ascii=False,indent=2))
