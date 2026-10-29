"""Read-only arithmetic and payload audit for the separate dense-KV development probe.

No GPU/model execution. SHA manifests and aggregates are checked; optional --tensors
also reads coefficient/map tensors on CPU. Neither mode independently repeats the
model gates or certifies absence of training leakage beyond source/data provenance.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics

from audit_motion_codec_probe import (ROOT, require, read, sha, numeric, close,
                                      all_finite, diagnostic_source_hash, model_source_hash)

PROMPTS=(0,4)
CHUNKS=(6,7)
LAYERS=(0,14,29)
ARMS=('direct','same_position','affine_same','dense_same','dense_affine')
FILES=('tools/dense_motion_core.py','tools/test_dense_motion_core.py','tools/dense_motion_probe.py',
       'tools/motion_codec_core.py','src/experiments/DENSE_MOTION_DEVELOPMENT.md')


def source_hash():
    h=hashlib.sha256()
    for rel in FILES:
        h.update(rel.encode());h.update((ROOT/rel).read_bytes())
    return h.hexdigest()


def table(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def audit(data,inputs,results,tensors=False):
    d=read(results/'decision.json');all_finite(d,'decision')
    source=source_hash()
    require(d['identity_decode_checks']==96,'expected96 forced identity decoder checks')
    require(d['source_sha256']==source,'current diagnostic source mismatch')
    expected=set(itertools.product(PROMPTS,CHUNKS,(0,1),LAYERS,ARMS))
    rows={}
    for r in table(results/'rows.csv'):
        key=(int(r['prompt']),int(r['chunk']),int(r['pair']),int(r['layer']),r['arm'])
        require(key in expected and key not in rows,f'unexpected/duplicate row {key}')
        for name in ('k_rel','v_rel','k_post_rope_rel','error_mean','encode_seconds_median3','decode_seconds_median3'):
            r[name]=numeric(float(r[name]),f'{key}/{name}',nonnegative=True)
        for name in ('layer_pair_resident_without_shared_map','layer_pair_packed_without_shared_map',
                     'coefficients_bytes','reference_workspace_bytes'):
            r[name]=int(r[name]);require(r[name]>=0,f'negative bytes {key}/{name}')
        close(r['error_mean'],(r['k_rel']+r['v_rel'])/2,'K/V summary')
        require(r['layer_pair_packed_without_shared_map']<=r['layer_pair_resident_without_shared_map'], 'packed exceeds container')
        rows[key]=r
    require(set(rows)==expected and len(rows)==120,'expected120 complete evaluation rows')
    expected_costs=set(itertools.product(PROMPTS,CHUNKS,(0,1),ARMS));costs={}
    for c in table(results/'costs.csv'):
        key=(int(c['prompt']),int(c['chunk']),int(c['pair']),c['arm'])
        require(key in expected_costs and key not in costs,f'duplicate/unexpected cost {key}')
        for name in ('resident_payload_bytes','packed_estimate_bytes'):
            c[name]=int(c[name]);require(c[name]>0,'nonpositive payload')
        for name in ('reference_seconds','search_seconds','encode_seconds','decode_seconds'):
            c[name]=numeric(float(c[name]),f'{key}/{name}',nonnegative=True)
        costs[key]=c
    require(set(costs)==expected_costs and len(costs)==40,'expected40 complete costs')
    expected_maps=set(itertools.product(PROMPTS,CHUNKS,(0,1)));maps={}
    tensor_checks=[]
    torch=None
    geometries={}
    if tensors:
        import torch
    for m in d['maps']:
        key=(m['prompt'],m['chunk'],m['pair'])
        require(key in expected_maps and key not in maps,'map IDs')
        require(m['file']==f'p{key[0]:02d}_c{key[1]}_pair{key[2]}_map.pt','map file identity')
        require(sha(data/m['file'])==m['sha256'],'map file SHA')
        counts=m['shift_counts'];tokens=sum(counts.values())
        require(tokens>0 and m['map_bytes']==2*tokens,'int8 dy/dx map payload')
        nonzero=0
        for shift,count in counts.items():
            dy,dx=map(int,shift.split(','));require(-2<=dy<=2 and -2<=dx<=2,'map radius changed')
            require(type(count) is int and count>0,'invalid map count')
            if dy or dx:nonzero+=count
        require(abs(m['nonzero_fraction']-nonzero/tokens)<=1e-7,'map nonzero fraction')
        if torch is not None:
            x=torch.load(data/m['file'],map_location='cpu',weights_only=True)
            require(x.dtype==torch.int8 and x.ndim==3 and x.shape[-1]==2,'map tensor type/shape')
            require(x.numel()*x.element_size()==m['map_bytes'],'map tensor payload')
            if key[0] not in geometries:
                sample=torch.load(inputs/f'p{key[0]:02d}_s101/c00_l00.pt',map_location='cpu',weights_only=True)
                geometries[key[0]]=tuple(sample['grid'][1:])
            height,width=geometries[key[0]]
            require(x.shape[1]==height*width,'map/geometry spatial token count')
            y=torch.arange(height).repeat_interleave(width)[None]+x[...,0].long()
            z=torch.arange(width).repeat(height)[None]+x[...,1].long()
            require(bool(((y>=0)&(y<height)&(z>=0)&(z<width)).all()),'map tensor out-of-bounds source')
            actual={}
            for dy,dx in x.reshape(-1,2).tolist():
                name=f'{dy},{dx}';actual[name]=actual.get(name,0)+1
            require(actual==counts,'map histogram differs from stored tensor')
            tensor_checks.append(m['file'])
        maps[key]=m
    require(set(maps)==expected_maps and len(maps)==8,'expected8 maps')
    fits={}
    for fit in d['fits']:
        prompt=fit['prompt'];require(prompt in PROMPTS and prompt not in fits,'fit IDs')
        require(fit['training_chunks']==[0,1] and fit['evaluation_chunks']==[6,7],'fit/evaluation split changed')
        require(fit['coefficients_file']==f'p{prompt:02d}_coefficients.pt','coefficient file identity')
        require(sha(data/fit['coefficients_file'])==fit['coefficients_sha256'],'coefficient file SHA')
        require(fit['coefficients_bytes']>0 and fit['fit_seconds']>=0,'fit payload/time')
        if torch is not None:
            coeff=torch.load(data/fit['coefficients_file'],map_location='cpu',weights_only=True)
            require(set(coeff)=={f'{l}_{k}' for l in LAYERS for k in ('k','v')},'coefficient feature set')
            total=0
            for name,item in coeff.items():
                require(item['a'].dtype==item['b'].dtype==torch.float16,'coefficient stored dtype')
                require(item['a'].shape==item['b'].shape,'coefficient shapes')
                require(bool(torch.isfinite(item['a']).all() and torch.isfinite(item['b']).all()),'nonfinite stored coefficient')
                require(item['ridge']==1e-4,'ridge changed')
                # Four fitting frame pairs, same fixed geometry as evaluation maps.
                n=maps[prompt,6,0]['map_bytes']//2
                require(item['training_tokens']==4*n,'fitting token count differs from chunks0/1')
                total+=sum(item[k].numel()*item[k].element_size() for k in ('a','b'))
            require(total==fit['coefficients_bytes'],'stored coefficient byte sum')
            tensor_checks.append(fit['coefficients_file'])
        fits[prompt]=fit
    require(set(fits)==set(PROMPTS) and len(fits)==2,'expected2 coefficient files')
    for prompt,chunk,pair in expected_maps:
        m=maps[prompt,chunk,pair]
        for layer in LAYERS:
            base=rows[prompt,chunk,pair,layer,'direct']
            for arm in ARMS:
                r=rows[prompt,chunk,pair,layer,arm]
                affine=arm in ('affine_same','dense_affine')
                require((r['coefficients_bytes']>0)==affine,'coefficient ownership')
                require(r['reference_workspace_bytes']==base['reference_workspace_bytes'],'reference workspace changed')
                for field in ('layer_pair_resident_without_shared_map','layer_pair_packed_without_shared_map'):
                    require(r[field]==base[field]+r['coefficients_bytes'],'only affine bytes may differ before shared map')
        for arm in ARMS:
            c=costs[prompt,chunk,pair,arm]
            selected=[rows[prompt,chunk,pair,l,arm] for l in LAYERS]
            extra=m['map_bytes'] if arm.startswith('dense_') else 0
            require(c['resident_payload_bytes']==sum(r['layer_pair_resident_without_shared_map'] for r in selected)+extra,
                    '3layer payload must include map exactly once')
            require(c['packed_estimate_bytes']==sum(r['layer_pair_packed_without_shared_map'] for r in selected)+extra,'packed map ownership')
            close(c['reference_seconds'],m['reference_encode_restore_seconds_median3'],'reference stage shared cost')
            close(c['search_seconds'],m['search_seconds_median3'] if arm.startswith('dense_') else 0,'map search charge')
            close(c['encode_seconds'],sum(r['encode_seconds_median3'] for r in selected),'encode stage sum')
            close(c['decode_seconds'],sum(r['decode_seconds_median3'] for r in selected),'decode stage sum')
            if arm in ('affine_same','dense_affine'):
                require(sum(r['coefficients_bytes'] for r in selected)==fits[prompt]['coefficients_bytes'],'all fitted coefficients retained once')
            if arm.startswith('dense_') and m['nonzero_fraction']==0:
                other='same_position' if arm=='dense_same' else 'affine_same'
                for layer in LAYERS:
                    for field in ('k_rel','v_rel','k_post_rope_rel','error_mean'):
                        close(rows[prompt,chunk,pair,layer,arm][field],rows[prompt,chunk,pair,layer,other][field],'identity-map decode metric equivalence')
    recomputed=[]
    require(len(d['decisions'])==2 and {v['prompt'] for v in d['decisions']}==set(PROMPTS),'decision IDs')
    for prompt in PROMPTS:
        means={}
        for arm in ARMS:
            selected=[r['error_mean'] for key,r in rows.items() if key[0]==prompt and key[-1]==arm]
            require(len(selected)==12,'expected12 prompt/arm evaluation cells')
            means[arm]=statistics.mean(selected)
        require(means['affine_same']>0,'zero primary denominator')
        improvement=1-means['dense_affine']/means['affine_same']
        overhead=max(costs[prompt,c,p,'dense_affine']['resident_payload_bytes']/costs[prompt,c,p,'affine_same']['resident_payload_bytes']-1
                     for c,p in itertools.product(CHUNKS,(0,1)))
        passed=improvement>=.05 and overhead<=.01
        recorded=next(v for v in d['decisions'] if v['prompt']==prompt)
        require(set(recorded['means'])==set(ARMS),'summary arms')
        for arm in ARMS:close(recorded['means'][arm],means[arm],'prompt arm mean')
        close(recorded['relative_error_reduction'],improvement,'5percent primary contrast')
        close(recorded['payload_overhead'],overhead,'1percent payload contrast')
        require(type(recorded['passes']) is bool and recorded['passes']==passed,'prompt GO rule')
        recomputed.append(dict(prompt=prompt,means=means,relative_error_reduction=improvement,payload_overhead=overhead,passes=passed))
    advance=all(v['passes'] for v in recomputed)
    require(type(d['advance']) is bool and d['advance']==advance,'overall both-prompt rule')
    verified=[]
    require(len(d['captures'])==2 and {v['prompt'] for v in d['captures']}==set(PROMPTS),'capture manifest IDs')
    for prompt in PROMPTS:
        capture=read(inputs/f'p{prompt:02d}_s101/manifest.json')
        require(capture==next(v for v in d['captures'] if v['prompt']==prompt),'embedded capture manifest changed')
        require(capture['source_sha256']==diagnostic_source_hash() and capture['source_model_sha256']==model_source_hash(),'original capture source/model mismatch')
        require(capture['teacher_bitexact_chunks']==8 and capture['rope_identity_records']==12,'capture gate history')
        expected_files={f'p{prompt:02d}_s101/c{c:02d}_l{l:02d}.pt':(c,l) for c,l in itertools.product((0,1,6,7),LAYERS)}
        seen=set()
        for item in capture['artifacts']:
            name=item['file'];require(name in expected_files and name not in seen,'capture identity/duplicate')
            seen.add(name);require((item['chunk'],item['layer'])==expected_files[name],'capture file metadata')
            path=inputs/name;require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'capture SHA/size')
            verified.append(name)
        require(seen==set(expected_files),'capture completeness')
    status=read(data/'status.json');require(status['state']=='pass','completed state')
    require(status['detail']=='COMPLETE '+('GO' if advance else 'NO-GO'),'status decision')
    return dict(passed=True,rows=120,costs=40,maps=8,coefficient_files=2,capture_artifacts_verified=len(verified),
        source_sha256=source,advance=advance,recomputed=recomputed,tensor_files_checked=tensor_checks,
        limits=['same-author arithmetic/hash audit, not external independent validation',
                'no model gates or codec numerical reconstruction independently rerun',
                'development chunks6/7 previously observed; not held-out confirmation',
                'per-token KV correspondence is not certified physical motion',
                'byte-container payload and hypothetical packing are distinct; inference throughput unmeasured'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path.home()/'sfq_data/dense_motion')
    p.add_argument('--inputs',type=Path,default=Path.home()/'sfq_data/motion_codec')
    p.add_argument('--results',type=Path,default=ROOT/'src/experiments/dense_motion_development')
    p.add_argument('--tensors',action='store_true',help='also validate persisted coefficient/map tensors on CPU')
    a=p.parse_args();print(json.dumps(audit(a.data,a.inputs,a.results,a.tensors),ensure_ascii=False,indent=2))
