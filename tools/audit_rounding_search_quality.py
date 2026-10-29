"""Read-only candidate-selection, finite-enumeration and artifact audit.

This does not rerun model/attention/optimization. Optional tensor checks validate
persisted small quadratics and selected states on CPU. Same-author verification.
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
FILES=('tools/rounding_search_quality_probe.py','tools/rounding_search_quality_core.py','tools/test_rounding_search_quality_core.py',
       'tools/postwo_round_core.py','tools/postwo_round_probe.py','tools/transition_round_core.py',
       'src/experiments/ROUNDING_SEARCH_QUALITY_DEVELOPMENT.md')
PROMPTS=(0,4)
LAYERS=(0,14,29)
SPACES=('common_pool','exact16')
SELECTORS=('mean','transition')
ARMS=('full_mean','full_transition','headlocal_transition')


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def source_hash():
    h=hashlib.sha256()
    for rel in FILES:h.update(rel.encode());h.update((ROOT/rel).read_bytes())
    return h.hexdigest()


def close(actual,expected,message,rtol=3e-6,atol=1e-13):
    require(math.isfinite(actual) and math.isfinite(expected),f'nonfinite {message}')
    require(math.isclose(actual,expected,rel_tol=rtol,abs_tol=atol),f'{message}: {actual!r} != {expected!r}')


def finite_tree(value,label):
    if isinstance(value,dict):
        for key,item in value.items():finite_tree(item,label+'.'+key)
    elif isinstance(value,list):
        for index,item in enumerate(value):finite_tree(item,f'{label}[{index}]')
    elif type(value) in (int,float):require(math.isfinite(value),f'nonfinite {label}')


def audit(data,inputs,results,tensors=False):
    import audit_postwo_round_probe as legacy_auditor
    legacy_results=ROOT/'src/experiments/postwo_rounding_development'
    previous=legacy_auditor.audit(inputs,legacy_results,tensors=tensors)
    d=read(results/'decision.json');finite_tree(d,'decision')
    require(d['source_sha256']==source_hash(),'diagnostic source hash mismatch')
    require(d['legacy_decision_sha256']==sha(legacy_results/'decision.json'),'legacy decision hash changed')
    legacy=read(legacy_results/'decision.json')
    expected_captures=[a for manifest in legacy['captures'] for a in manifest['artifacts']]
    require(d['captures']==expected_captures,'capture inputs do not match legacy manifests')
    cells={(v['prompt'],v['layer']):v for v in d['cells']}
    keys=set(itertools.product(PROMPTS,LAYERS))
    require(len(d['cells'])==6 and set(cells)==keys,'cell completeness/duplicates')
    parity={(v['prompt'],v['layer'],v['arm']):v for v in d['trace_parity']}
    require(len(d['trace_parity'])==18 and set(parity)==set(itertools.product(PROMPTS,LAYERS,ARMS)),'18 trace identities')
    require(all(v['bitexact_codes'] is True and v['bitexact_history'] is True for v in parity.values()),'trace parity failed')
    pool={}
    raw=list(csv.DictReader((results/'pool_candidates.csv').open()))
    require(len(raw)==582,'expected582 candidate rows')
    for r in raw:
        p,l,slot=int(r['prompt']),int(r['layer']),int(r['slot']);key=(p,l,slot)
        require((p,l) in keys and 0<=slot<97 and key not in pool,'candidate identity/duplicate')
        r=dict(r,prompt=p,layer=l,slot=slot,iteration=int(r['iteration']))
        for name in ('train_d','train_j','train_objective'):
            r[name]=float(r[name]);require(math.isfinite(r[name]) and r[name]>=0,'candidate metric invalid')
        close(r['train_objective'],r['train_d']+r['train_j'],'pool D+J')
        if slot==0:
            require(r['generator']=='rtn' and r['iteration']==0,'initial candidate identity')
        else:
            arm=ARMS[(slot-1)//32];iteration=(slot-1)%32+1
            require(r['generator']==arm and r['iteration']==iteration,'candidate order/budget')
        require(len(r['code_hash'])==64,'candidate hash shape')
        pool[key]=r
    require(set(pool)==set(itertools.product(PROMPTS,LAYERS,range(97))),'candidate slots missing')
    selected={};raw=list(csv.DictReader((results/'selections.csv').open()))
    expected=set(itertools.product(PROMPTS,LAYERS,SPACES,SELECTORS))
    require(len(raw)==24,'expected24 selected rows')
    for r in raw:
        r=dict(r);key=(int(r['prompt']),int(r['layer']),r['space'],r['selector'])
        require(key in expected and key not in selected,'selection identity/duplicate')
        for name in ('prompt','layer','chosen_index','resident_bytes','packed_estimate_bytes'):r[name]=int(r[name])
        for name in ('train_d','train_j','train_objective','eval_d','eval_j'):
            r[name]=float(r[name]);require(math.isfinite(r[name]) and r[name]>=0,'selected metric invalid')
        close(r['train_objective'],r['train_d']+r['train_j'],'selected D+J')
        if r['space']=='exact16':
            for name in ('quadratic_train_d','quadratic_train_j'):
                r[name]=float(r[name]);require(math.isfinite(r[name]) and r[name]>=0,'quadratic metric invalid')
            require(0<=r['chosen_index']<65536,'exact index range')
        else:
            require(r['quadratic_train_d']=='' and r['quadratic_train_j']=='','common row must not claim quadratic metric')
            candidates=[pool[r['prompt'],r['layer'],i] for i in range(97)]
            objective='train_d' if r['selector']=='mean' else 'train_objective'
            best=min(candidates,key=lambda row:row[objective])
            require(r['chosen_index']==best['slot'],'common candidate not training first-tie optimum')
            require(r['code_hash']==best['code_hash'],'common winner hash mismatch')
            for name in ('train_d','train_j','train_objective'):close(r[name],best[name],'common winner metric')
        path=data/r['code_file']
        require(sha(path)==r['code_sha256'],'selected state SHA changed')
        selected[key]=r
    require(set(selected)==expected,'missing selections')
    torch=None
    if tensors:
        import torch
    exact_checked=[];selected_checked=[];precision=[]
    for p,l in sorted(keys):
        c=cells[p,l];require(c['pool_slots']==97 and c['exact_patterns']==65536,'candidate universe size')
        require(c['unique_pool_codes']==len({pool[p,l,i]['code_hash'] for i in range(97)}),'distinct candidate hash count')
        exact_path=data/c['exact_file'];require(sha(exact_path)==c['exact_sha256'],'exact artifact SHA')
        require(len(c['selected_coordinates'])==16,'selected coordinate count')
        if torch is None:continue
        e=torch.load(exact_path,map_location='cpu',weights_only=True)
        require(tuple(e['d_values'].shape)==(65536,) and tuple(e['j_values'].shape)==(65536,),'all65536 exact values required')
        for name in ('d_values','j_values','linear_d','linear_j','quad_d','quad_j','const_d','const_j'):
            require(e[name].dtype==torch.float64 and bool(torch.isfinite(e[name]).all()),'exact float64 finite values')
        require(tuple(e['linear_d'].shape)==(16,) and tuple(e['quad_d'].shape)==(16,16),'quadratic shape')
        require(tuple(e['coords'].shape)==(16,3) and e['coords'].tolist()==c['selected_coordinates'],'stored coordinate identity')
        flat=e['flat_indices'];require(flat.numel()==16 and torch.unique(flat).numel()==16,'unique exact coordinates')
        expected_flat=(e['coords'][:,0]*12+e['coords'][:,1])*128+e['coords'][:,2]
        require(torch.equal(flat,expected_flat),'flat coordinate ordering')
        require(bool(((e['coords'][:,0]>=0)&(e['coords'][:,0]<4680)).all()),'valid token coordinates')
        require(bool(((e['coords'][:,1]>=0)&(e['coords'][:,1]<12)).all()),'valid head coordinates')
        require(bool(((e['coords'][:,2]>=0)&(e['coords'][:,2]<128)).all()),'valid channel coordinates')
        require(e['metric_denominator']==32*1536,'exact denominator')
        # Re-evaluate every quadratic independently on CPU, bounded batches.
        columns=torch.arange(16)
        for lo in range(0,65536,1024):
            bits=((torch.arange(lo,lo+1024)[:,None]>>columns)&1).double()
            for suffix in ('d','j'):
                expected_values=e['const_'+suffix]+bits@e['linear_'+suffix]+torch.einsum('bi,ij,bj->b',bits,e['quad_'+suffix],bits)
                torch.testing.assert_close(expected_values,e[suffix+'_values'][lo:lo+1024],rtol=1e-11,atol=1e-10)
        close(float(e['d_values'][0])/e['metric_denominator'],c['baseline_quadratic_d'],'RTN quadratic D',rtol=1e-12)
        close(float(e['j_values'][0])/e['metric_denominator'],c['baseline_quadratic_j'],'RTN quadratic J',rtol=1e-12)
        expected_indices={'mean':int(e['d_values'].argmin()),'transition':int((e['d_values']+e['j_values']).argmin())}
        initial_file=next(row['code_file'] for row in csv.DictReader((legacy_results/'rows.csv').open()) if int(row['prompt'])==p and int(row['layer'])==l and row['arm']=='rtn')
        initial=torch.load(inputs/initial_file,map_location='cpu',weights_only=True)
        for space,selector in itertools.product(SPACES,SELECTORS):
            r=selected[p,l,space,selector];state=torch.load(data/r['code_file'],map_location='cpu',weights_only=True)
            require(state['shape']==initial['shape'] and state['bits']==4 and state['group']==16,'state format changed')
            require(state['codes'].dtype==torch.uint8 and torch.equal(state['scale'],initial['scale']),'code dtype or scale changed')
            require(hashlib.sha256(state['codes'].contiguous().numpy().tobytes()).hexdigest()==r['code_hash'],'raw code hash mismatch')
            resident=state['codes'].numel()*state['codes'].element_size()+state['scale'].numel()*state['scale'].element_size()
            packed=math.ceil(state['codes'].numel()/2)+state['scale'].numel()*state['scale'].element_size()
            require(r['resident_bytes']==resident and r['packed_estimate_bytes']==packed,'byte accounting mismatch')
            require(torch.equal(state['codes'][4680:],initial['codes'][4680:]),'padding mutated')
            if space=='exact16':
                require(r['chosen_index']==expected_indices[selector],'exact first-tie optimum mismatch')
                reconstructed=initial['codes'].clone();bits=((r['chosen_index']>>columns)&1).bool();coords=e['coords'][bits]
                if coords.numel():reconstructed[coords[:,0],coords[:,1],coords[:,2]]=e['alternative_codes'][bits]
                require(torch.equal(reconstructed,state['codes']),'selected exact state differs outside chosen16 block')
                close(r['quadratic_train_d'],float(e['d_values'][r['chosen_index']])/e['metric_denominator'],'selected quadratic D',rtol=1e-12)
                close(r['quadratic_train_j'],float(e['j_values'][r['chosen_index']])/e['metric_denominator'],'selected quadratic J',rtol=1e-12)
                precision.append(dict(prompt=p,layer=l,selector=selector,relative_d_difference=abs(r['train_d']-r['quadratic_train_d'])/max(r['quadratic_train_d'],1e-30),relative_j_difference=abs(r['train_j']-r['quadratic_train_j'])/max(r['quadratic_train_j'],1e-30)))
            selected_checked.append([p,l,space,selector])
        # Alternative neighbor is defined by FP64 division, not legacy FP32 division.
        capture=torch.load(inputs/f'p{p:02d}_s101/l{l:02d}_c6.pt',map_location='cpu',weights_only=True)
        coords=e['coords'];scale=initial['scale'][coords[:,0]//16,0,coords[:,1],coords[:,2]].double()
        x=capture['new_value'][0,coords[:,0],coords[:,1],coords[:,2]].double()
        low=(x/scale).floor().clamp(-7,7);high=(x/scale).ceil().clamp(-7,7)
        current=initial['codes'][coords[:,0],coords[:,1],coords[:,2]].double()-7
        require(bool(((current==low)|(current==high)).all()),'RTN not an exact neighbor')
        alternative=torch.where(current==low,high,low).add(7).to(torch.uint8)
        require(torch.equal(alternative,e['alternative_codes']),'alternative FP64 neighboring codes mismatch')
        exact_checked.append([p,l])
    recomputed=[]
    for space,p in itertools.product(SPACES,PROMPTS):
        means={selector:{name:statistics.mean(selected[p,l,space,selector]['eval_'+name] for l in LAYERS) for name in ('d','j')} for selector in SELECTORS}
        b=means['mean'];c=means['transition'];valid=b['j']>1e-12*max(b['d'],1e-30)
        reduction=1-c['j']/b['j'] if valid else None;damage=c['d']/b['d'] if b['d'] else None
        passes=valid and reduction>=.1 and damage is not None and damage<=1.02
        recorded=next(v for v in d['decisions'] if v['space']==space and v['prompt']==p)
        require(recorded['baseline_jump_valid']==valid and recorded['passes']==passes,'decision validity/GO differs')
        for selector,name in itertools.product(SELECTORS,('d','j')):close(recorded['means'][selector][name],means[selector][name],'prompt aggregate',rtol=1e-12)
        for name,value in (('jump_reduction',reduction),('mean_damage_ratio',damage)):
            if value is None:require(recorded[name] is None,'null ratio mismatch')
            else:close(recorded[name],value,name,rtol=1e-12)
        recomputed.append(dict(space=space,prompt=p,jump_reduction=reduction,mean_damage_ratio=damage,passes=passes))
    require(len(d['decisions'])==4,'expected4 decisions')
    advance=any(all(r['passes'] for r in recomputed if r['space']==space) for space in SPACES)
    require(d['advance']==advance,'overall advance decision differs')
    return dict(passed=True,candidate_rows=len(pool),selection_rows=len(selected),trace_parity_records=len(parity),
        exact_cells_verified=len(exact_checked),selected_states_verified=len(selected_checked),
        input_legacy_audit_passed=previous['passed'],source_sha256=d['source_sha256'],advance=advance,
        recomputed=recomputed,fp32_vs_quadratic_precision=precision,
        limits=['same-author read-only hash/arithmetic audit, not external independent validation',
                'model, attention metrics, optimizer histories and training coordinate ranking are not rerun here',
                'quadratic enumeration exactness is restricted to selected16 binary coordinates and FP64 surrogate',
                'existing query/input data are development data; not global optimum or video-quality evidence'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path.home()/'sfq_data/rounding_search_quality')
    p.add_argument('--inputs',type=Path,default=Path.home()/'sfq_data/postwo_rounding')
    p.add_argument('--results',type=Path,default=ROOT/'src/experiments/rounding_search_quality_development')
    p.add_argument('--tensors',action='store_true');a=p.parse_args()
    print(json.dumps(audit(a.data,a.inputs,a.results,a.tensors),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
