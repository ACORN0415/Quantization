"""Acquire fixed inputs, validate the real path, then evaluate 12 allocations."""
import argparse
import gc
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from moe_allocation_objectives import (BITS, diagonal_scores, fixed_multiset_candidates,
                                       incremental_scores, gram_scores, lexicographic_choice)
from moe_allocation_runtime import (ROOT, RECORD, Guard, setup, event, write_json,
    memory, digest_tensor, bits_equal, load_block, routes, choose_experts,
    routing_summary, reassemble, build_quantized_candidates, load_versions,
    collect_bank, native_output)


def select_documents(tokenizer):
    from datasets import load_dataset
    start = time.perf_counter()
    dataset = load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',
                           revision='b08601e04326c79dfdd32d625aee71d232d685c3',
                           cache_dir=str(ROOT/'dataset'))
    chosen = []
    seen = set()
    available = {}
    for split,count,seed in [('train',8,20261031),('validation',16,20261032)]:
        articles = []
        current = []
        for line in dataset[split]['text']:
            if re.match(r'^\s*= [^=].*? =\s*$',line):
                if current:
                    articles.append(''.join(current))
                current = [line]
            elif current:
                current.append(line)
        if current:
            articles.append(''.join(current))
        pool = []
        for article_id,text in enumerate(articles):
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if text_hash in seen:
                continue
            tokens = tokenizer(text,add_special_tokens=False,verbose=False)['input_ids']
            if len(tokens)>=512:
                pool.append((article_id,text_hash,tokens))
        available[split] = dict(parsed_articles=len(articles),eligible_articles=len(pool))
        assert len(pool)>=count,(split,len(pool))
        rng = np.random.default_rng(seed)
        for index in rng.permutation(len(pool)):
            article_id,text_hash,tokens = pool[int(index)]
            if text_hash in seen:
                continue
            offset = int(rng.integers(0,len(tokens)-512+1))
            ids = torch.tensor(tokens[offset:offset+512],dtype=torch.int64)
            selected = dict(id=f'{split}_{article_id}',split=split,article_id=article_id,
                            text_sha256=text_hash,token_sha256=digest_tensor(ids),
                            offset=offset,article_tokens=len(tokens),input_ids=ids)
            chosen.append(selected)
            seen.add(text_hash)
            if sum(d['split']==split for d in chosen)==count:
                break
    manifest = dict(status='FROZEN_BEFORE_QUANTIZED_ERRORS',eligible=available,
                    documents=[{k:v for k,v in d.items() if k!='input_ids'} for d in chosen],
                    elapsed_seconds=time.perf_counter()-start)
    write_json(RECORD/'input_manifest.json',manifest)
    torch.save(chosen,ROOT/'tokens.pt')
    event('input_manifest_frozen',documents=len(chosen),eligible=available)
    return chosen, manifest['elapsed_seconds']


class Captured(Exception):
    pass


@torch.inference_mode()
def prepare():
    from transformers import AutoTokenizer, OlmoeForCausalLM
    start = time.perf_counter()
    guard = Guard('prepare')
    setup()
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    selected_documents,data_time = select_documents(tokenizer)
    event('model_load_start')
    load_start = time.perf_counter()
    model,loading = OlmoeForCausalLM.from_pretrained(
        ROOT/'model',local_files_only=True,dtype=torch.bfloat16,device_map='cuda',
        attn_implementation='sdpa',experts_implementation='eager',output_loading_info=True)
    model.eval()
    loading_summary = {k:v for k,v in loading.items() if v}
    write_json(RECORD/'loading_info.json',loading_summary)
    assert not loading_summary,loading_summary
    torch.cuda.synchronize()
    load_time = time.perf_counter()-load_start
    event('model_loaded',seconds=load_time,memory=memory())
    block = model.model.layers[8].mlp
    pending = {}

    def gate_hook(module, args, out):
        pending['gates'] = out[1].detach().cpu()
        pending['indices'] = out[2].detach().cpu()

    def block_hook(module,args,out):
        pending['x'] = args[0].detach().reshape(-1,2048).cpu()
        pending['reference'] = out.detach().reshape(-1,2048).cpu()
        raise Captured()

    hooks = [block.gate.register_forward_hook(gate_hook),block.register_forward_hook(block_hook)]
    captures = []
    capture_start = time.perf_counter()
    for i, document in enumerate(selected_documents):
        pending.clear()
        try:
            model(input_ids=document['input_ids'][None].cuda(),use_cache=False)
        except Captured:
            pass
        else:
            raise AssertionError('Expected capture at the registered block')
        assert set(pending)=={'x','reference','gates','indices'}
        item = dict(id=document['id'],split=document['split'],**pending)
        captures.append(item)
        event('input_captured',document=document['id'],completed=i+1,total=24)
    for hook in hooks:
        hook.remove()
    torch.cuda.synchronize()
    capture_time = time.perf_counter()-capture_start
    torch.save(captures,ROOT/'captures.pt')
    capture_hashes = [{k:digest_tensor(d[k]) for k in ('x','gates','indices','reference')} | {'id':d['id']}
                      for d in captures]
    write_json(RECORD/'capture_hashes.json',capture_hashes)

    # Mandatory full no-intervention identity precedes any quantized candidate.
    identity_start = time.perf_counter()
    identity = []
    for i,d in enumerate(captures):
        reconstructed = reassemble(block,d['x'].cuda(),d['indices'].cuda(),d['gates'].cuda()).cpu()
        same = bits_equal(reconstructed,d['reference'])
        row = dict(id=d['id'],shape=list(reconstructed.shape),dtype=str(reconstructed.dtype),
                   bitwise_equal=same,max_absolute_error=float((reconstructed.float()-d['reference'].float()).abs().max()),
                   reference_sha256=digest_tensor(d['reference']),reconstructed_sha256=digest_tensor(reconstructed))
        identity.append(row)
        write_json(RECORD/'identity.json',dict(status='RUNNING',documents=identity))
        assert same,row
    torch.cuda.synchronize()
    identity_time = time.perf_counter()-identity_start
    write_json(RECORD/'identity.json',dict(status='PASS',documents=identity,elapsed_seconds=identity_time,
                                         atol=0,rtol=0,quantization_enabled=False))
    event('identity_passed',documents=len(identity),seconds=identity_time)
    selection_start = time.perf_counter()
    calibration = [d for d in captures if d['split']=='train']
    held_out = [d for d in captures if d['split']=='validation']
    selected = choose_experts(calibration)
    routing = dict(selected_experts=selected,calibration=routing_summary(calibration,selected),
                   held_out=routing_summary(held_out,selected),
                   selection_bias='Chosen for calibration co-routing; not representative random experts')
    write_json(RECORD/'routing.json',routing)
    selection_time = time.perf_counter()-selection_start
    event('experts_frozen',selected=selected)
    save_start = time.perf_counter()
    torch.save({k:v.detach().cpu() for k,v in block.state_dict().items()},ROOT/'block.pt')
    save_time = time.perf_counter()-save_start
    packing_start = time.perf_counter()
    packing = build_quantized_candidates(block,selected)
    torch.cuda.synchronize()
    packing_time = time.perf_counter()-packing_start
    per_expert = {e:{b:sum(r['payload_bytes'] for r in packing if r['expert']==e and r['bits']==b)
                     for b in BITS} for e in selected}
    # Explicit fixed-size deployment selection metadata: four int32 expert IDs
    # and four uint8 bit widths. Common shape/format scalars are reported apart.
    metadata_bytes = 4*4+4
    candidate_bytes = [sum(per_expert[e][b] for e,b in zip(selected,c))+metadata_bytes
                       for c in fixed_multiset_candidates()]
    assert len(set(candidate_bytes))==1
    assert candidate_bytes[0]==10551296+metadata_bytes
    write_json(RECORD/'packing.json',dict(status='PASS',projections=packing,per_expert_bytes=per_expert,
        selection_metadata_bytes=metadata_bytes,candidate_payload_plus_selection_bytes=candidate_bytes,
        common_format_metadata='Layer int32, group_size int32, 3*(rows,cols) int32: 32 bytes shared by candidates',
        common_format_metadata_bytes=32,
        common_BF16_weights='All remaining experts, router and model weights; unchanged across allocations',
        container_bytes=(ROOT/'packed.pt').stat().st_size,container_scope='Archive of all three candidate precisions, not one deployed allocation',
        elapsed_seconds=packing_time))
    download = json.loads((RECORD/'download_result.json').read_text())
    measured_common = download['elapsed_seconds']+data_time+load_time+capture_time+selection_time+save_time+packing_time
    result = dict(status='PASS',prefix_forwards=24,prefix_last_layer_zero_based=8,whole_model_evaluations=0,
                  selected_experts=selected,elapsed_seconds=time.perf_counter()-start,
                  timing=dict(download=download['elapsed_seconds'],data=data_time,model_load=load_time,
                              capture=capture_time,expert_selection=selection_time,block_save=save_time,
                              packing=packing_time,identity_audit=identity_time),
                  measured_common_seconds=measured_common,memory=memory(),
                  residency='Full BF16 model resident during capture, identity and packing; no whole-dataset error bank',
                  input_hash_file='capture_hashes.json')
    write_json(RECORD/'preparation.json',result)
    event('preparation_complete',memory=memory(),common_seconds=measured_common)
    guard.close()


def direct_scores(bank,candidates):
    scores = []
    for c in candidates:
        # Independent full residual reconstruction, outside B/C timed sections.
        components = [bank[:,e,BITS.index(b)] for e,b in enumerate(c)]
        residual = np.stack(components,axis=0).sum(axis=0)
        scores.append(float(np.square(residual).sum()/len(bank)))
    return np.asarray(scores)


def bootstrap_gain(a,b):
    a,b = np.asarray(a),np.asarray(b)
    if float(a.sum())==0:
        return dict(gain=0.0,ci95=[0.0,0.0],zero_baseline=True)
    rng = np.random.default_rng(20261031)
    samples = rng.integers(0,len(a),size=(10000,len(a)))
    den = a[samples].sum(axis=1)
    values = 1-b[samples].sum(axis=1)/den
    return dict(gain=float(1-b.sum()/a.sum()),ci95=np.quantile(values,[.025,.975]).tolist(),
                resamples=10000,resampling_unit='document')


@torch.inference_mode()
def diagnose():
    guard = Guard('diagnostic')
    setup()
    assert json.loads((RECORD/'preparation.json').read_text())['status']=='PASS'
    block = load_block()
    versions = load_versions()
    documents = torch.load(ROOT/'captures.pt',weights_only=True)
    selected = json.loads((RECORD/'routing.json').read_text())['selected_experts']
    candidates = fixed_multiset_candidates()
    calibration = []
    bank_hashes = {}
    for d in documents:
        if d['split']!='train':
            continue
        bank = collect_bank(block,versions,d,selected)
        bank_hashes[d['id']] = hashlib.sha256(bank.tobytes()).hexdigest()
        a = diagonal_scores(bank,candidates)
        b,_ = incremental_scores(bank,candidates)
        c,_ = gram_scores(bank,candidates)
        reference = direct_scores(bank,candidates)
        np.testing.assert_allclose(b,reference,atol=1e-10,rtol=1e-9)
        np.testing.assert_allclose(c,reference,atol=1e-10,rtol=1e-9)
        row = dict(id=d['id'],A=a.tolist(),B=b.tolist(),C=c.tolist(),reference=reference.tolist(),
                   max_B_reference=float(np.abs(b-reference).max()),max_C_reference=float(np.abs(c-reference).max()))
        calibration.append(row)
        event('calibration_scored',document=d['id'])
        del bank
    means = {key:np.mean([d[key] for d in calibration],axis=0) for key in ('A','B','C','reference')}
    choices = {key:lexicographic_choice(values,candidates)[0] for key,values in means.items()}
    ties = {key:lexicographic_choice(values,candidates)[1] for key,values in means.items()}
    assert choices['B']==choices['C']==choices['reference']
    frozen = dict(candidates=candidates,means={k:v.tolist() for k,v in means.items()},
                  choices=choices,ties=ties,documents=calibration,
                  selection_frozen_before_heldout=True)
    write_json(RECORD/'calibration.json',frozen)
    write_json(RECORD/'bank_hashes.json',bank_hashes)
    event('allocations_frozen',choices=choices,ties=ties)
    heldout = []
    for d in documents:
        if d['split']!='validation':
            continue
        bank = collect_bank(block,versions,d,selected)
        losses = direct_scores(bank,[choices['A'],choices['B']])
        del bank
        native_a = native_output(block,versions,d,selected,choices['A'])
        native_b = native_a if choices['A']==choices['B'] else native_output(block,versions,d,selected,choices['B'])
        ref = d['reference'].double()
        native_a_loss = float((native_a.double()-ref).square().sum()/len(ref))
        native_b_loss = float((native_b.double()-ref).square().sum()/len(ref))
        row = dict(id=d['id'],A=float(losses[0]),B=float(losses[1]),
                   native_A=native_a_loss,native_B=native_b_loss,
                   native_A_sha256=digest_tensor(native_a),native_B_sha256=digest_tensor(native_b),
                   native_outputs_identical=bits_equal(native_a,native_b))
        heldout.append(row)
        write_json(RECORD/'held_out_progress.json',heldout)
        event('heldout_scored',document=d['id'],completed=len(heldout),total=16)
    fp64 = bootstrap_gain([d['A'] for d in heldout],[d['B'] for d in heldout])
    native = bootstrap_gain([d['native_A'] for d in heldout],[d['native_B'] for d in heldout])
    na = float(np.mean([d['native_A'] for d in heldout]))
    nb = float(np.mean([d['native_B'] for d in heldout]))
    native_band = 1e-10+1e-9*max(abs(na),abs(nb))
    native_pass = na-nb>native_band
    quality_pass = choices['A']!=choices['B'] and fp64['gain']>=.05 and fp64['ci95'][0]>0 and native_pass
    result = dict(status='VALID',selected_experts=selected,choices=choices,allocations_differ=choices['A']!=choices['B'],
                  held_out=heldout,fp64=fp64,native=native,native_mean_A=na,native_mean_B=nb,
                  native_tie_band=native_band,native_improvement_pass=native_pass,
                  continue_quality=bool(quality_pass),memory=memory(),
                  scope='One fixed layer; four experts chosen for calibration co-routing; other experts BF16')
    write_json(RECORD/'diagnostic.json',result)
    event('diagnostic_complete',allocations_differ=result['allocations_differ'],fp64=fp64,native=native,
          continue_quality=bool(quality_pass))
    guard.close()


@torch.inference_mode()
def cost_worker(method,run_id):
    started = time.perf_counter()
    guard = Guard(run_id)
    setup()
    stages = [dict(stage='startup',memory=memory(),resident='CUDA context only')]
    load_start = time.perf_counter()
    block = load_block()
    versions = load_versions()
    documents = [d for d in torch.load(ROOT/'captures.pt',weights_only=True) if d['split']=='train']
    selected = json.loads((RECORD/'routing.json').read_text())['selected_experts']
    candidates = fixed_multiset_candidates()
    torch.cuda.synchronize()
    load_time = time.perf_counter()-load_start
    stages.append(dict(stage='weights_and_inputs',memory=memory(),resident='One BF16 MoE block, 12 restored expert candidates, calibration inputs; full model released'))
    totals = np.zeros(12)
    output_seconds = score_seconds = hash_seconds = 0.0
    hashes = {}
    evaluator = incremental_scores if method=='B' else gram_scores
    for d in documents:
        torch.cuda.synchronize()
        start = time.perf_counter()
        bank = collect_bank(block,versions,d,selected)
        torch.cuda.synchronize()
        output_seconds += time.perf_counter()-start
        start = time.perf_counter()
        for offset in range(0,len(bank),64):
            tile = bank[offset:offset+64]
            scores,_ = evaluator(tile,candidates)
            totals += scores*len(tile)
        score_seconds += time.perf_counter()-start
        # Integrity overhead is separately measured and included conservatively
        # in complete-process time for both methods.
        start = time.perf_counter()
        hashes[d['id']] = hashlib.sha256(bank.tobytes()).hexdigest()
        hash_seconds += time.perf_counter()-start
        stages.append(dict(stage=d['id'],memory=memory(),resident='One document error bank plus persistent block/candidates/inputs',bank_bytes=bank.nbytes))
        del bank
    scores = totals/sum(len(d['x']) for d in documents)
    expected = json.loads((RECORD/'calibration.json').read_text())
    np.testing.assert_allclose(scores,expected['means']['B'],atol=1e-10,rtol=1e-9)
    assert lexicographic_choice(scores,candidates)[0]==tuple(expected['choices']['B'])
    assert hashes==json.loads((RECORD/'bank_hashes.json').read_text())
    torch.cuda.synchronize()
    result = dict(status='PASS',method=method,run_id=run_id,pid=__import__('os').getpid(),
                  elapsed_worker_seconds=time.perf_counter()-started,load_seconds=load_time,
                  output_collection_seconds=output_seconds,selection_seconds=score_seconds,
                  integrity_hash_seconds=hash_seconds,scores=scores.tolist(),
                  choice=lexicographic_choice(scores,candidates)[0],bank_hashes=hashes,
                  memory=memory(),residency=stages)
    write_json(RECORD/f'{run_id}.json',result)
    event('cost_worker_complete',run_id=run_id,method=method,seconds=result['elapsed_worker_seconds'],
          collection=output_seconds,selection=score_seconds,memory=result['memory'])
    guard.close()


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase',choices=['prepare','diagnose','cost'])
    p.add_argument('--method',choices=['B','C'])
    p.add_argument('--run-id')
    args = p.parse_args()
    try:
        if args.phase=='prepare': prepare()
        elif args.phase=='diagnose': diagnose()
        else: cost_worker(args.method,args.run_id)
    except Exception as exc:
        event('FAILED',phase=args.phase,error=repr(exc))
        write_json(RECORD/f'{args.run_id or args.phase}_failure.json',dict(status='INVALID',error=repr(exc)))
        raise
