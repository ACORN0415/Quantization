"""Gated development runs and fixed four-case group-mechanism diagnostic."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/scripts')]
import torch
import p3_tf_fr as pilot
import policy_study as original
from anchor_identity_gate import IdentityQuantizer
from kvq import Quantizer
from kvq.replay import store_cache
from mechanism_core import make,RawHistory,state_bytes,histories
from test_mechanism_core import run as synthetic_gates

DATA=Path.home()/'sfq_data/mechanism'
PLAN=json.loads((ROOT/'src/experiments/mechanism_diagnostic_plan_v0_3.json').read_text())
ARMS=('L','C_repeat','C_once')


def read(p):return json.loads(Path(p).read_text())


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
    t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)


def fingerprint():
    h=hashlib.sha256(original.source_hash().encode())
    for p in [ROOT/'tools/mechanism_core.py',Path(__file__),ROOT/'tools/test_mechanism_core.py',
              ROOT/'src/experiments/mechanism_diagnostic_plan_v0_3.json',ROOT/'src/experiments/P1_P3_MECHANISM_PLAN_v0_3.md',
              ROOT/'src/experiments/MECHANISM_MEASUREMENT_SPEC.md']:
        h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def report(detail,state='running',**extra):
    v=dict(state=state,detail=detail,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
    save(DATA/'status.json',v)
    (DATA.parent/'WORK_STATUS.md').write_text('MECHANISM: '+state+' '+detail+'\nCLI: bash ~/Quantization/tools/watch_progress.sh\nLog: tail -F ~/sfq_data/mechanism.log\n')
    print(time.strftime('%Y-%m-%d %H:%M:%S'),state,detail,flush=True)


def clean():
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()


def actual_gates(pipe,cond,source):
    synthetic=synthetic_gates();records=[]
    oldbase=Path.home()/'sfq_data/policy_study'
    for arm,bits in [(a,0) for a in ARMS]+[('L',4),('C_once',4)]:
        clean();cache=store_cache([make(arm,16,bits or 4,identity=IdentityQuantizer() if not bits else None)
                                  for _ in range(30)],pilot.gpu)
        xc=pilot.cross_cache(pipe,pilot.gpu);noise=pilot.chunk_noise(pipe,126,101,pilot.gpu);outputs=[]
        oldpath=oldbase/('reference/p00_s101/bf16.pt' if not bits else f'cases/p00_s101_{"rolling" if arm=="L" else "chunk"}_int4/fr.pt')
        expected=torch.load(oldpath,map_location='cpu',weights_only=True)['latent']
        for c in range(42):
            value=pilot.run_chunk(pipe,cache,xc,cond,noise,c,101)
            if not torch.equal(value.cpu(),expected[:,3*c:3*c+3]):raise RuntimeError(f'full-model gate mismatch {arm} bits{bits} chunk{c}')
            outputs.append(value.cpu());report(f'GATE {arm} bits{bits} chunk {c+1}/42')
        path=DATA/'gates'/f'{arm}_bits{bits}.pt';path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'latent':torch.cat(outputs,1)},path)
        records.append(dict(arm=arm,bits=bits,matched_chunks=42,latent_sha256=original.tensor_hash(torch.cat(outputs,1))))
        del cache,xc,noise,expected,outputs,value;clean()
    if fingerprint()!=source:raise RuntimeError('source changed during gates')
    save(DATA/'gates.json',dict(passed=True,source_sha256=source,synthetic=synthetic,full_model=records))


class ReplayLayer:
    def __init__(self,layer,group,rows,mem):
        self.layer=layer;self.group=group;self.rows=rows;self.mem=mem;self.calls=0
        self.raw=RawHistory(None,32760)
        self.stores={a:make(a,group,observe=True) for a in ARMS}

    def __call__(self,k,v,start,*,local=True):
        chunk=self.calls//5;call=self.calls%5;self.calls+=1
        rawk,rawv=self.raw.write(k,v,start,local=local);self.raw.commit(rawk,rawv)
        n=self.raw.local_end-k.shape[1];expected=(rawk[:,:n],rawv[:,:n])
        norms=[x.float().norm().clamp_min(1e-12) for x in expected]
        seen={};counts={}
        for arm,store in self.stores.items():
            q=store.quantizer;q.events.clear()
            def attend(query,ak,av,arm=arm,store=store):
                values=(ak[:,:n],av[:,:n]);errs=[]
                for actual,ref,norm in zip(values,expected,norms):errs.append(float((actual.float()-ref.float()).norm()/norm) if n else 0.)
                seen[arm]=tuple(t.clone() for t in values) if self.group==24 and arm=='L' else None
                if self.group==24 and arm=='C_repeat':
                    if not all(torch.equal(a,b) for a,b in zip(values,seen['L'])):raise RuntimeError(f'g24 read identity failed layer{self.layer} chunk{chunk} call{call}')
                counts[arm]=histories(store).copy()
                if self.group==24 and arm=='C_repeat' and counts['L']!=counts['C_repeat']:
                    raise RuntimeError('aligned retained-token encode count mismatch')
                row=dict(chunk=chunk,call=call,layer=self.layer,group=self.group,arm=arm,
                         history_tokens=n,k_rel=errs[0],v_rel=errs[1],reference_k_norm=float(norms[0]),reference_v_norm=float(norms[1]),
                         history_chunk_starts=list(counts[arm]),history_encode_counts=list(counts[arm].values()))
                self.rows.append(row)
            store.attend(k,k,v,start,attend,local=local)
            self.rows[-1]['encode_events']=list(q.events)
            if call==4:
                self.mem.append(dict(chunk=chunk,layer=self.layer,group=self.group,arm=arm,**state_bytes(store)))


def diagnose(pipe,conds,source,block_size):
    if not (DATA/'gates.json').exists() or read(DATA/'gates.json')['source_sha256']!=source:raise RuntimeError('gates missing/mismatched')
    total=4*3*((30+block_size-1)//block_size);completed=0
    texts=(ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines()
    for prompt in PLAN['development_prompt_ids']:
        for seed in PLAN['development_seeds']:
            expected=torch.load(Path.home()/f'sfq_data/policy_study/reference/p{prompt:02d}_s{seed}/bf16.pt',map_location='cpu',weights_only=True)['latent']
            for group in PLAN['groups']:
                for lo in range(0,30,block_size):
                    layers=list(range(lo,min(lo+block_size,30)));name=f'p{prompt:02d}_s{seed}_g{group}_l{lo:02d}'
                    path=DATA/'replay'/f'{name}.json'
                    if path.exists():
                        done=read(path)
                        if done['source_sha256']!=source:raise RuntimeError('cached diagnostic source mismatch')
                        completed+=1;continue
                    clean();torch.cuda.reset_peak_memory_stats();start_time=time.monotonic();rows=[];mem=[]
                    ref=pilot.bf16_cache(pipe,pilot.gpu);observers={layer:ReplayLayer(layer,group,rows,mem) for layer in layers}
                    for layer,observer in observers.items():ref[layer]['write_observer']=observer
                    xc=pilot.cross_cache(pipe,pilot.gpu);noise=pilot.chunk_noise(pipe,126,seed,pilot.gpu)
                    for c in range(42):
                        value=pilot.run_chunk(pipe,ref,xc,conds[prompt],noise,c,seed)
                        if not torch.equal(value.cpu(),expected[:,3*c:3*c+3]):raise RuntimeError('instrumented teacher differs from saved BF16')
                        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>48*2**30:raise RuntimeError('CPU 48GiB budget exceeded')
                        report(f'REPLAY {name} chunk {c+1}/42',completed=completed,total=total,chunk=c+1,
                               eta_hours_rough=(total-completed-1)*(time.monotonic()-start_time)*42/(c+1)/3600)
                    if any(o.calls!=210 for o in observers.values()):raise RuntimeError('write count mismatch')
                    if fingerprint()!=source:raise RuntimeError('source changed during diagnostic')
                    save(path,dict(source_sha256=source,prompt=prompt,seed=seed,group=group,layers=layers,rows=rows,memory=mem,
                         seconds=time.monotonic()-start_time,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(),cpu_lifetime_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                         teacher_bitexact_chunks=42,query_probe='not checked: frozen raw-write observer does not expose query'))
                    completed+=1;del ref,observers,xc,noise,value,rows,mem;clean()
    report('REPLAY COMPLETE; fixed decision analysis next',completed=completed,total=total)


def decision(source):
    files=list((DATA/'replay').glob('*.json'));pool={};memory=[]
    for path in files:
        d=read(path)
        if d['source_sha256']!=source:raise RuntimeError('mixed diagnostic source')
        memory.extend(dict(prompt=d['prompt'],seed=d['seed'],**r) for r in d['memory'])
        for r in d['rows']:
            if r['chunk']<7:continue
            key=(d['prompt'],d['seed'],d['group'],r['arm'])
            pool.setdefault(key,[]).append(r)
    values={}
    for key,rows in pool.items():
        expected={(c,l,a) for c in range(7,42) for l in range(30) for a in range(5)}
        ids=[(r['chunk'],r['layer'],r['call']) for r in rows]
        if len(ids)!=5250 or set(ids)!=expected:raise RuntimeError(f'missing/duplicate summary records {key}')
        ek=sum(r['k_rel'] for r in rows)/5250;ev=sum(r['v_rel'] for r in rows)/5250
        values[key]=dict(k=ek,v=ev,e=(ek+ev)/2)
    cases=[]
    for p in PLAN['development_prompt_ids']:
        for s in PLAN['development_seeds']:
            ds={g:values[p,s,g,'L']['e']-values[p,s,g,'C_repeat']['e'] for g in PLAN['groups']}
            cases.append(dict(prompt=p,seed=s,D16=ds[16],D24=ds[24],D32=ds[32],H=ds[16]-ds[24],
                              passes=ds[16]>1e-6 and ds[16]-ds[24]>1e-6))
    # Publish compact full-layer payload summaries, not every per-call record.
    totals={}
    for row in memory:
        key=(row['prompt'],row['seed'],row['group'],row['arm'],row['chunk'])
        target=totals.setdefault(key,{})
        for name in ('resident_tensor_bytes','hypothetical_packed_bytes','valid_code_bytes','padding_code_bytes','scale_bytes','bf16_pending_bytes','bf16_residual_bytes'):
            target[name]=target.get(name,0)+row[name]
    compact=[]
    for p in PLAN['development_prompt_ids']:
        for s in PLAN['development_seeds']:
            for g in PLAN['groups']:
                for arm in ARMS:
                    last=totals[p,s,g,arm,41]
                    compact.append(dict(prompt=p,seed=s,group=g,arm=arm,post_refresh_chunk41=last,
                         max_resident_tensor_bytes=max(totals[p,s,g,arm,c]['resident_tensor_bytes'] for c in range(42))))
    result=dict(source_sha256=source,advance=all(c['passes'] for c in cases),cases=cases,
                summaries=[dict(prompt=k[0],seed=k[1],group=k[2],arm=k[3],**v) for k,v in sorted(values.items())],
                memory=compact,query_probe='not checked',note='Development decision, not a generalization test')
    save(DATA/'decision.json',result)
    out=ROOT/'src/experiments/mechanism_diagnostic_results';out.mkdir(parents=True,exist_ok=True)
    save(out/'decision.json',result)
    lines=['# 그룹 재편 재생 진단 결과','',f'고정한 네 사례 진행 조건: {"GO" if result["advance"] else "NO-GO"}.',
           '개발 데이터 판단이며 프롬프트 전반의 원인 입증은 아니다.','',
           '| prompt | seed | D16 | D24 | H | 통과 |','|---|---|---:|---:|---:|---|']
    for c in cases:lines.append(f'| {c["prompt"]} | {c["seed"]} | {c["D16"]:.8f} | {c["D24"]:.8f} | {c["H"]:.8f} | {c["passes"]} |')
    lines += ['', 'query 방향의 오차는 미검사다. CPU/GPU 재생 진단의 시간·peak는 진단 포함 비용이며 추론 성능이 아니다.',
              'GO면 성능/FR 구현의 별도 게이트와 봉인이 필요하다. NO-GO면 자동 후속 생성 없이 이 계획을 종료한다.']
    (out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['gates','diagnose','decision'],required=True)
    ap.add_argument('--block-size',type=int,default=6);args=ap.parse_args()
    DATA.mkdir(parents=True,exist_ok=True);os.chdir(ROOT/'src');torch.set_grad_enabled(False);torch.set_num_threads(4)
    source=fingerprint()
    if args.mode=='decision':decision(source);return
    if original.source_hash()!=read(ROOT/'src/experiments/policy_study_registration.json')['source_sha256']:
        raise RuntimeError('original frozen source altered')
    report('MODEL loading fixed checkpoint')
    pipe=pilot.build(101);texts=(ROOT/'src/experiments/policy_study_prompts.txt').read_text().splitlines()
    conditions=pilot.encode(pipe,[texts[0],texts[4]]);conds={0:conditions[0],4:conditions[1]}
    if args.mode=='gates':actual_gates(pipe,conds[0],source);report('ALL GATES PASS',state='pass')
    else:diagnose(pipe,conds,source,args.block_size)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        report(f'{type(exc).__name__}: {exc}',state='failed');raise
