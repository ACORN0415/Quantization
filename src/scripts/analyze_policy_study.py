"""Preregistered prompt-paired analysis; refuses incomplete main-study data."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def paired_inference(differences,reps=20000,seed=872341,level=.975):
    x=np.asarray(differences,dtype=np.float64)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('finite prompt differences required')
    observed=float(x.mean())
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=len(x))))
    distribution=(signs*x).mean(axis=1)
    p=float(np.mean(np.abs(distribution)>=abs(observed)-1e-12))
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),size=(reps,len(x)))].mean(axis=1)
    lo,hi=np.quantile(means,[(1-level)/2,1-(1-level)/2])
    return dict(n_prompts=len(x),mean=observed,ci_level=level,ci_low=float(lo),ci_high=float(hi),
                p_exact_sign_flip=p,prompt_differences=x.tolist())


def holm(values):
    p=np.asarray(values,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('invalid p values')
    order=np.argsort(p);out=np.empty_like(p);previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,float(p[index])*(len(p)-rank)));out[index]=previous
    return out.tolist()


def mean(values):
    if not values:raise ValueError('empty endpoint')
    x=float(np.mean(values))
    if not np.isfinite(x):raise ValueError('nonfinite endpoint')
    return x


def read(path):return json.loads(path.read_text())


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def window(case,phase,lo,hi):
    row=dict(case=case['name'],prompt=case['prompt'],seed=case['seed'],policy=case['policy'],bits=case['bits'],
             phase=phase,first_chunk=lo,last_chunk=hi-1)
    for arm in ('bf16','tf','fr','noise'):
        if arm=='noise' and case['sampling']['noise'].get('state')=='nonfinite':
            for key in ('musiq','rel','read_history_k_rel','read_history_v_rel'):row['noise_'+key]=None
            continue
        values=case['quality'][arm]['per_chunk']
        row[arm+'_musiq']=mean([v['musiq'] for v in values if lo<=v['chunk']<hi])
        if arm=='bf16':continue
        sampling=case['sampling'][arm]
        key='tf_rel' if arm=='tf' else 'rel'
        row[arm+'_rel']=mean([v[key] for v in sampling['rows'] if lo<=v['chunk']<hi])
        history=[r for r in sampling['history'] if lo<=r['chunk']<hi and r['read_history_tokens']>0]
        for kind in ('k','v'):
            row[f'{arm}_read_history_{kind}_rel']=mean([r[f'read_history_{kind}_rel'] for r in history])
    for arm in ('tf','fr','noise'):
        row[arm+'_minus_bf16_musiq']=None if row[arm+'_musiq'] is None else row[arm+'_musiq']-row['bf16_musiq']
    row['fr_minus_noise_musiq']=None if row['noise_musiq'] is None else row['fr_musiq']-row['noise_musiq']
    return row


def analyze(data,out):
    plan=read(ROOT/'experiments/policy_study_plan.json')
    seal=read(ROOT/'experiments/policy_study_registration.json')
    queue=read(data/'queue.json')
    expected=plan['prompts']*len(plan['seeds'])*len(plan['bits'])*len(plan['policies'])
    if queue['complete']!=expected or queue['total']!=expected:raise RuntimeError('incomplete registered queue')
    if queue['source_sha256']!=seal['source_sha256']:raise RuntimeError('queue/registration mismatch')
    cases=[];artifacts=[]
    for prompt in range(plan['prompts']):
        for seed in plan['seeds']:
            ref=data/'reference'/f'p{prompt:02d}_s{seed}'
            reference=read(ref/'bf16_sampling.json')
            if file_hash(ref/'bf16.pt')!=reference['latent_file_sha256']:raise RuntimeError('BF16 tensor file altered')
            for bits in plan['bits']:
                for policy in plan['policies']:
                    name=f'p{prompt:02d}_s{seed}_{policy}_int{bits}';folder=data/'cases'/name
                    complete=read(folder/'complete.json')
                    if complete['source_sha256']!=seal['source_sha256']:raise RuntimeError('case source mismatch')
                    sampling={a:read(folder/f'{a}_sampling.json') for a in ('tf','fr','noise')}
                    for arm,meta in sampling.items():
                        if meta['reference_latent_sha256']!=reference['latent_sha256']:raise RuntimeError('reference tensor mismatch')
                        unstable=arm=='noise' and meta.get('state')=='nonfinite'
                        if unstable:
                            if len(meta['rows'])!=meta['numerical_failure']['chunk'] or len(meta['history'])>42*5*30:
                                raise RuntimeError('invalid noise instability record')
                        elif len(meta['rows'])!=42 or len(meta['history'])!=42*5*30:raise RuntimeError('missing calls or chunks')
                        if meta['source_sha256']!=seal['source_sha256']:raise RuntimeError('stage source mismatch')
                        if file_hash(folder/f'{arm}.pt')!=meta['latent_file_sha256']:raise RuntimeError('tensor file altered')
                    quality={a:read((ref if a=='bf16' else folder)/f'{a}_quality.json') for a in ('bf16','tf','fr','noise')}
                    for a,q in quality.items():
                        unstable=a=='noise' and sampling[a].get('state')=='nonfinite'
                        if unstable:
                            if q.get('state')!='nonfinite' or q['per_chunk']:raise RuntimeError('invalid unstable-control quality')
                        elif len(q['per_chunk'])!=42:raise RuntimeError('incomplete quality')
                        if q['source_sha256']!=seal['source_sha256']:raise RuntimeError('quality source mismatch')
                        if q['latent_file_sha256']!=(reference if a=='bf16' else sampling[a])['latent_file_sha256']:
                            raise RuntimeError('quality/latent mismatch')
                    cases.append(dict(name=name,prompt=prompt,seed=seed,bits=bits,policy=policy,
                                      sampling=sampling,quality=quality,complete=complete))
    windows=[window(c,phase,lo,hi) for c in cases for phase,lo,hi in [('early',1,7),('middle',7,21),('late',21,42)]]
    late={(r['prompt'],r['seed'],r['bits'],r['policy']):r for r in windows if r['phase']=='late'}
    primary=[];secondary=[]
    for bits in plan['bits']:
        d=[mean([late[p,s,bits,'chunk']['fr_musiq']-late[p,s,bits,'rolling']['fr_musiq'] for s in plan['seeds']])
           for p in range(plan['prompts'])]
        primary.append(dict(bits=bits,contrast='P3 FR minus P1 FR late MUSIQ',
                            **paired_inference(d,plan['bootstrap_replicates'],plan['analysis_seed'],.975)))
        for policy in plan['policies']:
            failures=sum(late[p,s,bits,policy]['noise_musiq'] is None for p in range(plan['prompts']) for s in plan['seeds'])
            if failures:
                secondary.append(dict(bits=bits,policy=policy,state='undefined',noise_failed_cases=failures,
                    planned_cases=plan['prompts']*len(plan['seeds']),mean=None,ci_low=None,ci_high=None,
                    contrast='quantized FR minus storage-noise FR late MUSIQ'))
                continue
            differences=[mean([late[p,s,bits,policy]['fr_minus_noise_musiq'] for s in plan['seeds']])
                         for p in range(plan['prompts'])]
            result=paired_inference(differences,plan['bootstrap_replicates'],plan['analysis_seed'],.95)
            result.pop('p_exact_sign_flip')
            secondary.append(dict(bits=bits,policy=policy,contrast='quantized FR minus storage-noise FR late MUSIQ',**result))
    for p,adjusted in zip(primary,holm([p['p_exact_sign_flip'] for p in primary])):p['p_holm']=adjusted
    out.mkdir(parents=True,exist_ok=True)
    with (out/'windows.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(windows[0]));writer.writeheader();writer.writerows(windows)
    with (out/'paired_primary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['prompt','bits','seed_mean_p3_minus_p1_fr_musiq']);writer.writeheader()
        for r in primary:
            for p,d in enumerate(r['prompt_differences']):writer.writerow(dict(prompt=p,bits=r['bits'],seed_mean_p3_minus_p1_fr_musiq=d))
    arm_summary=[]
    for bits in plan['bits']:
        for policy in plan['policies']:
            subset=[r for r in windows if r['phase']=='late' and r['bits']==bits and r['policy']==policy]
            arm_summary.append(dict(bits=bits,policy=policy,**{key:None if any(r[key] is None for r in subset) else mean([r[key] for r in subset]) for key in subset[0]
                if key not in ('case','prompt','seed','policy','bits','phase','first_chunk','last_chunk')}))
    for p in sorted(data.rglob('*')):
        if p.is_file() and p.suffix in ('.json','.pt'):
            artifacts.append(dict(path=str(p.relative_to(data)),bytes=p.stat().st_size,sha256=file_hash(p)))
    stats=dict(registration=seal,complete_cases=expected,primary=primary,secondary=secondary,
               noise_instability=[dict(case=c['name'],policy=c['policy'],bits=c['bits'],
                   **c['sampling']['noise']['numerical_failure']) for c in cases if c['sampling']['noise'].get('state')=='nonfinite'],
               late_arm_summary=arm_summary,local_artifact_root='${HOME}/sfq_data/policy_study',
               artifacts=artifacts)
    (out/'evidence.json').write_text(json.dumps(stats,indent=2)+'\n')
    figures(cases,out,plan)
    lines=['# P1/P3 registered study results','','Protocol: [P1_P3_MAIN_STUDY.md](../P1_P3_MAIN_STUDY.md).',
           '10 held-out original prompts × 2 seeds; 80 policy cases; 126 latent frames per trajectory.',
           'All registered cases completed; development gates preceded held-out sampling.','',
           '## Primary prompt-paired frame-quality contrast','',
           'Positive = P3 FR has higher late MUSIQ than P1 FR. Seeds are averaged within prompt; n=10 prompts.',
           'Intervals are 97.5% prompt-cluster percentile bootstrap; Holm adjusts the two sign-flip tests.','',
           '| Bits | P3−P1 MUSIQ | 97.5% CI | Exact p | Holm p |','|---|---:|---|---:|---:|']
    for r in primary:
        lines.append(f'| {r["bits"]} | {r["mean"]:+.3f} | [{r["ci_low"]:+.3f}, {r["ci_high"]:+.3f}] | {r["p_exact_sign_flip"]:.4f} | {r["p_holm"]:.4f} |')
    lines+=['','The sign-flip test assumes exchangeability/symmetry of paired differences under the null. Bootstrap coverage is approximate at n=10.','',
            '## Late arm scores (descriptive)','',
            '| Bits | Policy | BF16 | TF | FR | Noise FR |','|---|---|---:|---:|---:|---:|']
    for r in arm_summary:
        scores=['undefined' if r[a+'_musiq'] is None else f'{r[a+"_musiq"]:.2f}' for a in ('bf16','tf','fr','noise')]
        lines.append(f'| {r["bits"]} | {r["policy"]} | '+ ' | '.join(scores)+' |')
    lines+=['','## Storage-noise contrast (secondary)','',
            '| Bits | Policy | Quantized FR−noise FR MUSIQ | Exploratory 95% CI |',
            '|---|---|---:|---|']
    for r in secondary:
        if r['mean'] is None:
            lines.append(f'| {r["bits"]} | {r["policy"]} | undefined | {r["noise_failed_cases"]}/{r["planned_cases"]} controls numerically unstable |')
        else:lines.append(f'| {r["bits"]} | {r["policy"]} | {r["mean"]:+.3f} | [{r["ci_low"]:+.3f}, {r["ci_high"]:+.3f}] |')
    lines+=['','The noise control replays fixed per-encode absolute innovation norms recorded from paired quantized FR; eligible support comes from the control input. It does not match accumulated read errors or establish a chaos baseline.',
            f'Numerically unstable noise controls: {len(stats["noise_instability"])}/{expected}. Per the preregistered rule, no finite-only quality contrasts or imputed scores are reported. Primary TF/FR completeness remains mandatory.','',
            '## Scope and limitations','',
            '- Same nominal bit widths, not equal memory budgets; byte-container simulation with BF16 attention. No packed-kernel speed or memory claim.',
            '- Policy changes include grouping, padding, current-chunk precision and re-quantization. This does not isolate one mechanism or establish a new method.',
            '- MUSIQ measures sampled frame quality. No subject-consistency, semantic-faithfulness or temporal-quality claim; adjacent-frame difference is only a motion proxy.',
            '- TF images stitch counterfactual chunks, and VAE decoding is not teacher-forced. FR/TF latent distances are descriptive, not an additive causal decomposition.',
            '- Ten convenience prompts and two seeds provide limited coverage; chunks and seeds were not counted as independent samples.','',
            '![BF16-relative frame quality](musiq.png)','![Historical KV errors](kv_history.png)','',
            'Data: [all windows](windows.csv), [prompt-paired primary differences](paired_primary.csv), [evidence and artifact hashes](evidence.json).',
            'Final latent tensors, videos and raw call metrics remain outside Git under `${HOME}/sfq_data/policy_study`.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('FINAL ANALYSIS COMPLETE',out,flush=True)
    return stats


def figures(cases,out,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in ('musiq','kv_history'):
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True)
        for ax,bits in zip(axes,plan['bits']):
            for policy in plan['policies']:
                subset=[c for c in cases if c['bits']==bits and c['policy']==policy]
                if kind=='musiq':
                    curves=[[c['quality']['fr']['per_chunk'][i]['musiq']-c['quality']['bf16']['per_chunk'][i]['musiq'] for i in range(42)] for c in subset]
                    ax.plot(range(42),np.mean(curves,axis=0),label=policy)
                    ax.set_ylabel('FR minus BF16 MUSIQ')
                else:
                    for key,label in [('read_history_k_rel','K'),('read_history_v_rel','V')]:
                        curves=[[mean([r[key] for r in c['sampling']['fr']['history'] if r['chunk']==i]) for i in range(42)] for c in subset]
                        ax.plot(range(42),np.mean(curves,axis=0),label=policy+' '+label)
                    ax.set_ylabel('Historical relative KV error (macro mean)')
            ax.axvline(7,color='gray',linestyle=':');ax.axvspan(21,41,alpha=.05,color='black')
            ax.set_title(f'INT{bits}');ax.set_xlabel('Chunk');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/f'{kind}.png',dpi=160);fig.savefig(out/f'{kind}.pdf');plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path.home()/'sfq_data/policy_study')
    ap.add_argument('--out',type=Path,default=ROOT/'experiments/policy_study_results')
    args=ap.parse_args();analyze(args.data,args.out)
