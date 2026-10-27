"""Descriptive first-half report; no interim tests or full-study analysis."""
import csv
import json
from pathlib import Path
import statistics
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src/scripts'))
from analyze_policy_study import window


def main():
    data=Path.home()/'sfq_data/policy_study'
    read=lambda p:json.loads(p.read_text())
    checkpoint=read(data/'pause_checkpoint.json')
    if checkpoint['state']!='paused' or checkpoint['completed']!=40:raise RuntimeError('paused 40-case checkpoint required')
    if read(data/'status.json')['study']['state']!='paused':raise RuntimeError('study is not paused')
    seal=read(REPO/'src/experiments/policy_study_registration.json')
    plan=read(REPO/'src/experiments/policy_study_plan.json')
    records=[];seen=set()
    for p in sorted((data/'cases').glob('*/complete.json')):
        c=read(p);folder=p.parent;prompt=c['prompt_index'];seed=c['seed']
        if c['source_sha256']!=seal['source_sha256']:raise RuntimeError('source mismatch')
        seen.add((prompt,seed,c['bits'],c['policy']))
        reference=data/'reference'/f'p{prompt:02d}_s{seed}'
        case=dict(name=c['case'],prompt=prompt,seed=seed,bits=c['bits'],policy=c['policy'],
            sampling={a:read(folder/f'{a}_sampling.json') for a in ('tf','fr','noise')},
            quality={a:read((reference if a=='bf16' else folder)/f'{a}_quality.json') for a in ('bf16','tf','fr','noise')})
        if any(q['source_sha256']!=seal['source_sha256'] for q in case['quality'].values()):raise RuntimeError('quality source mismatch')
        records.append(window(case,'late',*plan['late_chunks']))
    expected={(p,s,b,policy) for p in range(5) for s in plan['seeds'] for b in plan['bits'] for policy in plan['policies']}
    if len(records)!=40 or seen!=expected:raise RuntimeError('first five prompts are not complete and balanced')
    lookup={(r['prompt'],r['seed'],r['bits'],r['policy']):r for r in records}
    summary=[];paired=[]
    for bits in plan['bits']:
        means={}
        for policy in plan['policies']:
            rows=[r for r in records if r['bits']==bits and r['policy']==policy]
            means[policy]={a:None if any(r[a+'_musiq'] is None for r in rows) else statistics.mean(r[a+'_musiq'] for r in rows)
                           for a in ('bf16','tf','fr','noise')}
        differences=[]
        for prompt in range(5):
            d=[lookup[prompt,s,bits,'chunk']['fr_musiq']-lookup[prompt,s,bits,'rolling']['fr_musiq'] for s in plan['seeds']]
            differences.append(statistics.mean(d));paired.append(dict(prompt=prompt,bits=bits,seed_mean_p3_minus_p1_fr_musiq=statistics.mean(d)))
        summary.append(dict(bits=bits,prompts=5,seeds=plan['seeds'],means=means,
                            paired_prompt_mean_p3_minus_p1_fr_musiq=statistics.mean(differences)))
    out=REPO/'src/experiments/policy_study_midpoint'
    for filename,rows in [('late_windows.csv',records),('paired_descriptive.csv',paired)]:
        with (out/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'descriptive.json').write_text(json.dumps(dict(completed_cases=40,independent_prompts=5,
        inference='none; registered full-sample analysis deferred',source_sha256=seal['source_sha256'],summary=summary),indent=2)+'\n')
    lines=['','## First-half descriptive frame quality','',
        'First five prompts × two seeds. Late = chunks 21–41, raw middle-frame MUSIQ. These are point estimates only; no interim p-values or confidence intervals.',
        'The policy difference is paired within prompt/seed and averaged within prompt before averaging five prompts.','',
        '| Bits | BF16 | P1 FR | P3 FR | P3−P1 |','|---|---:|---:|---:|---:|']
    for r in summary:
        m=r['means'];lines.append(f'| {r["bits"]} | {m["rolling"]["bf16"]:.2f} | {m["rolling"]["fr"]:.2f} | {m["chunk"]["fr"]:.2f} | {r["paired_prompt_mean_p3_minus_p1_fr_musiq"]:+.2f} |')
    lines+=['','No final policy-superiority, temporal-consistency or causal-propagation claim follows from this checkpoint. The remaining prompts, seeds, endpoints and full-sample analysis stay fixed.',
        'Data: [late windows](late_windows.csv), [paired point differences](paired_descriptive.csv), [descriptive summary](descriptive.json).']
    with (out/'REPORT.md').open('a') as f:f.write('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
