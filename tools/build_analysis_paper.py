"""Reaggregate existing tables and render the manuscript diagram; CPU only.

Does not import inference modules, modify sealed experiments or generate videos.
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'문서/paper/analysis_assets'


def read(name):
    return json.loads((ROOT/name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    evidence_name='src/experiments/mechanism_fr_results/evidence.json'
    table_name='src/experiments/mechanism_fr_results/windows.csv'
    final_name='src/experiments/repeat_once_final/summary.json'
    manifest_name='src/experiments/repeat_once_final/manifest.json'
    e=read(evidence_name);s=read(final_name);manifest=read(manifest_name)
    for name,sha in manifest['new_source_sha256'].items():
        assert digest(ROOT/name)==sha, f'Sealed experiment source changed: {name}'
    with (ROOT/table_name).open() as f:rows=list(csv.DictReader(f))
    keys={(int(r['prompt']),int(r['seed']),r['arm']) for r in rows}
    expected=set(itertools.product(range(12),(303,404),('BF16','L','C_repeat')))
    assert len(rows)==72 and keys==expected
    values={(int(r['prompt']),int(r['seed']),r['arm']):float(r['late_raw_musiq']) for r in rows}
    delta=np.array([statistics.mean(values[p,t,'C_repeat']-values[p,t,'L'] for t in (303,404))
                    for p in range(12)])
    mean=float(delta.mean())
    signs=np.array(list(itertools.product((-1.,1.),repeat=12)))
    pvalue=float(np.mean(np.abs((signs*delta).mean(1))>=abs(mean)-1e-12))
    boot=delta[np.random.default_rng(872341).integers(0,12,(20000,12))].mean(1)
    ci=np.quantile(boot,[.025,.975]).tolist()
    assert np.allclose(delta,e['prompt_differences'],rtol=0,atol=1e-12)
    assert abs(mean-e['mean'])<1e-12 and pvalue==e['exact_two_sided_sign_flip_p']
    assert np.allclose(ci,e['ci95'],rtol=0,atol=1e-12)
    assert s['conclusion']=='all_24_exact' and s['stop_reason']=='complete'
    assert len(s['gates'])==2 and all(g['status']=='valid_equal' for g in s['gates'])
    assert len(s['comparisons'])==24 and all(c['status']=='valid_equal' for c in s['comparisons'])
    assert {(c['prompt'],c['seed']) for c in s['comparisons']}==set(itertools.product(range(12),(303,404)))
    assert len(s['budget']['attempts'])==26 and s['completed_valid']==26
    assert s['budget']['elapsed_through_report_s']<14400
    performance=read('src/experiments/mechanism_diagnostic_results/performance.json')['performance']
    timings={}
    for arm in ('BF16','L','C_repeat','C_once'):
        selected=[r for r in performance if r['arm']==arm and r['group']==16]
        assert len(selected)==3
        timings[arm]=dict(median_generation_s=statistics.median(r['generation_s'] for r in selected),
                         resident_gib=statistics.median(r['cache']['resident_tensor_bytes']/2**30 for r in selected))
    tt_name='src/experiments/tt_kv_final/summary.csv'
    with (ROOT/tt_name).open() as f:tt=list(csv.DictReader(f))
    tt_summary={}
    for bits in (2,4):
        group={rank:[r for r in tt if int(r['bits'])==bits and int(r['rank'])==rank] for rank in (0,192)}
        assert all(len(rs)==2 for rs in group.values())
        means={rank:{k:statistics.mean(float(r[k]) for r in rs)
                     for k in ('late_musiq','late_dino_subject','cache_total_bytes','generation_seconds')}
               for rank,rs in group.items()}
        tt_summary[str(bits)]=dict(means=means,
            musiq_delta=means[192]['late_musiq']-means[0]['late_musiq'],
            cache_reduction=1-means[192]['cache_total_bytes']/means[0]['cache_total_bytes'],
            generation_ratio=means[192]['generation_seconds']/means[0]['generation_seconds'])
    result=dict(scope='existing-result table reaggregation and sealed-source preservation; no new GPU runs',
                n_prompts=12,positive_prompts=int((delta>0).sum()),mean=mean,ci95=ci,p=pvalue,
                prompt_differences=delta.tolist(),
                arm_means={a:statistics.mean(v for (p,t,arm),v in values.items() if arm==a)
                           for a in ('BF16','L','C_repeat')},
                final_equal_pairs=24,repeat_reproduction_count=2,attempts=26,
                verification_elapsed_s=s['budget']['elapsed_through_report_s'],
                performance=timings,tt=tt_summary,
                inputs={n:digest(ROOT/n) for n in (evidence_name,table_name,final_name,manifest_name,tt_name)},
                sealed_sources_unchanged=True)
    (OUT/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    draw_paths()
    print(json.dumps(dict(mean=mean,ci95=ci,p=pvalue,positive_prompts=int((delta>0).sum()),
                          final_equal_pairs=24,sealed_sources_unchanged=True),indent=2))


def draw_paths():
    fig,ax=plt.subplots(figsize=(12,8))
    ax.set_xlim(0,12);ax.set_ylim(-.5,8.4);ax.axis('off')
    names=['L: rolling','C-repeat: chunk groups','C-once: chunk groups']
    steps=[
        ['Restore full buffer','On eviction: shift survivors','Overwrite current chunk (BF16)',
         'BF16 attention on valid tokens','Encode entire buffer\nRecompute scales','Keep codes + FP16 scales'],
        ['On chunk change: evict oldest','Restore + re-encode survivors\nRecompute scales',
         'On chunk change: encode\nprevious pending once','Replace current pending (BF16)',
         'Restore history + concatenate','BF16 attention'],
        ['On chunk change: evict oldest','Reuse stored history\nNo re-encode',
         'On chunk change: encode\nprevious pending once','Replace current pending (BF16)',
         'Restore history + concatenate','BF16 attention']]
    for col in range(3):
        x=.15+4*col
        ax.text(x+1.8,8.05,names[col],ha='center',va='center',fontsize=13,fontweight='bold')
        for row,label in enumerate(steps[col]):
            y=6.85-1.05*row
            highlight=(col==0 and row==4) or (col==1 and row==1)
            fill='#ffedcf' if highlight else '#e9f2f7'
            if col==2 and row==1:fill='#e7f2e8'
            ax.add_patch(FancyBboxPatch((x,y),3.6,.72,boxstyle='round,pad=.06',
                                       edgecolor='#49616f',facecolor=fill,linewidth=1.1))
            ax.text(x+1.8,y+.36,label,ha='center',va='center',fontsize=10.2)
            if row<5:ax.annotate('',xy=(x+1.8,y-.29),xytext=(x+1.8,y-.07),
                                  arrowprops=dict(arrowstyle='->',color='#49616f',lw=1.3))
    ax.text(2,.48,'Groups follow physical positions.\nWrites/eviction can change group support.',
            ha='center',va='center',fontsize=10,color='#394b55')
    ax.text(8,.48,'Chunk-local groups, padding, eviction and pending rules are shared.\n'
             'Only C-repeat takes the history re-encoding branch.',ha='center',va='center',fontsize=10,color='#394b55')
    ax.text(6,-.25,'Scales are recomputed on every encode. This is an operation diagram, not a numerical-equality claim.',
            ha='center',va='center',fontsize=10,fontstyle='italic')
    fig.tight_layout()
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'storage_paths.{ext}',dpi=180,bbox_inches='tight',
                    metadata={'Creator':'Quantization analysis manuscript'})
    svg=OUT/'storage_paths.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
