"""Create descriptive tables and scientific plots after all pilot cases finish."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kvq.verification import public_snapshot
ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path.home()/'sfq_data/p3_pilot')
a=ap.parse_args();out=ROOT/'experiments/p3_pilot_results';out.mkdir(parents=True,exist_ok=True)
records=[];cases={};commands={}
for bits in (4,2):
    for prompt in (0,1):
        name=f'p{prompt}_s0_int{bits}';folder=a.data/name
        complete=json.loads((folder/'complete.json').read_text())
        quality=json.loads((folder/'quality.json').read_text())
        cases[name]={'sampling':complete,'quality':quality,
                     'cleanup':json.loads((folder/'cleanup.json').read_text())}
        for phase,lo,hi in [('early',1,7),('middle',7,21),('late',21,42)]:
            rows=complete['rows'][lo:hi]
            row={'case':name,'prompt':prompt,'bits':bits,'phase':phase,'first_chunk':lo,'last_chunk':hi-1}
            for key in ('tf_rel','fr_rel','fr_vs_tf_rel','fr_minus_tf_rel'):
                row[key]=sum(r[key] for r in rows)/len(rows)
            for arm in ('bf16','tf','fr'):
                vals=quality[arm]['per_chunk'][lo:hi]
                row[f'{arm}_musiq']=sum(v['musiq'] for v in vals)/len(vals)
            row['tf_minus_bf16_musiq']=row['tf_musiq']-row['bf16_musiq']
            row['fr_minus_bf16_musiq']=row['fr_musiq']-row['bf16_musiq']
            records.append(row)
with (out/'windows.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
artifact={'gates':json.loads((a.data/'gates.json').read_text()),
          'queue':json.loads((a.data/'queue.json').read_text()),'cases':cases,
          'gate_cleanup':json.loads((a.data/'gate_cleanup.json').read_text()),
          'limitations':['n=2 prompts; one seed','No inferential or equivalence test',
                         'No BF16 noise control or subject-consistency metric',
                         'Latent-distance difference includes chaotic divergence',
                         'Byte-code simulation, no packed low-bit kernel',
                         'Sampling peak includes concurrent reference/TF/FR caches; no per-policy memory savings claim']}
artifact=public_snapshot(artifact,home=Path.home(),project=ROOT,data=a.data.parent)
(out/'evidence.json').write_text(json.dumps(artifact,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for metric in ('latent','musiq'):
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for i,bits in enumerate((4,2)):
        for prompt in (0,1):
            ax=axes[i,prompt];case=cases[f'p{prompt}_s0_int{bits}'];rows=case['sampling']['rows']
            if metric=='latent':
                for key,label in [('tf_rel','TF vs BF16'),('fr_rel','FR vs BF16')]:
                    ax.plot([r['chunk'] for r in rows],[r[key] for r in rows],label=label)
                ax.set_ylabel('Relative latent L2')
            else:
                for arm in ('bf16','tf','fr'):
                    values=case['quality'][arm]['per_chunk']
                    ax.plot([r['chunk'] for r in values],[r['musiq'] for r in values],label=arm.upper())
                ax.set_ylabel('MUSIQ')
            ax.axvline(7,color='gray',linestyle=':',linewidth=1,label='first eviction')
            ax.set_title(f'INT{bits}, prompt {prompt}');ax.grid(alpha=.2);ax.legend(fontsize=8)
            ax.set_xlabel('Chunk index')
    fig.tight_layout();fig.savefig(out/f'{metric}.png',dpi=160);fig.savefig(out/f'{metric}.pdf');plt.close(fig)
lines=['# P3 TF/FR pilot results','','Fixed protocol: [P3_TF_FR_PILOT.md](../P3_TF_FR_PILOT.md).',
       'Two new prompts, seed 0, 126 latent frames, RTN P3 group 16, no sink/bias.',
       'All source-bound no-quantization and reference gates passed before measurement.','',
       '| Bits | Prompt | Late TF rel | Late FR rel | Late TF−BF16 MUSIQ | Late FR−BF16 MUSIQ |',
       '|---|---|---|---|---|---|']
for r in records:
    if r['phase']=='late':lines.append(f"| {r['bits']} | {r['prompt']} | {r['tf_rel']:.4f} | {r['fr_rel']:.4f} | {r['tf_minus_bf16_musiq']:+.2f} | {r['fr_minus_bf16_musiq']:+.2f} |")
lines += ['', 'Late = chunks 21–41. All window summaries exclude chunk 0.',
          'These are descriptive observations from n=2, not evidence of equivalence or generalization.',
          'FR−TF distances are not an additive causal decomposition; chaotic divergence is included.',
          'MUSIQ and the motion proxy do not establish subject/identity preservation.',
          'Sampling peak includes concurrent reference/TF/FR caches and is not a per-policy memory comparison.', '',
          '![Latent distances](latent.png)','![MUSIQ](musiq.png)','',
          'Per-window values: [windows.csv](windows.csv). Full compact evidence: [evidence.json](evidence.json).',
          f'Videos remain outside Git at `{a.data}`. Derived latent tensors were removed only after successful video/metric export; hashes and removed byte counts are preserved.','']
lines=public_snapshot(lines,home=Path.home(),project=ROOT,data=a.data.parent)
(out/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines[:14]))
