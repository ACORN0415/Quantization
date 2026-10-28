"""Plot completed diagnostics and an explicitly fixed qualitative example."""
import csv
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/experiments/policy_supplement_results'
DATA=Path.home()/'sfq_data/policy_study'


def main():
    rows=list(csv.DictReader((OUT/'windows.csv').open()))
    musiq=list(csv.DictReader((ROOT/'src/experiments/policy_study_results/windows.csv').open()))
    spec=[('musiq','Frame MUSIQ (raw input)',True),('subject','DINO feature consistency',True),
          ('warp_error','Flow-compensated RGB residual',False),('flow_pixels','Flow magnitude (resized pixels)',None),
          ('coverage','Valid correspondence fraction',None)]
    cells=[]
    fig,axes=plt.subplots(2,5,figsize=(18,6.4))
    for row,bits in enumerate((4,2)):
        for col,(metric,title,higher) in enumerate(spec):
            values=[]
            for policy in ('reference','rolling','chunk'):
                if metric=='musiq':
                    r=[v for v in musiq if int(v['bits'])==bits and v['policy']=='rolling' and v['phase']=='late'] if policy=='reference' else [v for v in musiq if int(v['bits'])==bits and v['policy']==policy and v['phase']=='late']
                    key='bf16_musiq' if policy=='reference' else 'fr_musiq'
                    values.append(float(np.mean([float(v[key]) for v in r])))
                else:
                    r=[v for v in rows if v['window']=='late' and ((v['arm']=='bf16') if policy=='reference' else (int(v['bits'])==bits and v['policy']==policy and v['arm']=='fr'))]
                    values.append(float('nan') if any(v[metric]=='' for v in r) else float(np.mean([float(v[metric]) for v in r])))
            ax=axes[row,col];ax.bar(['BF16','P1 FR','P3 FR'],values,color=['#777777','#d98c4b','#387cac'])
            ax.set_title(title,fontsize=10);ax.set_ylabel(f'INT{bits} / late chunks 21-41' if col==0 else '')
            ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.2)
            for x,v in enumerate(values):
                ax.text(x,v if np.isfinite(v) else 0,f'{v:.3f}' if np.isfinite(v) else 'undefined',ha='center',va='bottom',fontsize=8)
            ax.margins(y=.2)
            cells.extend(dict(bits=bits,policy=p,metric=metric,mean=v if np.isfinite(v) else None) for p,v in zip(('reference','rolling','chunk'),values))
    fig.suptitle('Exploratory video diagnostics: interpret appearance and residual together with motion and coverage',fontsize=12)
    fig.tight_layout(rect=(0,.03,1,.94))
    fig.text(.5,.01,'20 prompt/seed videos per cell. Supplement uses compressed MP4; MUSIQ uses original raw frames. No temporal-quality composite.',ha='center',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(OUT/f'diagnostics.{ext}',dpi=160)
    plt.close(fig)
    with (OUT/'late_arm_means.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
    paths=[('BF16',DATA/'reference/p00_s101/bf16.mp4')]
    paths += [(f'P{1 if policy=="rolling" else 3} FR INT{bits}',DATA/f'cases/p00_s101_{policy}_int{bits}/fr.mp4') for bits in (4,2) for policy in ('rolling','chunk')]
    fig,axes=plt.subplots(3,5,figsize=(15,5.8))
    for col,(label,path) in enumerate(paths):
        cap=cv2.VideoCapture(str(path))
        for row,chunk in enumerate((3,21,41)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,12*chunk+3);ok,frame=cap.read()
            if not ok:raise RuntimeError(f'cannot read {path}')
            axes[row,col].imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(label)
            if col==0:axes[row,col].set_ylabel(f'Chunk {chunk}')
        cap.release()
    fig.suptitle('Fixed example: first registered prompt, seed 101; chunks 3, 21, 41 (no quality-based selection)',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'fixed_example.png',dpi=160);plt.close(fig)
    print('Plots and late-arm means complete')


if __name__=='__main__':main()
