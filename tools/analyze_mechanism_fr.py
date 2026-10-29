"""The single fixed held-out contrast. Does not change the diagnostic GO rule."""
import csv
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import mechanism_run as d
from mechanism_worker import source_hash,sha


def infer(differences):
    x=np.asarray(differences,dtype=np.float64)
    if x.shape!=(12,) or not np.isfinite(x).all():raise ValueError('exactly twelve finite prompt-level differences required')
    observed=float(x.mean());signs=np.asarray(list(itertools.product((-1.,1.),repeat=12)))
    p=float(np.mean(np.abs((signs*x).mean(1))>=abs(observed)-1e-12))
    boot=x[np.random.default_rng(872341).integers(0,12,(20000,12))].mean(1);lo,hi=np.quantile(boot,[.025,.975])
    return observed,float(lo),float(hi),p


def main():
    data=Path.home()/'sfq_data/mechanism';seal=d.read(ROOT/'src/experiments/mechanism_fr_registration.json')
    if source_hash()!=seal['source_sha256']:raise RuntimeError('registered FR source changed')
    rows=[];artifacts=[];auxiliary=[]
    for prompt in range(12):
        for seed in (303,404):
            for arm in ('BF16','L','C_repeat'):
                folder=data/'fr'/f'p{prompt:02d}_s{seed}_{arm}'
                q=d.read(folder/'quality.json');s=d.read(folder/'sampling.json')
                if q['source_sha256']!=seal['source_sha256'] or s['source_sha256']!=seal['source_sha256']:raise RuntimeError('mixed heldout source')
                if sha(folder/'latent.pt')!=s['latent_file_sha256']:raise RuntimeError('heldout tensor altered')
                scores=q['per_frame_musiq']
                if len(scores)!=501 or not np.isfinite(scores).all():raise RuntimeError('invalid raw quality record')
                rows.append(dict(prompt=prompt,seed=seed,arm=arm,late_raw_musiq=float(np.mean(scores[249:501])),
                    boundary_raw_musiq=q['boundary_raw_musiq'],inside_raw_musiq=q['inside_raw_musiq'],
                    generation_s=s['generation_s'],peak_allocated_bytes=s['peak']['allocated']))
                auxiliary.append(dict(prompt=prompt,seed=seed,arm=arm,**q['raw_auxiliary']))
                for name in ('latent.pt','video.mp4','sampling.json','quality.json'):
                    path=folder/name;artifacts.append(dict(path=str(path.relative_to(data)),bytes=path.stat().st_size,sha256=sha(path)))
    differences=[]
    for prompt in range(12):
        values=[]
        for seed in (303,404):
            c={r['arm']:r['late_raw_musiq'] for r in rows if r['prompt']==prompt and r['seed']==seed}
            values.append(c['C_repeat']-c['L'])
        differences.append(float(np.mean(values)))
    observed,lo,hi,p=infer(differences)
    out=ROOT/'src/experiments/mechanism_fr_results';out.mkdir(parents=True,exist_ok=True)
    result=dict(registration=seal,complete_trajectories=72,n_prompts=12,primary='C-repeat16 minus L16 late all-raw-frame MUSIQ',
                mean=observed,ci95=[float(lo),float(hi)],exact_two_sided_sign_flip_p=p,prompt_differences=differences,artifacts=artifacts)
    d.save(out/'evidence.json',result)
    aux_summary=[]
    for arm in ('BF16','L','C_repeat'):
        selected=[r for r in auxiliary if r['arm']==arm]
        entry=dict(arm=arm,subject_late=float(np.mean([r['subject_late'] for r in selected])),
                   empty_pairs=sum(r['empty_pairs'] for r in selected))
        for kind in ('boundary','inside'):
            for metric in ('warp_error','coverage','flow_pixels','raw_difference'):
                values=[r['windows']['late_'+kind][metric] for r in selected]
                entry[kind+'_'+metric]=None if any(v is None for v in values) else float(np.mean(values))
        aux_summary.append(entry)
    d.save(out/'auxiliary.json',dict(exploratory=True,raw_input=True,arm_means=aux_summary,
             note='All 24 prompt/seed videos per arm included; full-cell residual undefined if any window is undefined; no finite-only subset'))
    with (out/'windows.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 그룹 규칙 개입 FR 결과','',f'72/72 궤적 완료, 독립 분석 단위 12개 프롬프트. 주 대비 평균 {observed:+.4f} MUSIQ.',
           f'95% 프롬프트 bootstrap 구간 [{lo:+.4f}, {hi:+.4f}], 양측 정확 부호 반전 p={p:.6f}.',
           '짝 차이의 대칭성/교환 가능성 가정과 작은 편의 표본의 한계가 있다.',
           '이는 청크 단위 그룹 규칙 개입의 효과다. eviction 위상 하나의 순수 효과나 새 양자화 방법의 입증이 아니다.',
           '주지표는 모든 후반 raw 프레임의 MUSIQ다. 경계/내부 품질은 보조 기술 통계이며 시간 품질을 입증하지 않는다.',
           '저장·성능은 byte-container BF16-attention 구현의 비용으로, 실제 packed INT4 커널의 이득이 아니다.',
           '', '[전체 사례](windows.csv) · [등록과 원시 산출물 해시](evidence.json) · [경계/내부·DINO·움직임 보조 결과](auxiliary.json)']
    (out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print('HELDOUT ANALYSIS COMPLETE',observed,p,flush=True)


if __name__=='__main__':main()
