"""Read completed sampling records and export progress without quality/GPU work.

This script never modifies measurement inputs, manifests, sampling records or
final reports. Cache counters and latent errors are reported as measured; this
is not the final FP64 latent/quality audit in report_tt_video.py.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DATA=Path.home()/'sfq_data/tt_video'
DEFAULT_OUT=ROOT/'src/experiments/tt_video_development'
LABELS={'bf16':'BF16 기준','tt':'TT만','quant':'양자화만','quant_tt':'양자화→TT','tt_quant':'TT→양자화'}
ARMS=('tt','quant','quant_tt','tt_quant')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def case_specs():
    return [(p,0,'bf16') for p in (0,4)]+[(p,b,a) for b in (2,3,4) for p in (0,4) for a in ARMS]


def case_key(p,b,a):return f'p{p}_b{b}_{a}'


def record(folder,manifest):
    sampling=read(folder/'sampling.json')
    if sampling['source_sha256']!=manifest['source_sha256']:raise RuntimeError(f'mixed source: {folder.name}')
    if sampling['model_source_sha256']!=manifest['model_source_sha256']:raise RuntimeError(f'mixed model: {folder.name}')
    if sha(folder/'latent.pt')!=sampling['latent_sha256']:raise RuntimeError(f'latent SHA mismatch: {folder.name}')
    chunks=sampling['chunks']
    if len(chunks)!=42 or [c['chunk'] for c in chunks]!=list(range(42)):raise RuntimeError(f'incomplete completed sampling: {folder.name}')
    final=chunks[-1]['cache']
    if final['k_bytes']+final['v_bytes']!=final['total_bytes']:raise RuntimeError('cache accounting mismatch')
    generation=sampling.get('generation_seconds',sampling.get('sum_chunk_seconds',sum(c['seconds'] for c in chunks)))
    row=dict(case=folder.name,prompt=sampling['prompt'],bits=sampling['bits'],arm=sampling['arm'],
        generation_seconds=generation,instrumented_sampling_seconds=sampling['sampling_seconds'],
        latent_nmse_whole=sampling['latent_nmse_whole'],latent_nmse_late=sampling['latent_nmse_late'],
        latent_frames_per_generation_second=126/generation,
        cache_k_bytes=final['k_bytes'],cache_v_bytes=final['v_bytes'],cache_total_bytes=final['total_bytes'],
        cache_pending_kv_bytes=final['pending_kv_bytes'],
        generation_peak_allocated_bytes=sampling['peak_allocated_bytes'],generation_peak_reserved_bytes=sampling['peak_reserved_bytes'],
        encode_seconds=sampling['encode_seconds'],restore_seconds=sampling['restore_seconds'],
        history_v_rel_late=statistics.mean(c['history_v_rel'] for c in chunks[21:]),
        ranks=','.join(str(r) for r in sampling.get('ranks',[])),latent_sha256=sampling['latent_sha256'],
        sampling_sha256=sha(folder/'sampling.json'))
    for k,v in row.items():
        if isinstance(v,(float,int)) and not math.isfinite(v):raise RuntimeError(f'nonfinite {folder.name}/{k}')
    return row


def remaining_estimate(rows,missing,status,estimates):
    """Rough generation-only estimate; excludes future quality evaluation."""
    predictions={}
    for p,b,a in case_specs():
        key=case_key(p,b,a)
        if key not in missing:continue
        same=[r['generation_seconds'] for r in rows if r['arm']==a and r['bits']==b]
        arm=[r['generation_seconds'] for r in rows if r['arm']==a]
        if same:value=statistics.mean(same)
        elif arm:value=statistics.mean(arm)
        elif a in estimates:value=estimates[a]
        else:return None
        fraction=0.
        if status.get('case')==key and status.get('phase')=='sampling' and status.get('chunks')==42:
            fraction=min(1.,max(0.,status.get('chunk',0)/42))
        predictions[key]=value*(1-fraction)
    return sum(predictions.values())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    manifest=read(args.data/'manifest.json')
    for name,digest in manifest['source_files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError(f'measured source changed: {name}')
    specs=case_specs();expected=[case_key(*s) for s in specs]
    if set(manifest['cases'])!=set(expected):raise RuntimeError('manifest case set differs from 26-case plan')
    rows=[];missing=[]
    for key in expected:
        folder=args.data/key
        if not (folder/'sampling.json').exists():missing.append(key);continue
        rows.append(record(folder,manifest))
    status=read(args.data/'status.json') if (args.data/'status.json').exists() else {}
    estimates=read(args.data/'estimate.json').get('per_case_seconds_from_full_window',{}) if (args.data/'estimate.json').exists() else {}
    remaining=remaining_estimate(rows,missing,status,estimates)
    quality_files=sum((args.data/k/'quality.json').exists() for k in expected)
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
    lines=['# TT 네 경로 실제 영상: 생성 중간 보고','',f'갱신: {stamp}.', '',
        f'완료된 생성 **{len(rows)}/26**. 품질 결과 파일 **{quality_files}/26**개(이 중간 보고에는 품질 수치를 반영하지 않음).', '',
        '**이 표는 생성 지표만 보고한다. MUSIQ·DINO·RAFT·육안 영상 품질을 아직 판정하는 보고서가 아니다.** latent 오차가 작다고 영상이 더 좋다고 결론내리지 않는다.', '',
        f'현재 실행 상태: `{status.get("phase","unknown")}` — {status.get("detail","상태 기록 없음")}', '']
    if remaining is not None:
        lines.append(f'남은 생성의 거친 추정: **{remaining/60:.1f}분**. 완료 사례/초기 게이트의 시간을 사용하고 현재 청크 진행을 선형으로 반영했다. 모델 초기화·바깥 CPU 기록·품질평가 시간은 포함하지 않으며 확정 ETA가 아니다.')
    else:lines.append('남은 생성 시간: 같은 팔의 완료/게이트 측정이 아직 부족하여 추정하지 않음.')
    lines+=['','## 완료 사례별 측정','',
        '생성 초는 CUDA 동기화한 실제 run_chunk 시간의 합이다. 바깥 latent 오차 계산/상태 IO를 포함한 경과 시간은 CSV의 instrumented_sampling_seconds에 별도 보존한다. 캐시는 K/V/pending tensor payload, peak는 모델·분해·복원 작업 버퍼 포함이다.', '',
        '| 예산 | 경로 | 프롬프트 | 생성 초 | 후반 latent NMSE | 캐시 GiB | K / V GiB | 생성 peak GiB |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["bits"] or "BF16"} | {LABELS[r["arm"]]} | p{r["prompt"]} | {r["generation_seconds"]:.2f} | {r["latent_nmse_late"]:.6g} | {r["cache_total_bytes"]/2**30:.3f} | {r["cache_k_bytes"]/2**30:.3f} / {r["cache_v_bytes"]/2**30:.3f} | {r["generation_peak_allocated_bytes"]/2**30:.3f} |')
    lines+=['','## 범위와 해석','',
        '- 모든 과거 V만 압축하며 K와 현재 pending은 BF16이다. 전체 KV가 해당 비트로 저장되는 실험이 아니다.',
        '- 같은 V 저장 예산 비교라 TT→양자화는 더 높은 rank를 사용한다. 같은 rank에서 순서만 바꾼 비교와 구분한다.',
        '- 고정 symmetric RTN은2비트 패킹에 −1/0/1의3수준을 저장한다(3/4비트는7/15수준). 최적4수준2비트 양자화나 모든 저비트 방식과의 비교라고 일반화하지 않는다.',
        '- 개발 프롬프트2개·생성 seed1개이며, 아직 완료되지 않은 팔과의 순위나 유의성을 주장하지 않는다.',
        '- 중간 보고는 완료 latent/sampling의 SHA와 계수 일관성을 확인한다. 최종 FP64 latent 재계산·raw 품질 요약 감사는 report_tt_video.py가 따로 수행한다.',
        '', '생성 미완료 사례: '+(', '.join(f'`{k}`' for k in missing) if missing else '없음. 품질 평가 및 최종 보고 완료 여부는 별도 확인해야 한다.'), '',
        '[생성 원시 요약](progress_sampling.csv) · [계획](../TT_VIDEO_DEVELOPMENT.md).']
    args.out.mkdir(parents=True,exist_ok=True)
    if rows:
        with (args.out/'progress_sampling.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    else:(args.out/'progress_sampling.csv').write_text('case,prompt,bits,arm\n')
    (args.out/'PROGRESS_KO.md').write_text('\n'.join(lines)+'\n')
    progress=dict(updated_kst=stamp,completed_sampling=len(rows),total=26,quality_files=quality_files,
        missing_sampling=missing,estimated_remaining_generation_seconds=remaining,current_status=status,
        final_quality_conclusion=False,manifest_sha256=sha(args.data/'manifest.json'))
    (args.out/'progress_sampling.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:progress[k] for k in ('completed_sampling','quality_files','estimated_remaining_generation_seconds')},ensure_ascii=False))

if __name__=='__main__':main()
