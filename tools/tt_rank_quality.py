"""Raw RGB evaluation and audited reporting for the TT rank/memory sweep.

Keeps the sealed tt_video evaluator unchanged; shares its metric implementations.
"""
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import tt_video_quality as quality
import report_tt_video as audit

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home()/'sfq_data/tt_rank_sweep'
BASE = Path.home()/'sfq_data/tt_video'
OUT = ROOT/'src/experiments/tt_rank_sweep'
RANKS = (64, 128, 192)


def cases(prompt=None):
    return [dict(prompt=p,bits=b,rank=r,arm='tt_quant',key=f'p{p}_b{b}_r{r}')
            for p in ((0,4) if prompt is None else (prompt,))
            for b in (2,3,4) for r in RANKS]


def provenance():
    value=quality.provenance()
    value['files_sha256']['tools/tt_rank_quality.py']=quality.digest(__file__)
    value['files_sha256']['src/experiments/TT_RANK_SWEEP.md']=quality.digest(ROOT/'src/experiments/TT_RANK_SWEEP.md')
    return value


def inputs(case,source_hash):
    folder=DATA/case['key'];s=json.loads((folder/'sampling.json').read_text())
    h=quality.digest(folder/'latent.pt')
    if s['source_sha256']!=source_hash or s['latent_sha256']!=h:
        raise RuntimeError(f'sampling signature/artifact mismatch: {case["key"]}')
    if len(s['chunks'])!=42 or s['prompt']!=case['prompt'] or s['bits']!=case['bits']:
        raise RuntimeError('sampling metadata/incomplete case')
    if s.get('rank',case['rank'])!=case['rank'] or s['ranks']!=[case['rank']]:
        raise RuntimeError('sampling TT rank mismatch')
    return h,quality.digest(folder/'sampling.json')


def reusable(case,source_hash,own_hash,ref_hash,sources):
    folder=DATA/case['key'];path=folder/'quality.json'
    if not path.exists():return None
    q=json.loads(path.read_text());latent_hash,sampling_hash=inputs(case,source_hash)
    expected=dict(**case,frames=501,source_sha256=source_hash,quality_source_sha256=own_hash,
                  latent_sha256=latent_hash,sampling_sha256=sampling_hash,
                  reference_latent_sha256=ref_hash,provenance=sources)
    for k,v in expected.items():
        if q.get(k)!=v:raise RuntimeError(f'completed evaluation changed: {case["key"]}/{k}')
    if quality.digest(folder/'preview.mp4')!=q['preview_sha256']:raise RuntimeError('preview changed')
    audit.audit_quality(q)
    return q


@torch.inference_mode()
def evaluate_all(source_hash,status_callback):
    import pyiqa
    from torchvision.io import write_video
    from utils.wan_wrapper import WanVAEWrapper
    from mechanism_worker import auxiliary_evaluator
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    own_hash=quality.digest(__file__);sources=provenance()
    refs={p:quality.digest(BASE/f'p{p}_b0_bf16/latent.pt') for p in (0,4)}
    existing={}
    for c in cases():
        inputs(c,source_hash)
        q=reusable(c,source_hash,own_hash,refs[c['prompt']],sources)
        if q is not None:existing[c['key']]=q
    if len(existing)==18:
        status_callback('QUALITY complete: 18 verified cached videos');return list(existing.values())
    status_callback(f'QUALITY loading VAE/MUSIQ/DINO/RAFT: {len(existing)}/18 cached')
    vae=WanVAEWrapper().to(device='cuda',dtype=torch.bfloat16).eval()
    metric=pyiqa.create_metric('musiq',device='cuda');evaluator=auxiliary_evaluator()
    records=[]
    try:
        for prompt in (0,4):
            group=cases(prompt)
            if all(c['key'] in existing for c in group):
                records.extend(existing[c['key']] for c in group);continue
            reference,ref_seconds=quality.decode_pixels(vae,BASE/f'p{prompt}_b0_bf16/latent.pt')
            for case in group:
                key=case['key'];folder=DATA/key
                if key in existing:records.append(existing[key]);continue
                latent_hash,sampling_hash=inputs(case,source_hash)
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                status_callback(f'QUALITY {len(records)+1}/18 {key}: decode/evaluate')
                pixels,decode_seconds=quality.decode_pixels(vae,folder/'latent.pt')
                scores=quality.score_pixels(pixels,reference,metric,evaluator,
                    lambda msg:status_callback(f'QUALITY {key}: {msg}'))
                begin=time.perf_counter();tmp=folder/'preview.tmp.mp4'
                write_video(str(tmp),pixels,fps=16,video_codec='libx264',options={'crf':'18'})
                tmp.replace(folder/'preview.mp4')
                row=dict(**case,frames=501,source_sha256=source_hash,quality_source_sha256=own_hash,
                    latent_sha256=latent_hash,sampling_sha256=sampling_hash,
                    reference_latent_sha256=refs[prompt],preview_sha256=quality.digest(folder/'preview.mp4'),
                    raw_rgb_sha256=hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
                    metric_input='raw decoded RGB8 before MP4',decode_seconds=decode_seconds,
                    reference_decode_seconds_shared=ref_seconds,preview_seconds=time.perf_counter()-begin,
                    quality_seconds=time.perf_counter()-started,
                    quality_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    quality_peak_reserved_bytes=torch.cuda.max_memory_reserved(),provenance=sources,**scores)
                if quality.digest(__file__)!=own_hash or inputs(case,source_hash)!=(latent_hash,sampling_hash):
                    raise RuntimeError('quality source/input changed during scoring')
                quality.save(folder/'quality.json',row);records.append(row)
                status_callback(f'QUALITY {len(records)}/18 {key}: late MUSIQ={row["windows"]["late"]["musiq"]:.3f}')
                del pixels
            del reference
            gc.collect();torch.cuda.empty_cache()
    finally:
        del vae,metric,evaluator
        gc.collect();torch.cuda.empty_cache()
    if len(records)!=18:raise RuntimeError('incomplete evaluation')
    return records


def report():
    manifest=audit.read(DATA/'manifest.json');base_manifest=audit.read(BASE/'manifest.json')
    if audit.sha(BASE/'manifest.json')!=manifest['baseline_manifest_sha256']:raise RuntimeError('baseline manifest changed')
    for rel,h in manifest['baseline_artifact_sha256'].items():
        if audit.sha(BASE/rel)!=h:raise RuntimeError(f'baseline artifact changed: {rel}')
    for rel,h in manifest['source_files'].items():
        if audit.sha(ROOT/rel)!=h:raise RuntimeError(f'frozen source changed: {rel}')
    for rel,h in base_manifest['source_files'].items():
        if audit.sha(ROOT/rel)!=h:raise RuntimeError(f'baseline source changed: {rel}')
    sources=provenance();rows=[];checks=[]
    for prompt in (0,4):
        ref=torch.load(BASE/f'p{prompt}_b0_bf16/latent.pt',map_location='cpu',weights_only=True)['latent']
        for bits in (2,3,4):
            for arm in ('quant','tt_quant'):
                folder=BASE/f'p{prompt}_b{bits}_{arm}'
                row,check=audit.audit_case(folder,base_manifest,ref)
                row.update(key=folder.name,rank=299 if arm=='tt_quant' else 0,origin='baseline')
                rows.append(row);checks.append(check)
        for case in cases(prompt):
            folder=DATA/case['key'];s=audit.read(folder/'sampling.json')
            q=reusable(case,manifest['source_sha256'],audit.sha(__file__),audit.sha(BASE/f'p{prompt}_b0_bf16/latent.pt'),sources)
            if q is None:raise RuntimeError(f'missing evaluation: {case["key"]}')
            if s['model_source_sha256']!=manifest['model_source_sha256']:raise RuntimeError('model source mismatch')
            if manifest['model_source_sha256']!=base_manifest['model_source_sha256']:raise RuntimeError('baseline model differs')
            if s['encode_calls']!=30*41:raise RuntimeError('encode count mismatch')
            if not audit.close(s['generation_seconds'],s['sum_chunk_seconds'],rtol=1e-10):raise RuntimeError('timing totals mismatch')
            for chunk in s['chunks']:
                for field in ('seconds','latent_nmse','latent_rel','history_v_rel','encode_seconds','restore_seconds'):
                    audit.finite(chunk[field],field)
                    if chunk[field]<0:raise RuntimeError(f'negative metric: {field}')
            latent=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True)['latent']
            errors=audit.latent_errors(latent,ref);del latent
            for name,v in errors.items():
                if not audit.close(v,s['latent_nmse_'+name]):raise RuntimeError('latent recomputation mismatch')
            row=audit.row_from_records(s,q);row.update(key=case['key'],rank=case['rank'],origin='rank_sweep')
            rows.append(row);checks.append(dict(case=case['key'],sampling_sha256=audit.sha(folder/'sampling.json'),quality_sha256=audit.sha(folder/'quality.json'),latent_nmse_fp64=errors))
        del ref
    contrasts=[]
    for r in rows:
        if r['arm']=='quant':continue
        base=next(b for b in rows if b['prompt']==r['prompt'] and b['bits']==r['bits'] and b['arm']=='quant')
        contrasts.append(dict(key=r['key'],prompt=r['prompt'],bits=r['bits'],rank=r['rank'],
            late_musiq_delta=r['late_musiq']-base['late_musiq'],late_dino_delta=r['late_dino_subject']-base['late_dino_subject'],
            total_cache_ratio=r['cache_total_bytes']/base['cache_total_bytes'],v_cache_ratio=r['cache_v_bytes']/base['cache_v_bytes'],
            peak_ratio=r['generation_peak_allocated_bytes']/base['generation_peak_allocated_bytes'],time_ratio=r['generation_seconds']/base['generation_seconds'],
            musiq_memory_screen=(r['late_musiq']-base['late_musiq']>=-1 and r['cache_total_bytes']<base['cache_total_bytes'])))
    OUT.mkdir(parents=True,exist_ok=True);audit.write_csv(OUT/'summary.csv',rows);audit.write_csv(OUT/'contrasts.csv',contrasts)
    quality.save(OUT/'audit.json',dict(complete=True,new_cases=18,baseline_cases=12,checks=checks))
    text=['# TT → 양자화: rank별 품질·메모리 개발 실험','',
        '개발 프롬프트 p0/p4, seed101, 42청크/501 RGB 프레임. 새 rank 64/128/192와 기존 rank299 및 직접 양자화를 비교한다.',
        '과거 V만 압축하며 K와 현재 청크는 BF16이다. 실제 패킹 바이트와 전체 생성 GPU peak를 구분한다. MUSIQ는 MP4 압축 전 모든 RGB8 프레임에서 측정했다.',
        '이것은 n=2 탐색적 개발 곡선이며 품질 동등성·비열등성을 확증하는 검정이 아니다. MUSIQ와 함께 DINO, 움직임, warp 잔차와 영상을 확인해야 한다.',
        '2/3/4비트 양자화는 고정 symmetric RTN의 3/7/15수준이다. 이전 동일 예산 비교의 rank299도 함께 표시한다.','',
        '| 비트 | 경로/rank | 후반 MUSIQ | 후반 DINO | 내부 flow px | 경계 warp | 총캐시 GiB | V캐시 GiB | peak GiB | 생성 초 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for bits in (2,3,4):
        for rank in (0,64,128,192,299):
            group=[r for r in rows if r['bits']==bits and r['rank']==rank]
            mean=lambda k:audit.mean(r[k] for r in group)
            vals=[mean(k) for k in ('late_musiq','late_dino_subject','late_inside_flow_pixels','late_boundary_warp_error')]
            vals += [mean(k)/2**30 for k in ('cache_total_bytes','cache_v_bytes','generation_peak_allocated_bytes')]
            vals += [mean('generation_seconds')]
            text.append(f'| {bits} | '+('양자화만' if rank==0 else f'TT→양자화 r{rank}')+' | '+' | '.join(audit.fmt(v) for v in vals)+' |')
    text+=['','고정된 K와 현재 BF16 V만 약 3.214 GiB이다. 따라서 과거 V payload의 큰 절감도 총 KV 절감에서는 작게 보인다. rank64의 TT 결합 rank는 (64,64), rank128/192/299는 각각 (128,128)/(192,128)/(299,128)이다.','',
        '## 개발 스크리닝','', '| 비트 | rank | 두 프롬프트 모두 MUSIQ 손실 ≤1점 및 총캐시 감소 |', '|---|---:|---|']
    for bits in (2,3,4):
        for rank in (64,128,192,299):
            group=[c for c in contrasts if c['bits']==bits and c['rank']==rank]
            passed=len(group)==2 and all(c['musiq_memory_screen'] for c in group)
            text.append(f'| {bits} | {rank} | '+('예 — 움직임 검토 필요' if passed else '아니오')+' |')
    text+=['','## 프롬프트별 직접 양자화 대비','',
        '| 비트 | rank | 프롬프트 | 후반 MUSIQ 차이 | 총캐시 비율 | V캐시 비율 | peak 비율 | 시간 배수 |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for c in sorted(contrasts,key=lambda c:(c['bits'],c['rank'],c['prompt'])):
        text.append(f'| {c["bits"]} | {c["rank"]} | p{c["prompt"]} | '+' | '.join(audit.fmt(c[k]) for k in ('late_musiq_delta','total_cache_ratio','v_cache_ratio','peak_ratio','time_ratio'))+' |')
    text+=['','생성 시간은 CUDA 동기화한 run_chunk 시간의 합이며 모델 로딩·품질 평가는 제외한다.',
        '단일 실행 타이밍으로 반복 측정된 배포 성능이 아니다. TT의 FP32 분해/복원 버퍼 때문에 저장 바이트 절감이 peak 감소와 같지는 않다.',
        '개발 스크리닝은 두 프롬프트 각각 후반 MUSIQ 손실 1점 이내와 실제 총캐시 감소를 함께 확인한다. 통과해도 품질 동등성 확증이 아니며 움직임·일관성·warp와 영상 검토를 추가해야 한다.','',
        '[전체 수치](summary.csv) · [대응 차이](contrasts.csv) · [검증](audit.json)',
        '새 영상/latent는 `~/sfq_data/tt_rank_sweep/p{prompt}_b{bits}_r{rank}/`에 보존한다. 기존 기준선은 `~/sfq_data/tt_video/`에 있다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(text)+'\n')
    return dict(complete=True,cases=18,out=str(OUT))


if __name__=='__main__':
    print(json.dumps(report(),ensure_ascii=False))
