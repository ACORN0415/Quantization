"""Audit completed TT video records and export Korean tables/plots.

This recalculates stored latent errors and scalar summaries. It does not rerun
video generation or evaluate models independently. --partial writes progress
artifacts only and never replaces the complete report.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DATA=Path.home()/'sfq_data/tt_video'
DEFAULT_OUT=ROOT/'src/experiments/tt_video_development'
ARMS=('tt','quant','quant_tt','tt_quant')
LABELS={'bf16':'BF16 기준','tt':'TT만','quant':'양자화만','quant_tt':'양자화→TT','tt_quant':'TT→양자화'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def keys():
    for p in (0,4):yield p,0,'bf16'
    for bits in (2,3,4):
        for p in (0,4):
            for arm in ARMS:yield p,bits,arm


def mean(values):
    values=list(values)
    return None if not values or any(v is None for v in values) else statistics.mean(values)


def close(a,b,*,rtol=2e-5,atol=1e-10):
    if a is None or b is None:return a is None and b is None
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(a-b)<=atol+rtol*max(abs(a),abs(b))


def finite(x,label):
    if not isinstance(x,(int,float)) or not math.isfinite(x):raise RuntimeError(f'nonfinite/missing {label}')


def frame_windows():
    all_indices=list(range(501));boundary=[12*c-3 for c in range(1,42)]
    inside=[i for i in all_indices if i not in set(boundary)]
    return dict(whole=all_indices,late=list(range(249,501)),boundary=boundary,
                late_boundary=[i for i in boundary if i>=249],inside=inside,
                late_inside=[i for i in inside if i>=249])


def audit_quality(q):
    if q['frames']!=501:raise RuntimeError('expected 501 raw decoded frames')
    musiq=q['per_frame_musiq'];mse=q['per_frame_pixel_mse'];dino=q['per_transition_dino']
    if len(musiq)!=501 or len(mse)!=501 or len(dino)!=500:raise RuntimeError('metric record lengths')
    for k,values in [('musiq',musiq),('mse',mse),('dino',dino)]:
        for v in values:finite(v,k)
    for name,indices in frame_windows().items():
        window=q['windows'][name]
        if window['frame_count']!=len(indices):raise RuntimeError(f'frame selection mismatch: {name}')
        expected_dino=mean(dino[i-1] for i in indices if i>0)
        if not close(window['dino_subject'],expected_dino,rtol=1e-8):raise RuntimeError(f'DINO summary mismatch: {name}')
        if not close(window['musiq'],mean(musiq[i] for i in indices),rtol=1e-8):raise RuntimeError(f'MUSIQ summary mismatch: {name}')
        pixel=mean(mse[i] for i in indices)
        if not close(window['pixel_mse'],pixel,rtol=1e-8):raise RuntimeError(f'pixel MSE summary mismatch: {name}')
        expected=None if pixel==0 else -10*math.log10(pixel)
        if not close(window['pixel_psnr_db'],expected,rtol=1e-8):raise RuntimeError(f'pixel PSNR mismatch: {name}')
        if bool(window['pixel_psnr_infinite'])!=(pixel==0):raise RuntimeError('PSNR zero-MSE flag mismatch')
    pairs=q['pairs']
    expected_pairs={('boundary',c,12*c-4,12*c-3) for c in range(1,42)}
    expected_pairs|={('inside',c,12*c+o,12*c+o+1) for c in range(42) for o in (2,3)}
    actual={(p['kind'],p['chunk'],p['i'],p['j']) for p in pairs}
    if len(pairs)!=125 or actual!=expected_pairs:raise RuntimeError('RAFT sample indices mismatch')
    for name in ('whole_boundary','whole_inside','late_boundary','late_inside'):
        late,kind=name.split('_')
        selected=[p for p in pairs if p['kind']==kind and (late!='late' or p['chunk']>=21)]
        result=q['temporal_windows'][name]
        if result['pairs']!=len(selected):raise RuntimeError(f'temporal sample count mismatch: {name}')
        for metric in ('warp_error','coverage','flow_pixels','raw_difference'):
            target=mean(p[metric] for p in selected)
            if not close(result[metric],target,rtol=1e-7):raise RuntimeError(f'temporal summary mismatch: {name}/{metric}')
        if result['empty_mask_pairs']!=sum(p['coverage']==0 for p in selected):raise RuntimeError('empty mask count mismatch')


def latent_errors(latent,reference):
    """FP64 CPU reductions, with late window in latent chunks 21..41."""
    import torch
    if tuple(latent.shape)!=(1,126,16,60,104) or latent.shape!=reference.shape:
        raise RuntimeError('unexpected saved latent geometry')
    values={}
    for name,start in (('whole',0),('late',63)):
        a=latent[:,start:].double();b=reference[:,start:].double()
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):raise RuntimeError('nonfinite latent')
        values[name]=float((a-b).square().sum()/b.square().sum().clamp_min(1e-30))
    return values


def row_from_records(s,q):
    chunks=s['chunks']
    if len(chunks)!=42 or [c['chunk'] for c in chunks]!=list(range(42)):raise RuntimeError('expected all 42 chunk rows')
    full=chunks[6:];last=chunks[-1]['cache']
    for c in chunks:
        cache=c['cache']
        if cache['total_bytes']!=cache['k_bytes']+cache['v_bytes']:raise RuntimeError('cache accounting sum mismatch')
        if cache['pending_kv_bytes']>cache['total_bytes']:raise RuntimeError('pending larger than cache')
    for field in ('k_bytes','v_bytes','total_bytes','pending_kv_bytes'):
        if len({c['cache'][field] for c in full})!=1:raise RuntimeError(f'full-window cache byte plateau not stable: {field}')
    generation_seconds=s.get('generation_seconds',s.get('sum_chunk_seconds',sum(c['seconds'] for c in chunks)))
    result=dict(prompt=s['prompt'],bits=s['bits'],arm=s['arm'],ranks=','.join(str(r) for r in s.get('ranks',[])),
                generation_seconds=generation_seconds,sampling_seconds=s['sampling_seconds'],
                generation_latent_frames_per_second=126/generation_seconds,
                generation_pixel_frames_per_second=501/generation_seconds,
                encode_seconds=s['encode_seconds'],restore_seconds=s['restore_seconds'],
                cache_k_bytes=last['k_bytes'],cache_v_bytes=last['v_bytes'],cache_total_bytes=last['total_bytes'],
                cache_pending_kv_bytes=last['pending_kv_bytes'],
                generation_peak_allocated_bytes=s['peak_allocated_bytes'],generation_peak_reserved_bytes=s['peak_reserved_bytes'],
                latent_nmse_whole=s['latent_nmse_whole'],latent_nmse_late=s['latent_nmse_late'],
                history_v_rel_whole=mean(c['history_v_rel'] for c in chunks),history_v_rel_late=mean(c['history_v_rel'] for c in chunks[21:]),
                quality_seconds=q['quality_seconds'],quality_peak_allocated_bytes=q['quality_peak_allocated_bytes'],
                quality_peak_reserved_bytes=q['quality_peak_reserved_bytes'])
    for name in frame_windows():
        for metric in ('musiq','dino_subject','pixel_mse','pixel_psnr_db','pixel_psnr_infinite'):
            result[name+'_'+metric]=q['windows'][name][metric]
    for name in ('whole_boundary','whole_inside','late_boundary','late_inside'):
        for metric in ('warp_error','coverage','flow_pixels','raw_difference','empty_mask_pairs'):
            result[name+'_'+metric]=q['temporal_windows'][name][metric]
    return result


def audit_case(folder,manifest,reference):
    import torch
    s=read(folder/'sampling.json');q=read(folder/'quality.json')
    signature=manifest['source_sha256']
    if s['source_sha256']!=signature or q['source_sha256']!=signature:raise RuntimeError('mixed generation source signatures')
    if s['model_source_sha256']!=manifest['model_source_sha256']:raise RuntimeError('mixed model source signatures')
    if folder.name!=f'p{s["prompt"]}_b{s["bits"]}_{s["arm"]}':raise RuntimeError('case directory/metadata mismatch')
    for path,h in q['provenance']['files_sha256'].items():
        if sha(ROOT/path)!=h:raise RuntimeError(f'quality dependency changed: {path}')
    if q['sampling_sha256']!=sha(folder/'sampling.json'):raise RuntimeError('quality uses different sampling record')
    h=sha(folder/'latent.pt')
    if h!=s['latent_sha256'] or h!=q['latent_sha256']:raise RuntimeError('latent artifact changed')
    if sha(folder/'preview.mp4')!=q['preview_sha256']:raise RuntimeError('preview artifact changed')
    if (s['prompt'],s['bits'],s['arm'])!=(q['prompt'],q['bits'],q['arm']):raise RuntimeError('metadata mismatch')
    expected_ref=folder.parent/f'p{s["prompt"]}_b0_bf16/latent.pt'
    if q['reference_latent_sha256']!=sha(expected_ref):raise RuntimeError('raw quality reference changed')
    audit_quality(q)
    latent=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True)['latent']
    errors=latent_errors(latent,reference)
    del latent
    for name in ('whole','late'):
        if not close(errors[name],s['latent_nmse_'+name]):raise RuntimeError(f'latent recomputation mismatch: {folder.name}/{name}')
    row=row_from_records(s,q)
    record=dict(case=folder.name,sampling_sha256=sha(folder/'sampling.json'),quality_sha256=sha(folder/'quality.json'),
                latent_sha256=h,preview_sha256=q['preview_sha256'],latent_nmse_fp64=errors,
                quality_source_sha256=q['quality_source_sha256'])
    return row,record


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fmt(v,digits=3):return '미정의' if v is None else f'{v:.{digits}f}'


def paired_contrasts(rows):
    metrics=('whole_musiq','late_musiq','boundary_musiq','late_boundary_musiq','late_dino_subject',
             'late_boundary_flow_pixels','late_inside_flow_pixels','late_boundary_warp_error','late_inside_warp_error',
             'latent_nmse_whole','latent_nmse_late','generation_seconds','sampling_seconds','cache_total_bytes','generation_peak_allocated_bytes')
    contrasts=[]
    by={(r['prompt'],r['bits'],r['arm']):r for r in rows}
    for bits in (2,3,4):
        for arm in ('tt','quant_tt','tt_quant'):
            for p in (0,4):
                if (p,bits,arm) not in by or (p,bits,'quant') not in by:continue
                a=by[p,bits,arm];b=by[p,bits,'quant']
                row=dict(prompt=p,bits=bits,arm=arm,reference='quant')
                for metric in metrics:
                    row[metric+'_difference']=None if a[metric] is None or b[metric] is None else a[metric]-b[metric]
                row['generation_time_ratio']=a['generation_seconds']/b['generation_seconds']
                row['instrumented_sampling_time_ratio']=a['sampling_seconds']/b['sampling_seconds']
                row['cache_byte_ratio']=a['cache_total_bytes']/b['cache_total_bytes']
                contrasts.append(row)
    return contrasts


def report(rows,contrasts,complete):
    lines=['# TT 실제 영상: 네 경로 × 2·3·4비트 비교','',
           '2026-10-30. 개발 프롬프트 p0/p4, 생성 seed101, 각126 latent frames/42청크/501 pixel frames.','',
           f'완료된 생성·품질 평가: **{len(rows)}/26 영상**. '+('모든 예정 사례를 포함했다.' if complete else '**진행 중인 부분 보고이며 최종 비교가 아니다.**'),'',
           '## 비교 정의','',
           '- **TT만:** 원본 V→TT→FP16 인자 저장.',
           '- **양자화만:** 원본 V→실제 packed RTN2/3/4.',
           '- **양자화→TT:** RTN 복원값→TT→FP16 인자 저장.',
           '- **TT→양자화:** 원본 V→TT→인자를 실제2/3/4비트 패킹.',
           '',
           '30개 층의 **과거 V 캐시만** 압축했다. K와 현재 청크는 모든 팔에서 BF16이다. 직전 완료 청크를 한번 인코딩하는 같은 ChunkStore 정책을 사용한다. 전체 KV를2/3/4비트로 저장했다는 뜻이 아니다.',
           '',
           '각 비트는 직접 V 양자화의 실제 저장 예산이다. TT만/양자화→TT의 rank는60/77/94, TT→양자화는 같은 예산에서 rank299를 사용한다. 이 표는 같은 rank 순서 효과와 구분한다. 인자 코드·scale은 실제 패킹했고 정수 rank로 남는 예산은 실제 바이트로 보고한다.',
           '',
           '**양자화 기준의 정확한 의미:** 기존 고정 symmetric RTN은 qmax=2^(b−1)−1, 코드 범위[−qmax,qmax]를 쓴다. 따라서2비트는 실제2비트 패킹 안에 −1/0/1의 **3수준(ternary)**을 저장하며4수준 전체를 쓰지 않는다. 3/4비트도 각각7/15수준이다. TT의 비교 이득은 이 고정 기준선에 대한 결과이며 최적2비트 양자화기나 모든 저비트 방식에 대한 우위로 일반화하지 않는다. 결과를 본 뒤 양자화기를 바꾸지 않았다.', '',
           '## 영상 품질·오차·속도·메모리 종합','',
           '아래는 두 프롬프트 개발 평균이다. 불완전 상태에서 한 프롬프트만 끝난 조건은 n=1로 표시한다. MUSIQ는 raw RGB8 전프레임, 후반은249..500이다. latent NMSE는 같은 BF16 생성과의 궤적 거리이지 시각적 품질 점수가 아니다. 캐시는 K+V+pending tensor payload이며 peak는 모델/작업 버퍼를 포함한다.',
           '',
           '| 예산 | 경로 | n | MUSIQ 전체 / 후반 | 후반 latent NMSE | 생성 초 | 캐시 GiB | 생성 peak GiB |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for bits,arms in [(0,('bf16',))]+[(b,ARMS) for b in (2,3,4)]:
        for arm in arms:
            g=[r for r in rows if r['bits']==bits and r['arm']==arm]
            if not g:continue
            avg=lambda key:mean(r[key] for r in g)
            lines.append(f'| {"BF16" if bits==0 else str(bits)+"비트"} | {LABELS[arm]} | {len(g)} | {fmt(avg("whole_musiq"),2)} / {fmt(avg("late_musiq"),2)} | {fmt(avg("latent_nmse_late"),5)} | {fmt(avg("generation_seconds"),2)} | {fmt(avg("cache_total_bytes")/2**30)} | {fmt(avg("generation_peak_allocated_bytes")/2**30)} |')
    lines+=['','## 같은 프롬프트의 직접 양자화 대비 차이','',
            '양수 MUSIQ 차이는 더 높은 점수, 시간 비율1 초과는 더 느림이다. 두 프롬프트를 따로 제시하며 n=2에 유의성이나 일반화 주장을 붙이지 않는다.','',
            '| 예산 | 경로 | 프롬프트 | 후반 MUSIQ 차이 | 경계 MUSIQ 차이 | 시간 배수 | 캐시 배수 |',
            '|---|---|---|---:|---:|---:|---:|']
    for c in contrasts:
        lines.append(f'| {c["bits"]} | {LABELS[c["arm"]]} | p{c["prompt"]} | {fmt(c["late_musiq_difference"])} | {fmt(c["boundary_musiq_difference"])} | {fmt(c["generation_time_ratio"])} | {fmt(c["cache_byte_ratio"])} |')
    if complete:
        lines+=['','## 개발 결과의 해석','']
        for bits in (2,3,4):
            values=[c for c in contrasts if c['bits']==bits and c['arm']=='tt']
            if len(values)!=2:continue
            diffs=[c['late_musiq_difference'] for c in values]
            if all(v>0 for v in diffs):direction='두 개발 프롬프트 모두 후반 MUSIQ가 높았다'
            elif all(v<0 for v in diffs):direction='두 개발 프롬프트 모두 후반 MUSIQ가 낮았다'
            else:direction='후반 MUSIQ 차이의 방향이 프롬프트에 따라 같지 않았다'
            times=[c['generation_time_ratio'] for c in values]
            lines.append(f'- **{bits}비트 예산의 TT만:** {direction}. 직접 양자화 대비 차이는 p0 {fmt(values[0]["late_musiq_difference"])}점, p4 {fmt(values[1]["late_musiq_difference"])}점, 생성 시간은 각각 {fmt(times[0])}/{fmt(times[1])}배였다. 이 비교는 개발 표본2개이며 보류 데이터의 확증 결과가 아니다.')
            for arm in ('quant_tt','tt_quant'):
                group=[c for c in contrasts if c['bits']==bits and c['arm']==arm]
                if len(group)==2:
                    lines.append(f'  - {LABELS[arm]}의 직접 양자화 대비 후반 MUSIQ 차이는 p0 {fmt(group[0]["late_musiq_difference"])}점, p4 {fmt(group[1]["late_musiq_difference"])}점이다. TT만의 효과를 이 결합 경로로 자동 확장하지 않는다.')
        lines+=['','속도·오차·메모리·움직임을 종합해야 하므로 단일 지표로 최종 승자를 선언하지 않는다. TT 분해 자체가 좋은 경우와 저비트 인자 양자화까지 결합해 좋은 경우를 분리한다.']
    lines+=['','## 움직임·일관성·경계','',
            'DINO는 프레임 특징 일관성 proxy다. 움직임이 줄거나 화면이 정지하면 높아질 수 있으므로 flow magnitude를 함께 본다. RAFT warp residual도 움직임/가림/대응 영역에 영향을 받는다. 전체 subject identity 또는 시간 품질을 이 숫자만으로 확정하지 않는다. 빈 대응 영역이 있으면 warp를 미정의로 유지한다.','',
            '| 예산 | 경로 | 후반 DINO | 후반 내부 flow px | 후반 경계 flow px | 후반 내부 warp | 후반 경계 warp |',
            '|---|---|---:|---:|---:|---:|---:|']
    for bits,arms in [(0,('bf16',))]+[(b,ARMS) for b in (2,3,4)]:
        for arm in arms:
            g=[r for r in rows if r['bits']==bits and r['arm']==arm]
            if not g:continue
            lines.append(f'| {bits or "BF16"} | {LABELS[arm]} | '+' | '.join(fmt(mean(r[k] for r in g),5 if 'warp' in k else 3) for k in ('late_dino_subject','late_inside_flow_pixels','late_boundary_flow_pixels','late_inside_warp_error','late_boundary_warp_error'))+' |')
    lines+=['','## 비용과 오차를 읽는 방법','',
            '- 표의 생성 시간(generation_seconds)은 CUDA 동기화한 실제 run_chunk wall time의 합이다. 모델 로딩·영상 디코딩·지표 평가·바깥 CPU latent 오차 계산/상태 IO는 제외한다. sampling_seconds는 그 바깥 계측까지 포함한 경과 시간으로 CSV에 별도 보존한다. ChunkStore 내부 저장 오차 계산은 생성 경로에 포함된다. 단일 실행의 개발 타이밍이며 반복/순서 교차 없는 결과를 최적화 kernel 성능으로 일반화하지 않는다.',
            '- encode/restore CUDA event 합은 저장/복원 비용을 보여준다. ChunkStore의 저장 오차 기록용 복원도 포함되므로 순수 필수 attention 비용만이라고 부르지 않는다. 전체 wall과 합계가 정확히 같아야 하는 지표가 아니다.',
            '- 실제 cache bytes는 K/V/pending을 나눈다. PyTorch peak allocated/reserved에는 모델·FP32 TT분해·전체 복원 작업 버퍼가 들어간다. cache 압축률과 peak 감소율은 다르다. 품질평가 peak는 별도 열이다.',
            '- history V relative L2는 각 FR 실행의 삽입 당시 BF16 V 기준이다. BF16 기준 생성의 동일 V와 비교한 값이 아니다. latent 오차와도 다른 지표다.',
            '- pixel MSE/PSNR은 같은 난수의 BF16 영상과 비교한 보조 거리다. 궤적이 달라져도 보기 좋은 영상일 수 있으므로 MUSIQ·움직임·육안 영상과 함께 판단한다.',
            '',
            '## 산출물과 검증','',
            '[영상별 전체 수치](summary.csv) · [프롬프트별 직접 양자화 대비 차이](contrasts.csv) · [검증 기록](audit.json) · [사전 개발 계획](../TT_VIDEO_DEVELOPMENT.md).',
            '',
            '잠재텐서와 프리뷰는 로컬 `~/sfq_data/tt_video/p{prompt}_b{bits}_{arm}/`의 `latent.pt`, `preview.mp4`에 보존한다. 지표는 MP4에서 측정하지 않았다. 원시 프레임별 MUSIQ·오차 및 RAFT 쌍은 각 `quality.json`, 청크별 시간/오차/캐시는 `sampling.json`에 있다.',
            '',
            '보고서 검증은 저장 latent의 CPU FP64 오차 재계산, 프레임/쌍 선택과 요약 재계산, source/input/artifact SHA 확인이다. 동일 작성자의 재검산이며 외부 독립 재현이나 모델 재실행을 의미하지 않는다.']
    return '\n'.join(lines)+'\n'


def export_plots(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'tt':'#2b8cbe','quant':'#444444','quant_tt':'#e34a33','tt_quant':'#31a354'}
    fig,axs=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for ax,bits in zip(axs,(2,3,4)):
        for arm in ARMS:
            g=[r for r in rows if r['bits']==bits and r['arm']==arm]
            if not g:continue
            x=mean(r['generation_seconds'] for r in g);y=mean(r['late_musiq'] for r in g)
            size=mean(r['cache_total_bytes'] for r in g)/2**30
            ax.scatter(x,y,s=70*size,color=colors[arm],alpha=.8,label=arm)
            ax.annotate(f'{arm}\n{size:.2f} GiB',(x,y),xytext=(4,5),textcoords='offset points',fontsize=7)
        bf=[r for r in rows if r['arm']=='bf16']
        if bf:ax.axhline(mean(r['late_musiq'] for r in bf),ls='--',lw=1,color='#777777',label='BF16 MUSIQ')
        ax.set_title(f'{bits}-bit V storage budget');ax.set_xlabel('Synchronized generation seconds');ax.set_ylabel('Late raw-frame MUSIQ');ax.grid(alpha=.2)
    fig.suptitle('TT video development: quality / time / actual cache bytes\nTwo prompts, one generation seed; K and pending V remain BF16',fontsize=11)
    fig.savefig(out/'quality_time_memory.png',dpi=180);fig.savefig(out/'quality_time_memory.svg');plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=DEFAULT_DATA);ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--partial',action='store_true');ap.add_argument('--no-plots',action='store_true');args=ap.parse_args()
    manifest=read(args.data/'manifest.json')
    expected_cases={f'p{p}_b{b}_{a}' for p,b,a in keys()}
    if set(manifest['cases'])!=expected_cases:raise RuntimeError('manifest does not describe planned 26 cases')
    gates=read(args.data/'gates.json')
    if not gates['pass_all'] or gates['source_sha256']!=manifest['source_sha256']:raise RuntimeError('model gates missing or mismatched')
    identity=[g for g in gates['records'] if g['arm']=='identity']
    if len(identity)!=2 or {g['prompt'] for g in identity}!={0,4}:raise RuntimeError('both identity gates required')
    for g in identity:
        if len(g['chunks'])!=8 or g['latent_nmse_whole']!=0:raise RuntimeError('identity gate not exact')
    for path,h in manifest['source_files'].items():
        if sha(ROOT/path)!=h:raise RuntimeError(f'source changed since sampling: {path}')
    import torch
    references={};rows=[];audits=[];missing=[]
    for p,bits,arm in keys():
        folder=args.data/f'p{p}_b{bits}_{arm}'
        if not all((folder/f).exists() for f in ('sampling.json','quality.json','latent.pt','preview.mp4')):
            missing.append(folder.name);continue
        if p not in references:
            references[p]=torch.load(args.data/f'p{p}_b0_bf16/latent.pt',map_location='cpu',weights_only=True)['latent']
        row,record=audit_case(folder,manifest,references[p]);rows.append(row);audits.append(record)
    if missing and not args.partial:raise RuntimeError(f'final report requires all 26 cases; missing {missing}')
    complete=not missing
    if complete:
        quality_signatures={r['quality_source_sha256'] for r in audits}
        if len(quality_signatures)!=1:raise RuntimeError('mixed quality evaluator signatures')
    contrasts=paired_contrasts(rows);args.out.mkdir(parents=True,exist_ok=True)
    if args.partial:
        write_csv(args.out/'progress_summary.csv',rows)
        (args.out/'PROGRESS_KO.md').write_text(report(rows,contrasts,False))
    else:
        write_csv(args.out/'summary.csv',rows);write_csv(args.out/'contrasts.csv',contrasts)
        (args.out/'REPORT_KO.md').write_text(report(rows,contrasts,True))
        audit=dict(complete=True,cases=26,source_sha256=manifest['source_sha256'],manifest_sha256=sha(args.data/'manifest.json'),
                   model_gate_sha256=sha(args.data/'gates.json'),actual_artifact_hashes_verified=True,cpu_fp64_latent_recomputed=True,quality_scalar_summaries_recomputed=True,
                   external_independent=False,records=audits)
        (args.out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
        if not args.no_plots:
            export_plots(rows,args.out)
            with (args.out/'REPORT_KO.md').open('a') as f:f.write('\n![품질·시간·캐시 비교](quality_time_memory.png)\n')
    print(json.dumps(dict(complete=complete,cases=len(rows),missing=missing,out=str(args.out)),ensure_ascii=False))

if __name__=='__main__':main()
