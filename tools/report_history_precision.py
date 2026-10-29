"""Recompute saved latent errors and audit history-schedule comparisons."""
import csv,json,hashlib,statistics,math
from pathlib import Path
import torch
from history_precision_core import schedules,POLICIES
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'src/experiments/history_precision_development';DATA=Path.home()/'sfq_data/history_precision'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 e=json.loads((OUT/'evidence.json').read_text());rows=list(csv.DictReader((OUT/'rows.csv').open()))
 assert len(rows)==26 and sha(OUT/'rows.csv')==e['rows_sha256']
 for p,h in e['source_sha256'].items():assert sha(ROOT/p)==h
 for a in e['reference_files']:assert sha(Path.home()/'sfq_data'/a['file'])==a['sha256']
 delta=0.
 for row in rows:
  p=int(row['prompt']);seed=int(row['selection_seed']);policy=row['policy']
  path=DATA/f'p{p}_{policy}_s{seed}.pt';record=next(a for a in e['artifacts'] if a['file']==path.name)
  assert sha(path)==record['sha256']
  obj=torch.load(path,map_location='cpu',weights_only=True)
  expect=torch.zeros(4,4680,dtype=torch.bool) if policy=='low_only' else schedules(seed=seed)[policy]
  assert torch.equal(obj['mask'],expect)
  # CPU FP64 accumulation, independent of measured GPU FP32 norm reductions.
  ref=obj['reference'].double();error=obj['latent'].double()-ref
  nmse=float(error.square().mean()/ref.square().mean())
  relative=abs(nmse-float(row['latent_nmse']))/max(nmse,1e-30)
  delta=max(delta,relative);assert relative<2e-5 and math.isfinite(nmse)
 summary=[]
 for p in (0,4):
  mean={policy:statistics.mean(float(r['latent_nmse']) for r in rows if int(r['prompt'])==p and r['policy']==policy) for policy in ('low_only',)+POLICIES}
  summary.append(dict(prompt=p,means=mean,banlast_over_independent=mean['banlast1']/mean['independent'],roundrobin_over_independent=mean['roundrobin']/mean['independent'],fixed_over_independent=mean['fixed']/mean['independent']))
 gate=all(s['banlast_over_independent']<=.9 for s in summary)
 (OUT/'decision.json').write_text(json.dumps(dict(primary_gate=gate,summary=summary,scope='one actual target chunk; three layers; one past V chunk; fixed BF16 prefix'),indent=2)+'\n')
 text=['# 방법 2: 이력 기반 정밀 읽기 배분 — 실제 한 청크 생성','',
 '2026-10-30. 같은 과거 V를 읽을 때 호출별 BF16 위치 배분을 바꾸었다. 교란 출력 합의 대용 지표가 아니라 실제 4회 denoise를 거친 최종 latent를 측정했다.', '',
 '## 결론','',
 'BanLast의 최종 latent NMSE는 independent보다 p0에서3.18%, p4에서9.38% 높았다. 사전10%개선 기준을 통과하지 못했다. Round-robin은 p0에서1.32% 높고 p4에서0.76% 낮아 일관된 이점이 없었다. 이 범위에서는 이력 배분의 추가 이득을 확인하지 못했으며, 대규모 확장 근거로 사용하지 않는다.', '',
 '고정밀25%를 읽는 것 자체는 INT2-only보다 오차를 줄였다. 그러나 이는 추가 정밀도 예산의 효과이며 이력 정책의 신규 이득과 구분한다. 현재3층·한 과거 청크·25% 조건 밖의 결과를 단정하지 않는다.', '',
 '## 조건','',
 'BF16 prefix chunks0..6 뒤 target chunk7. 개발 p0/p4, 생성 seed101. 층0/14/29의 직전 chunk6 V만 INT2, K·다른 V는 BF16. 각 호출1170/4680토큰(25%)을 BF16로 읽고 나머지는 고정 INT2 복원값으로 읽는다. 마스크는 head/channel/layer 공통. 선택seed9101/9102/9103, 4정책*3seed*2prompt=24개 target 실행 +2개 low-only baseline.', '',
 '- fixed: 같은25%를 네 호출에서 유지. independent: 매 호출 새25%.',
 '- banlast1: 바로 앞 호출의 선택을 제외하고25% 선택. roundrobin: 네 호출에 중복 없이 한 바퀴.',
 '- 첫 마스크는 같은 선택seed끼리 일치. future query/latent로 마스크를 선택하지 않음.', '',
 '## 결과','',f'사전에 정한 primary 조건(양 프롬프트 모두 BanLast 평균 NMSE가 independent보다10% 감소): **{gate}**.', '',
 'NMSE는 ||latent−BF16||² / ||BF16||². 선택seed3개의 평균. 낮을수록 좋으며 영상 품질 점수가 아니다.', '',
 '| 프롬프트 | INT2만 | fixed | independent | banlast1 | roundrobin | BanLast/independent | roundrobin/independent |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for s in summary:
  m=s['means'];text.append(f'| p{s["prompt"]} | '+' | '.join(f'{m[a]:.8g}' for a in ('low_only',)+POLICIES)+f' | {s["banlast_over_independent"]:.4f} | {s["roundrobin_over_independent"]:.4f} |')
 text+=['','각 선택seed를 개별로 공개한다. 생성 seed가3개라는 뜻이 아니다.','', '| 프롬프트 | seed | fixed NMSE | independent NMSE | banlast1 NMSE | roundrobin NMSE |','|---|---|---:|---:|---:|---:|']
 for p in (0,4):
  for seed in (9101,9102,9103):
   values={a:next(float(r['latent_nmse']) for r in rows if int(r['prompt'])==p and int(r['selection_seed'])==seed and r['policy']==a) for a in POLICIES}
   text.append(f'| p{p} | {seed} | '+' | '.join(f'{values[a]:.8g}' for a in POLICIES)+' |')
 mem=e['memory'][0]
 text+=['','## 비용·검증·한계','',
 f'- 대상3층의 low-bit 코드+scale {mem["low_code_scale_bytes"]:,}bytes에 **전체 BF16 참조 {mem["full_bf16_reference_bytes"]:,}bytes**를 함께 보관했다. 다음 호출에 어느 위치든 선택할 수 있기 때문이다. 25% 읽는다고 저장량이25%가 되는 것은 아니다.',
 '- 실제 구현은 BF16 attention 입력을 clone/교체한다. selective fused kernel이나 저장 절감/대역폭 절감 성능을 증명하지 않는다. 정책끼리는 같은 저장·논리적 읽기 예산이다.',
 f'- 각 target 샘플링 평균 {statistics.mean(float(r["sampling_seconds"]) for r in rows):.2f}s, 전체 {e["seconds"]:.2f}s. peak allocated {e["peak_allocated_bytes"]/2**30:.3f}GiB, reserved {e["peak_reserved_bytes"]/2**30:.3f}GiB. 진단/원본 캐시 포함, 배포 footprint 아님.',
 '- 양 프롬프트의 prefix7개·target1개가 기존 BF16 latent와 정확히 일치. all-high actual hook도 정확히 일치. 3층*4호출 위치/개수·캐시 불변·mask budget 검증.',
 f'- 저장 latent26개를 CPU FP64로 재계산했고 GPU 결과와 최대 상대차 {delta:.3g}. 입력/소스/결과 SHA 및 마스크 재현 확인. 같은 작성자의 재검산이며 외부 독립 검증 아님.',
 '- 초기 잘못된 작업 디렉터리에서 모델 경로 탐색이 실패하여 src/에서 다시 실행했다. 그 실행에서는 측정이 시작되지 않았고 실패 로그를 보존했다.',
 '- 한 청크·3층·한 과거 V 청크·INT2/25% 예산에 한정. 긴 FR·decoded 품질·다른 예산·모델로 일반화하지 않는다. 논문의 분산 최적화 정리를 생성기 성능 보장으로 사용하지 않는다.', '',
 '[계획 및 출처](../HISTORY_PRECISION_DEVELOPMENT.md) · [원시 행](rows.csv) · [검증](audit.json).']
 (OUT/'REPORT_KO.md').write_text('\n'.join(text)+'\n')
 (OUT/'audit.json').write_text(json.dumps(dict(latents=26,all_source_reference_artifact_hashes_pass=True,mask_reproduction_pass=True,cpu_float64_recomputed=True,max_nmse_relative_difference=delta,model_gates=e['gates'],external_independent=False),indent=2)+'\n')
 print(json.dumps(dict(primary_gate=gate,summary=summary),indent=2))
if __name__=='__main__':main()
