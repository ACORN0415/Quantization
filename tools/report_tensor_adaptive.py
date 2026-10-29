"""Audit saved evidence and generate the Korean development report."""
import csv,json,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'src/experiments/tensor_adaptive_development'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 e=json.loads((OUT/'evidence.json').read_text());rows=list(csv.DictReader((OUT/'rows.csv').open()))
 assert sha(OUT/'rows.csv')==e['rows_sha256'] and len(rows)==54
 for p,s in e['source_sha256'].items():assert sha(ROOT/p)==s
 for a in e['payload_artifacts']:assert sha(Path.home()/'sfq_data/tensor_adaptive'/a['file'])==a['sha256']
 for r in rows:
  assert int(r['payload_bytes'])<=int(r['budget_bytes']) and float(r['cpu_gpu_rel'])<2e-5
  if r['max_selected_residual']:assert float(r['max_selected_residual'])<1e-10
 summaries=[]
 for p in (0,4):
  means={a:statistics.mean(float(r['D']) for r in rows if int(r['prompt'])==p and r['arm']==a) for a in sorted({r['arm'] for r in rows})}
  avg=statistics.mean(means[a] for a in means if a.startswith('arp'))
  old=next(s for s in e['summary'] if s['prompt']==p)
  assert abs(avg-old['arp_mean'])<1e-12
  summaries.append((p,means,avg))
 text=['# 방법 1: 적응적 공통 열 선택 결과','',
 '2026-10-30. 기존 개발 데이터 p0/p4, seed101, 층0/14/29. 새 보류/생성 실험 아님.', '',
 '## 판단', '',
 f'사전 개발 기준 결과: RTN 대비 10% 개선 gate **{e["rtn_gate"]}**, SVD 대비 5% 개선 gate **{e["svd_gate"]}**. 랜덤 선택 시드 5개의 평균으로 판단했으며 좋은 시드만 선택하지 않았다.', '',
 '이번에는 T-ARP Algorithm 8의 공통 열 선택을 구현했다. 복원은 FP16 선택 열과 최소제곱 계수를 사용하는 방식이며, 논문의 행·열 동시 선택 Algorithm 9 전체 CUR 재현은 아니다. 결정론적 adaptive argmax도 별도 측정했지만 이는 Osinsky 알고리즘이 아니다.', '',
 '## 미래 query 출력 오차 D', '',
 'V만 압축하며 BF16 K를 고정. 두 창에서 새 chunk6 V 오차의 post-Wo 기여를 평균하고 3개 층을 평균한다. 미래 chunk7 query32는 평가에만 사용한다. 낮을수록 좋다.', '',
 '| 프롬프트 | RTN2 | 기존 공통 열 | SVD | 적응적 argmax | T-ARP 5시드 평균 | T-ARP / RTN | T-ARP / SVD |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
 for p,m,avg in summaries:text.append(f'| p{p} | {m["rtn"]:.7f} | {m["static"]:.7f} | {m["svd"]:.7f} | {m["greedy"]:.7f} | {avg:.7f} | {avg/m["rtn"]:.3f} | {avg/m["svd"]:.3f} |')
 text+=['','선택 시드별 층 평균 D (전부 공개):','', '| 프롬프트 | 9101 | 9102 | 9103 | 9104 | 9105 |','|---|---:|---:|---:|---:|---:|']
 for p,m,avg in summaries:text.append(f'| p{p} | '+' | '.join(f'{m[f"arp{s}"]:.7f}' for s in range(9101,9106))+' |')
 text+=['','## 실제 payload와 시간','',
 'rank22. 한 층 V의 tensor storage 바이트이며 직렬화 header/allocator 및 복원 workspace와 구별한다. 양자화·근사 결과를 FP32로 전체 복원하는 진단이므로 배포 메모리 이득을 주장하지 않는다. 시간은 CUDA 동기화, warmup1회 후 3회 median을 각 셀에서 측정하고 아래는 6셀 평균이다. 인코딩에 SVD/선택/계수 계산을 포함했다. 순서 고정·단일 GPU 개발 측정이다.', '',
 '| 방식 | payload bytes | encode ms | decode ms | 평균 V 상대 L2 |','|---|---:|---:|---:|---:|']
 for name in ('rtn','static','svd','greedy','arp'):
  rr=[r for r in rows if (r['arm'].startswith('arp') if name=='arp' else r['arm']==name)]
  text.append(f'| {name} | {rr[0]["payload_bytes"]} | {statistics.mean(float(r["encode_seconds"]) for r in rr)*1000:.3f} | {statistics.mean(float(r["decode_seconds"]) for r in rr)*1000:.3f} | {statistics.mean(float(r["v_rel"]) for r in rr):.4f} |')
 text+=['','## 검증 및 한계','',
 f'- Synthetic projector/재현성, full-rank 복원, 공통 low-rank 복원 3개 검사 통과. 54행 finite/예산 준수, 48개 압축 payload SHA/CPU 복원 검사 통과. 기존 3개 기준선 최대 D 차이 {e["previous_max_abs_D_delta"]:.3g}. 같은 작성자의 재계산이며 독립 검증 아님.',
 '- 실제 선택 경로에서 주파수별 rank 증가와 선택 행 annihilation 확인. 전체 frequency-alignment 가정이나 기대오차 정리를 실제 데이터에서 증명한 것은 아니다.',
 f'- 전체 측정 구간 {e["seconds"]:.2f}초. peak allocated {e["peak_allocated"]/2**30:.3f} GiB, reserved {e["peak_reserved"]/2**30:.3f} GiB. import/모델 생성 제외, 기존 저장 텐서 사용.',
 '- 시드 5개는 선택 알고리즘의 변동이며 생성 시드/독립 프롬프트 수를 늘린 것이 아니다. n=2 개발 관찰에 유의성/일반화 주장을 붙이지 않는다.',
 '- SVD의 tensor-MSE 최적성은 출력 D 최적성을 뜻하지 않는다. 그래도 동일 예산의 실제 기준선에 비교 이점이 있어야 새 방법을 진행할 근거가 된다.', '',
 '## 다음 판단','',
 'RTN과 SVD에 대한 두 gate가 모두 실패했고 SVD보다 encode도 느렸다. 현재 공통 열 표현은 보류하며 전체 생성 실험으로 확대하지 않는다. 이 판단은 rank22·FP16 인자·현재 선택/복원 구성에 한정한다. SVD의 2비트 예산 가능성은 기존 방법의 활용 후보로 남기되 새 기여라고 부르지 않는다. 방법 2는 별도 계획과 인과적 읽기 기록이 필요하며 이 실험에 섞지 않았다.', '',
 '[개발 계획](../TENSOR_ADAPTIVE_DEVELOPMENT.md) · [원시 행](rows.csv) · [근거/해시](evidence.json) · [원 논문](https://arxiv.org/html/2606.26688v2)']
 (OUT/'REPORT_KO.md').write_text('\n'.join(text)+'\n')
 (OUT/'audit.json').write_text(json.dumps(dict(rows=54,payload_artifacts=48,hashes_pass=True,budgets_pass=True,selected_annihilation_pass=True,summary_recomputed=True,external_independent=False),indent=2)+'\n')
 print(json.dumps(e['summary'],indent=2))
if __name__=='__main__':main()
