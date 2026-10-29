"""Audit and summarize six-codec development sweep; no re-execution."""
import csv,json,hashlib,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'src/experiments/codec_order_development'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 e=json.loads((OUT/'evidence.json').read_text());rows=list(csv.DictReader((OUT/'rows.csv').open()))
 assert len(rows)==450 and sha(OUT/'rows.csv')==e['rows_sha256']
 assert len({tuple(r[k] for k in ('prompt','layer','bits','method','order')) for r in rows})==450
 for p,h in e['source_sha256'].items():assert sha(ROOT/p)==h
 for a in e['input_artifacts']:assert sha(Path.home()/'sfq_data/postwo_rounding'/a['file'])==a['sha256']
 for a in e['payload_artifacts']:assert sha(Path.home()/'sfq_data/codec_order'/a['file'])==a['sha256']
 assert len(e['payload_artifacts'])==432
 for r in rows:
  assert int(r['payload_bytes'])<=int(r['budget_bytes']) and math.isfinite(float(r['D'])) and float(r['cpu_gpu_rel'])<3e-5
  if r['method']=='plain':assert float(r['added_output_D'])==0 and float(r['D_ratio'])==1
 methods=('tarp','tsvd','pivot_qr','range','tucker','tt')
 orders=('method_only','quant_then_method','method_then_quant_rank','method_then_quant_budget')
 summaries=[]
 for b in (2,3,4):
  for m in methods:
   for o in orders:
    for p in (0,4):
     rr=[r for r in rows if int(r['bits'])==b and r['method']==m and r['order']==o and int(r['prompt'])==p]
     assert len(rr)==3
     avg=lambda k:statistics.mean(float(r[k]) for r in rr)
     summaries.append(dict(bits=b,method=m,order=o,prompt=p,D=avg('D'),plain_D=avg('plain_D'),ratio=avg('D')/avg('plain_D'),payload_ratio=avg('payload_ratio'),payload_bytes=int(avg('payload_bytes')),rank=int(avg('rank')),v_rel=avg('v_rel'),added_D=avg('added_output_D'),encode_ms=1000*avg('encode_seconds'),decode_ms=1000*avg('decode_seconds')))
 with (OUT/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows(summaries)
 def group(b,m,o):return [s for s in summaries if s['bits']==b and s['method']==m and s['order']==o]
 def interval(g,k='ratio'):return f'{min(x[k] for x in g):.3f}–{max(x[k] for x in g):.3f}'
 quality=[];compression=[]
 for b in (2,3,4):
  for m in methods:
   for o in orders:
    g=group(b,m,o)
    if all(s['ratio']<=.9 for s in g):quality.append(dict(bits=b,method=m,order=o))
    if all(s['ratio']<=1.1 and s['payload_ratio']<=.5 for s in g):compression.append(dict(bits=b,method=m,order=o))
 combined=[s for s in quality+compression if s['order']!='method_only']
 decision=dict(quality_candidates=quality,compression_candidates=compression,quantization_combination_candidates=combined,proceed_method2=not bool(combined))
 (OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 t=['# 6개 방법 × 양자화 순서 × 2·3·4비트','',
 '2026-10-30. 개발 검사 450행 완료. 원본 입력 방법 6개, 4개 저장/순서 조건, 직접 RTN 기준선.','',
 '## 결과와 다음 단계','',
 '요청한 양자화 결합 경로에서는 두 프롬프트 모두를 만족하는 품질 개선 또는 절반 용량/오차 유지 후보가 없었다. 사용자 지시에 따라 결합 변형을 더 늘리지 않고 방법 2로 진행한다.', '',
 '단, 분해만 하고 FP16 인자로 저장한 2비트 예산에서는 Tucker와 TT가 양쪽 프롬프트에서 직접 RTN과 t-SVD보다 오차가 작았다. 이 관찰은 버리지 않고 기존 분해 활용 후보로 보존한다. 양자화 결합의 성공이나 새로운 알고리즘 기여는 아니다.', '',
 '## 범위와 표현','',
 '기존 개발 p0/p4 seed101, 층0/14/29, chunk6 V, 미래 chunk7 query32. BF16 K 고정. 두 창에서 새 V 오차의 post-Wo squared contribution D를 평균하고 3층 평균한다. 전체 출력/긴 생성 품질이 아니다. 미래 query는 fitting에 사용하지 않는다. 선택 시드9101 하나(이전 5시드 실험과 구별).', '',
 'T-ARP는 공통 열 선택+최소제곱, t-SVD는 Fourier 저랭크, pivot_qr은 펼친 행렬의 피벗 QR, range는 Gaussian range+power1, Tucker는 HOSVD, TT는 공간→시간→채널 TT-SVD다. 모두 기존 방법이며 신규성 주장 없음.', '',
 '방법→양자화는 실제 인자 2/3/4비트 패킹이다. 같은 rank에서는 공간이 크게 줄어들고, 같은 예산에서는 rank를 높인다. 인자 양자화는 연속 16원소 블록 RTN이며 원래 V의 토큰축 g16과 축이 다르다. Gauge/scale balancing 최적화 없음. 이번 T-ARP의 fit은 FP32 열 기준이므로 과거 FP16 열 기준 버전과 비트 일치 요구하지 않음.', '',
 '## 직접 양자화 D=1인 최종 오차 비율','',
 '각 구간은 두 프롬프트의 값이며 신뢰구간이 아니다. 낮을수록 좋다. 높은 비트의 큰 비율은 직접 양자화 기준 오차가 작아지기 때문이며 영상 품질 배수가 아니다.','',
 '| 비트 | 방법 | 방법만(FP16) | 양자화→방법 | 방법→양자화 같은 rank | 방법→양자화 같은 예산 |','|---|---|---:|---:|---:|---:|']
 for b in (2,3,4):
  for m in methods:t.append(f'| {b} | {m} | '+' | '.join(interval(group(b,m,o)) for o in orders)+' |')
 t+=['','## 실제 저장량과 rank','',
 '한 층 V tensor payload: 인자 코드/scale/index 포함. 원본·중간/전체 복원 버퍼·파일 header·allocator는 별도. 디스크 artifact 크기와 GPU 상주 payload를 혼동하지 않는다. 정수 rank/표현 상한 때문에 남는 예산은 쓰지 않았다.','',
 '| 비트 | 방법 | FP16 rank/bytes | 같은 rank 양자화 bytes | 같은 예산 양자화 rank/bytes |','|---|---|---:|---:|---:|']
 for b in (2,3,4):
  for m in methods:
   a=group(b,m,'method_only')[0];c=group(b,m,'method_then_quant_rank')[0];d=group(b,m,'method_then_quant_budget')[0]
   t.append(f'| {b} | {m} | {a["rank"]} / {a["payload_bytes"]:,} | {c["payload_bytes"]:,} | {d["rank"]} / {d["payload_bytes"]:,} |')
 t+=['','RTN 기준 bytes: 2비트 2,697,216 / 3비트 3,595,776 / 4비트 4,494,336.', '',
 '## 비용과 검증','',
 'summary.csv에 모든 조건의 encode/decode 시간을 기록했다. Encode는 선양자화/복원·분해·인자 저장 비용을 포함한다. 분해는 첫 실행 시간, 공유 계산해도 각 행에 비용을 청구했다. Decode는 warmup1+3회 median. 배포 속도 순위를 확정하는 벤치마크가 아니다.', '',
 f'전체 실행 {e["seconds"]:.2f}초; 진단 peak allocated {e["peak_allocated_bytes"]/2**30:.3f} GiB, reserved {e["peak_reserved_bytes"]/2**30:.3f} GiB. 모델 실행 없이 저장된 BF16 입력 사용.', '',
 'Synthetic fullrank matrix/lowrank tensor/scalar bitpack/actual byte rank 검사4개 통과. 450행 finite·예산, 432상태 CPU/GPU 복원, source/input/payload SHA 검증. 직접 RTN 기준값은 이전 결과와 일치. 같은 작성자의 검증이며 독립 외부 감사 아님.', '',
 '[계획·출처](../CODEC_ORDER_DEVELOPMENT.md) · [원시 행](rows.csv) · [프롬프트별 요약](summary.csv) · [진행 판단](decision.json).']
 (OUT/'REPORT_KO.md').write_text('\n'.join(t)+'\n')
 (OUT/'audit.json').write_text(json.dumps(dict(rows=450,states=432,hashes_pass=True,budget_pass=True,cpu_decode_pass=True,plain_identity_pass=True,external_independent=False),indent=2)+'\n')
 print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
