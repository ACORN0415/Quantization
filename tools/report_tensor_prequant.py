"""Read-only evidence audit plus combined 2/3/4-bit Korean reports."""
import csv,json,hashlib,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'src/experiments'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(folder,data):
 e=json.loads((folder/'evidence.json').read_text());r=list(csv.DictReader((folder/'rows.csv').open()))
 assert len(r)==54 and sha(folder/'rows.csv')==e['rows_sha256']
 assert len({(x['prompt'],x['layer'],x['arm']) for x in r})==54
 for p,h in e['source_sha256'].items():assert sha(ROOT/p)==h
 for a in e['input_artifacts']:assert sha(Path.home()/'sfq_data/postwo_rounding'/a['file'])==a['sha256']
 for a in e['payload_artifacts']:assert sha(data/a['file'])==a['sha256']
 assert len(e['payload_artifacts'])==48
 for x in r:
  assert int(x['payload_bytes'])<=int(x['budget_bytes'])
  assert math.isfinite(float(x['D'])) and float(x['cpu_gpu_rel'])<2e-5
  if 'added_output_D' in x:
   assert math.isfinite(float(x['added_output_D'])) and float(x['added_output_D'])>=0
   assert abs(float(x['D'])/float(x['baseline_D'])-float(x['D_over_plain']))<1e-9
   if x['arm']=='rtn':assert float(x['added_output_D'])==0 and float(x['added_v_rel'])==0
 return r,e

def get(rows,p,arm):return [r for r in rows if int(r['prompt'])==p and (r['arm'].startswith('arp') if arm=='arp' else r['arm']==arm)]
def avg(rr,key):return statistics.mean(float(r[key]) for r in rr)

def main():
 pre={};orig={};evidence={}
 for b in (2,3,4):
  pre[b],evidence[b]=audit(EXP/f'tensor_prequant_b{b}',Path.home()/f'sfq_data/tensor_prequant_b{b}')
  name='tensor_adaptive_development' if b==2 else f'tensor_adaptive_b{b}'
  data='tensor_adaptive' if b==2 else f'tensor_adaptive_b{b}'
  orig[b],_=audit(EXP/name,Path.home()/'sfq_data'/data)
 out=EXP/'tensor_prequant_comparison';out.mkdir(exist_ok=True)
 allrows=[r for b in (2,3,4) for r in pre[b]]
 with (out/'rows.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
 text=['# 선양자화 후 방법 적용: 2·3·4비트 비교','',
 '2026-10-30. 사용자 요청: 원본 → 양자화 → 방법과 원본 → 양자화만 비교.', '',
 '## 결론','',
 '2·3·4비트 모두 선양자화 후 방법을 적용하면 양자화만 한 것보다 최종 출력 오차가 증가했다. T-ARP는 2비트에서 2.45–2.54배, 3비트에서 14.43–16.05배, 4비트에서 57.33–69.88배였다. SVD도 모든 비트에서 기준선보다 나빴다. 최종 payload 절감은 0.83–2.61%에 그쳐 현재 구성의 추가 손상을 정당화할 저장 이득은 보이지 않는다.', '',
 '4비트의 큰 비율은 최종 절대 오차가 2비트보다 커졌다는 뜻이 아니다. 직접 양자화의 오차는 비트 증가에 따라 빠르게 줄지만 저랭크 근사의 오차는 덜 줄어 비율이 커진다. 이는 국소 squared output error이며 영상 품질 배수가 아니다.', '',
 '## 조건','',
 '기존 개발 프롬프트 p0/p4, 생성 seed101, layers0/14/29. V-only, chunk6 압축, chunk7 첫 query32 평가. 원본 BF16 K 고정. 총 162행. 방법 입력은 RTN 복원값뿐이며 원본/미래 query는 fitting에 사용하지 않는다. 원본은 평가용이다.', '',
 '방법은 원래의 정수 코드를 무손실 압축하는 것이 아니라 복원 수치를 손실 근사한다. 최종 인자는 FP16이다. 2/3/4비트는 **앞단 RTN 비트 및 기준 저장 예산**을 뜻하며 tensor 인자 자체의 비트가 아니다. 각 예산의 최대 rank22/29/36을 사용한다.', '',
 '## 원본 대비 최종 출력 오차','',
 'D는 두 BF16 cache 창에서 새 V 오류의 post-Wo 기여 squared error 평균이며 3층 평균한다. T-ARP는 5개 선택 시드 평균이다. 추가 노름²와 최종 D 증감은 교차항 때문에 같지 않다.', '',
 '| 비트 | 프롬프트 | 양자화만 D | 양자화→T-ARP D | T-ARP / 양자화만 | 양자화→SVD D | SVD / 양자화만 |','|---|---|---:|---:|---:|---:|---:|']
 summaries=[]
 for b in (2,3,4):
  for p in (0,4):
   plain=avg(get(pre[b],p,'rtn'),'D');arp=avg(get(pre[b],p,'arp'),'D');svd=avg(get(pre[b],p,'svd'),'D')
   text.append(f'| {b} | p{p} | {plain:.7f} | {arp:.7f} | {arp/plain:.3f} | {svd:.7f} | {svd/plain:.3f} |')
   summaries.append(dict(bits=b,prompt=p,plain_D=plain,arp_D=arp,arp_ratio=arp/plain,svd_D=svd,svd_ratio=svd/plain))
 text+=['','## 방법이 양자화 결과에 추가한 변화','',
 'V 변화는 ||V_method−V_quant|| / ||V_quant||의 층·선택시드 평균이다. 추가 출력 D는 ||O_method−O_quant||²를 같은 방식으로 평균했다. 원본 대비 최종 오차와 구분한다.', '',
 '| 비트 | 프롬프트 | T-ARP V 변화 | T-ARP 추가 출력 D | SVD V 변화 | SVD 추가 출력 D |','|---|---|---:|---:|---:|---:|']
 for b in (2,3,4):
  for p in (0,4):
   a=get(pre[b],p,'arp');s=get(pre[b],p,'svd')
   text.append(f'| {b} | p{p} | {avg(a,"added_v_rel"):.4f} | {avg(a,"added_output_D"):.7f} | {avg(s,"added_v_rel"):.4f} | {avg(s,"added_output_D"):.7f} |')
 text+=['','## 다른 선택 방식 및 난수 변동','',
 '| 비트 | 프롬프트 | static / plain | adaptive argmax / plain | T-ARP 시드별 비율 범위 |','|---|---|---:|---:|---:|']
 for b in (2,3,4):
  for p in (0,4):
   plain=avg(get(pre[b],p,'rtn'),'D');ratios=[avg(get(pre[b],p,f'arp{s}'),'D')/plain for s in range(9101,9106)]
   text.append(f'| {b} | p{p} | {avg(get(pre[b],p,"static"),"D")/plain:.3f} | {avg(get(pre[b],p,"greedy"),"D")/plain:.3f} | {min(ratios):.3f}–{max(ratios):.3f} |')
 text+=['','## 저장량과 비용','',
 '한 층 V의 실제 tensor payload이다. 예산 이하의 최대 정수 rank라 바이트가 완전히 같지는 않다. 최종 상태는 원래 quantized stream과 함께 유지하지 않는다. 중간 복원/인코딩 workspace와 원본 평가 버퍼는 payload에 포함하지 않으며 GPU peak/배포 메모리와 같지 않다.', '',
 '| 비트 | RTN bytes | T-ARP bytes | SVD bytes | T-ARP 절감률 |','|---|---:|---:|---:|---:|']
 for b in (2,3,4):
  values=[int(get(pre[b],0,a)[0]['payload_bytes']) for a in ('rtn','arp','svd')]
  text.append(f'| {b} | {values[0]} | {values[1]} | {values[2]} | {(1-values[1]/values[0])*100:.3f}% |')
 text+=['','encode는 양자화+중간 복원+방법을 모두 포함한다. warmup1회 후 3회 median의 셀 평균; T-ARP는 시드도 평균. decode는 최종 표현만 복원한다. 최적화 커널/전체 모델 속도 주장이 아니다.','',
 '| 비트 | 방식 | encode ms | 최종 decode ms |','|---|---|---:|---:|']
 for b in (2,3,4):
  for a in ('rtn','static','svd','greedy','arp'):
   rr=get(pre[b],0,a)+get(pre[b],4,a)
   text.append(f'| {b} | {a} | {1000*avg(rr,"encode_seconds"):.3f} | {1000*avg(rr,"decode_seconds"):.3f} |')
 text+=['','## 선양자화 없이 방법을 적용했던 결과와 비교','',
 '두 경로 모두 최종 FP16 인자, 같은 rank·예산이다. 아래는 원본 대비 D / 직접 RTN D. 선양자화로 방법 입력이 바뀐 효과이며 인자 재양자화 비교가 아니다. 2비트 선양자화 없는 행은 동결 결과 재사용, 3/4비트는 이번에 추가 측정했다.','',
 '| 비트 | 프롬프트 | 원본→T-ARP | 양자화→T-ARP | 원본→SVD | 양자화→SVD |','|---|---|---:|---:|---:|---:|']
 for b in (2,3,4):
  for p in (0,4):
   plain=avg(get(pre[b],p,'rtn'),'D')
   text.append(f'| {b} | p{p} | {avg(get(orig[b],p,"arp"),"D")/plain:.3f} | {avg(get(pre[b],p,"arp"),"D")/plain:.3f} | {avg(get(orig[b],p,"svd"),"D")/plain:.3f} | {avg(get(pre[b],p,"svd"),"D")/plain:.3f} |')
 text+=['','## 검증と限界'.replace('と限界',' 및 한계'),'',
 '입력·코드·결과·payload SHA 검증. 162행/144개 payload의 CPU-GPU 복원 일치, 실제 예산 준수, 직접 양자화 기준값 재현. plain의 추가 오차는 정확히 0이다. 이는 같은 작성자의 재계산이며 독립 외부 검증은 아니다.', '',
 '기존 개발 표본 n=2, 생성 시드1개, query32개/3층의 국소 오차이다. 전체 영상 품질·FR 안정성·무손실성·일반 모델 결론을 주장하지 않는다. 이번에 추가 결과로 임계값을 재조정하지 않았다.', '',
 '[실험 계획](../TENSOR_PREQUANT_DEVELOPMENT.md) · [전체 원시 행](rows.csv). 원 알고리즘 출처/구현 차이는 [적응적 선택 계획](../TENSOR_ADAPTIVE_DEVELOPMENT.md)을 따른다.']
 (out/'REPORT_KO.md').write_text('\n'.join(text)+'\n')
 (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
 (out/'audit.json').write_text(json.dumps(dict(prequant_rows=162,raw_input_comparison_rows=162,payloads_verified=288,source_input_rows_payload_sha_pass=True,cpu_decode_checks_pass=True,plain_added_error_zero=True,rows_sha256=sha(out/'rows.csv'),inputs={f'prequant_b{b}':sha(EXP/f'tensor_prequant_b{b}/evidence.json') for b in (2,3,4)}),indent=2)+'\n')
 print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
