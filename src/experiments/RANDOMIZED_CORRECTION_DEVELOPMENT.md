# 방법 2 개발 진단: 무작위 고정밀 보정

2026-10-30. 기존 개발 자료 재사용이며 신규성·확증 실험 등록이 아니다.
측정 전에 고정하는 범위: p0/p4, seed101, layer0/14/29,
chunk6 refresh의 V 양자화 오차, chunk7 첫 호출의 32 query.
K는 BF16으로 고정하고 chunk6 V의 추가 오차만 모델링한다.
전체 attention·생성·실제 속도·메모리 절감 실험이 아니다.

## 출처와 신규성

Rhee–Glynn, https://web.stanford.edu/~glynn/papers/2015/RheeG15.html
및 Beatson–Adams, https://proceedings.mlr.press/v97/beatson19a.html
의 불편 추정·무작위 telescoping 원리는 기존 방법이다.
KV 추론에서 동일 구성을 사용한 선행연구를 아직 확인하지 못했을 뿐,
사용되지 않았다고 보증하거나 새 알고리즘이라고 주장하지 않는다.

## 진단과 보완

고정 입력에서 d=O_L-O_H, I~Bernoulli(p), 출력 오차는 (1-gamma I/p)d.
편향 제곱=(1-gamma)^2||d||², 분산=gamma²(1/p-1)||d||².
MSE=(1-2gamma+gamma²/p)||d||².
gamma=1은 불편 보정, gamma=p는 고정밀 단순 교체이다.
같은 p에서 후자가 전자보다 항상 작거나 같은 MSE를 갖는다.
MSE 최소 gamma=p이므로 shrinkage만으로 새 방법이 되는 것은 아니다.

1. 균일 p=0.1/0.25/0.5/0.75, gamma=1과 단순 교체, gain<=2 보완.
2. 보정 확률 보완: chunk6 query별 잔차 norm으로 p를 할당.
   p∈[.02,1], 평균 p 동일. 현재 query 위치를 다음 청크에 대응하는
   개발 proxy이며 미래 query는 사용하지 않는다. 보조 BF16 계산 비용은 별도다.
3. 미래 query 잔차를 아는 oracle 할당은 진단 상한으로만 표시.
4. 두 Bernoulli 결과를 정확히 열거하여 기대 MSE/편향/분산 검증.
   무작위 반복 표본의 오차를 수학적 기대값으로 오인하지 않는다.

결과는 원인과 다음 보완 조건으로 정리한다. 단일 local MSE의 불리함으로
긴 생성에서의 모든 가능성을 부정하지 않으며, 평균 불편성으로 생성 품질을
보장하지도 않는다. 고정밀 캐시 비용을 제외한 query 선택률은 속도나
메모리 이득이 아니다. 생성 실험 전에는 잔차 예측·분산과 고정밀 정보 저장
문제에 대한 추가 설계가 필요하다.
