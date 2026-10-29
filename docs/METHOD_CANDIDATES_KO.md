# 후속 방법 후보 장부

갱신: 2026-10-30. 개발 아이디어 장부이며 실험 사전 등록이 아니다.
판단 기준: (1) 동일한 구성이 KV/attention 선행연구에 사용됐는가,
(2) 우리 설정에서 이득을 만들 구체적인 수학적·계산적 조건이 있는가.
검색에서 찾지 못한 것은 미사용 증명이 아니다. 아래 적용 설계는 우리의 가설이다.

## 보관 후보 (실행 순서는 방법 1 → 2 → 3 → 4)

### 방법 3: 압축 상태로 KV를 저장하고 계산 때 복원

공간 절감에는 타당하지만 CacheGen, SplitZip, ZipFusion과 큰 틀에서 중복.
실용 구현 후보이며 원안의 신규성은 낮다. BF16 원본을 GPU에 함께 유지하거나
전체 복원 버퍼를 상주시킨다면 공간 절감 주장을 별도로 제한해야 한다.

### 방법 4: 무작위 다단계 attention 보정

Rhee–Glynn 불편 추정 원리는 기존이다. 저비트 출력에 드문 고정밀 차이를
역확률 보정하는 정확한 KV 적용은 이번 조사에서 확인하지 못했다.
고정 입력 MSE는 (1-p)/p * ||O_H-O_L||²이며 단순 교체의
(1-p)*||O_H-O_L||²보다 크다. 분산·원본 저장·복원 비용 보완이 필요하다.
준비 코드/계획: tools/randomized_correction_probe.py,
src/experiments/RANDOMIZED_CORRECTION_DEVELOPMENT.md.
합성 대수 검사는 통과했으나 실제 데이터 진단 및 생성 실험은 미실행이다.

## 최근 수학 논문에서 가져온 잠정 후보

### 방법 1: 공통 인덱스 tensor cross 압축 (먼저 실행)

- 원 논문: Akdodshoev, Leplat, Ahmadi-Asl,
  Adaptive Randomized Pivoting for Tensor Cross Approximation in the T-Product Framework.
  2026-06-25 초고, 2026-09-07 v2, math.NA preprint.
  https://arxiv.org/abs/2606.26688v2
- 기반: Cortinovis–Kressner, Adaptive Randomized Pivoting for Column Subset
  Selection, DEIM, and Low-Rank Approximation. SIAM J. Matrix Anal. Appl.,
  2026-01 온라인 출판(초고 2024).
  https://doi.org/10.1137/24M1719189
- 가져올 것: 각 시간 조각을 따로 축소하지 않고, 완료된 청크의 공간×특징×시간
  텐서에 공통 slice 인덱스를 선택해 cross 표현을 저장. 청크끼리 의존성을
  만들지 않아 제거 시 다른 청크를 재인코딩하지 않도록 한다.
- 중복: CUR 자체의 KV 적용은 CurDKV(2025)에 이미 있다.
  https://arxiv.org/abs/2509.15038
  따라서 새 주장 후보는 단순 CUR가 아니라 공통 인덱스 tensor 표현 및
  청크 독립 저장의 비용/오차 특성이다. 해당 정확한 T-ARP KV 적용은
  이번 검색에서 확인하지 못했지만 모든 tensor KV 연구 검토가 끝난 것은 아니다.
- 이득 조건: 시간축 공통 구조가 충분하며 선택 slice/계수/인덱스/scale을
  포함해 같은 저장 바이트의 직접 양자화보다 읽기 오차가 작아야 한다.
- 첫 검사: 기존 개발 KV에서 같은 바이트의 RTN, 시간별 독립 근사,
  공통 인덱스 근사를 비교. tensor 오차와 실제 teacher-query 출력 오차를
  분리한다. K의 RoPE와 시간축 단 3개의 한계를 명시한다.
- 주의: 논문의 frequency-alignment와 rank-growth 가정은 자동 성립하지 않는다.
  해당 정리는 attention/긴 생성 품질 보장이 아니다. 현재 효과 미측정.

### 방법 2: 이력에 따라 정밀 계산 대상을 순환시키기 (방법 4 보완 가능성)

- 원 논문: Markovian Compression: Looking to the Past Helps Accelerate the Future,
  2026-01, math.OC preprint. https://arxiv.org/abs/2601.05398
- 가져올 것: BanLast처럼 최근 선택 이력을 사용해 같은 좌표만 반복해서
  정밀 처리하지 않고 제한된 정밀 계산을 분산한다.
- 적용 가설: 여러 denoising 호출에 걸쳐 같은 영역만 고정밀/저정밀로
  남지 않도록 정밀 읽기/보정 스케줄을 설계한다. 이미 저장한 코드를
  호출마다 재양자화하는 설계는 아니다.
- 중복/차이: ReadKV의 query 적응 정밀 읽기, recurrent-state low-discrepancy
  dither와 가깝다. 단순 난수 연동 자체는 새 기여로 주장하지 않는다.
  https://arxiv.org/abs/2610.11245
  https://arxiv.org/abs/2609.39185
- 첫 검사: 같은 호출별 정밀 예산의 독립 선택/고정 선택/이력 선택으로
  solver에 전달되는 결합 오차를 비교한다. 각 호출 오차 상관만으로
  최종 latent 손상을 추정하지 않는다.
- 위험: 분산학습의 수렴 정리는 frozen 생성 추론에 직접 적용되지 않는다.
  정상분포에서 균일 선택도 adaptive query에 대한 조건부 불편성을 보장하지 않는다.
  고정밀 refinement 저장량과 비용 문제는 방법 4와 마찬가지로 남는다.

### 도구 후보 C: AeroSketch로 sliding-window 통계를 싸게 유지

- 원 논문: AeroSketch: Near-Optimal Time Matrix Sketch Framework for Persistent,
  Sliding Window, and Distributed Streams, 2026-01; PACMMOD/SIGMOD.
  https://arxiv.org/abs/2601.02019
- 가져올 것: 창에서 오래된 행이 빠질 때에도 중요한 방향의 요약을
  유지하면서 반복 분해 비용을 줄이는 프레임워크.
- 정확한 AeroSketch KV 적용은 이번 검색에서 확인하지 못했다. 일반적
  sketch/low-rank KV 압축은 기존이므로 이것만으로 독립 방법을 주장하지 않는다.
- 역할: window별 query/KV 분포 추적, basis 갱신의 보조 알고리즘 후보.
  공분산 오차 보존만으로 softmax 출력이 보존되지는 않는다.
  실행 비용과 메타데이터가 원래 양자화보다 클 수도 있어 우선순위는 낮다.

## 이번 조사에서 독립 방법 후보로 제외/보류한 것

- 표본을 늘리다 오차 인증이 충분해지면 멈추는 attention: 최근 직접 적용 있음.
  Certified Sparse Attention via Perturbed Retrieval, 2026-10-18.
  https://arxiv.org/abs/2610.24991
- 단순 CUR 기반 토큰 선택: CurDKV와 중복.
- 단순 시간 연동 rounding: recurrent-state dither와 가까우며 현재 fixed-group
  RTN 반복이 추가 오차를 내지 않았다는 우리 진단상 출발 근거도 약함.
- 행렬 ODE용 dynamical low-rank integrator를 KV에 바로 대입: 필요한 저렴한
  변화 연산자 F와 low-rank 조건이 확보되지 않아 우선 후보에서 제외.

현재 두 기준을 확정 통과한 새 방법은 없다. 사용자 지정 순서대로 방법 1부터
개발 진단한다. 방법 2는 방법 4의 보완 가능성도 가지며 C는 보조 도구이다.
방법 1 계획: src/experiments/TENSOR_CROSS_DEVELOPMENT.md.

### 방법 1 첫 실행 결과

개발 54행 + 사후 평균 보완/SVD 참고 36행 완료. 공통 열 근사는 실패,
SVD 참고는 2비트 예산에서만 양 프롬프트 출력 오차 19.56–22.74% 개선.
T-ARP 구현은 아직 아니며 열 선택 개선과 기존 SVD 대비 이득을 다음 과제로 남김.
[결과](../src/experiments/tensor_cross_development/REPORT_KO.md).

### 방법 1 적응적 선택 후속 결과

Algorithm 8 기반 선택(5개 고정 시드)과 adaptive argmax, RTN2/static/SVD를
총 54행 비교했다. 랜덤 평균 D는 RTN의 1.804/1.837배, SVD의 2.335/2.284배.
encode도 SVD보다 느려 현재 rank22 공통 열 구성은 보류한다. SVD의 2비트
가능성과 모든 tensor 압축의 실패를 혼동하지 않는다. 방법 2는 아직 미실행.
[후속 결과](../src/experiments/tensor_adaptive_development/REPORT_KO.md).

### 방법 1 선양자화 순서 비교

2/3/4비트 각각 원본→RTN→방법 vs 원본→RTN만을 162행 측정했다.
T-ARP 최종 D 비율은 2.45–2.54 / 14.43–16.05 / 57.33–69.88이었다.
SVD도 모든 비트에서 악화. 최종 payload 차이 0.83–2.61%로 작다.
현재 순서/표현 보류 판단을 유지한다. 모든 수학적 압축의 불가능성은 아님.
[상세 비교](../src/experiments/tensor_prequant_comparison/REPORT_KO.md).
