# 6개 표현 × 양자화 순서 × 2/3/4비트 개발 비교

2026-10-30. 사용자 요청에 따른 폭넓은 개발 비교. 모든 선택은 본 측정 전 고정.
기존 개발 p0/p4, 생성 seed101, layers0/14/29, chunk6 V와 chunk7 query32.
BF16 K 고정, 기존 두 창의 새 V 기여 post-Wo squared error D. 원본은 평가용,
query는 어떤 분해·rank 선택에도 사용하지 않는다. 긴 영상 생성 실험 아님.

## 방법: 기존 2개 + 일반 근사 2개 + 텐서 분해 2개
1. T-ARP: Algorithm8 공통 열 선택, Fourier 최소제곱 복원. Algorithm9 CUR 아님.
2. t-SVD: 기존과 같은 head별 Fourier 저랭크 표현.
3. pivot_qr: 시간·공간을 펼친 행렬, residual column norm pivot과 재직교화 QR,
   Q와 Q^T X를 저장. 대표 열 인덱스도 기록/계수. 강한 RRQR이라고 부르지 않는다.
4. range: 같은 펼친 행렬, Gaussian range finder + QR 정규화된 power iteration1,
   oversampling/SVD refinement 없음. Q와 Q^T X 저장.
5. Tucker: 공간·시간·채널 mode-SVD(HOSVD), HOOI 최적화 아님.
   rank=(r,3,min(r,128)); 단 3개인 시간축은 강제로 자르지 않는다.
6. TT: 공간→시간→채널 순서의 TT-SVD, ranks=(r,min(r,128)).

T-ARP/range의 난수 시드는 모든 비교에 9101 하나로 고정. 이전 5시드 실험과
구분한다. 다수 방법의 screening이며 우수 결과는 추가 시드·보류 검증 필요.
방법명 자체나 이번 비교를 신규 알고리즘으로 주장하지 않는다.

## 각 2/3/4비트 예산에서 비교할 4경로
- method_only: 원본→방법→FP16 인자. 순서의 기준점.
- quant_then_method: 원본→RTN b비트→복원→방법→FP16 인자.
- method_then_quant_rank: 원본→방법→인자 b비트, 위 두 경로와 동일 rank.
- method_then_quant_budget: 원본→방법→인자 b비트, RTN 예산 이하 최대 rank.
직접 RTN 양자화 18행 + 6방법*4경로*3비트*6셀=432행, 총450행.
같은 rank 비교와 같은 예산 비교를 구분한다. 전체 인자 압축을 실제 pack한다.

## 저장 표현과 rank
입력 V RTN은 원래 토큰축 g16 per-feature. 인자는 각 연속 메모리의 16원소
블록별 symmetric RTN, FP16 maximum, 실제 LSB-first uint8 bitstream.
두 quantizer의 축이 다름을 명시. 인자 group/scale/gauge를 결과로 조정하지 않음.
SVD는 U*S와 Vh, QR는 Q와 Q^T X, TT는 앞 코어 orthogonal/마지막 scale,
Tucker는 orthogonal mode factors/core. 별도 scale balancing 학습 없음.

분해 자체는 FP32이며 store에서 FP16 또는 packed low-bit로 변환한다.
이전 T-ARP는 FP16 열에 맞춰 계수를 fit했지만 이번에는 FP32 열 기준이다.
새 codec 버전이므로 과거 T-ARP 결과와 비트 단위 일치를 요구하지 않는다.
RTN 기준선의 일치는 필수. 실험 간 다른 factor convention은 명시한다.

rank는 값/query가 아니라 shape/실제 byte 식으로만 정함. FP16인자 경로는
RTN payload 이하 최대 rank. post-quant budget은 실제 code+scale+index를
포함해 최대 rank. matrix/Fourier r<=128, Tucker/TT 공간 r<=384, 채널<=128.
표현 가능한 rank 상한/정수 rank로 예산 미사용이 생기면 실제 bytes로 보고한다.
FP16 최대값 및 index 포함 tensor payload이며 파일 header/allocator는 별도.
기존 양자화값/원본/전체 복원 workspace를 deployment residency로 합치지 않음.

## 측정 및 통과 기준
D_final/D_plain, V relative L2, plain 결과에서 추가된 V/output error를 분리.
순서 비교의 추가 squared error와 총오차 증감은 교차항 때문에 같지 않음.
각 단계 CUDA 동기화 wall time 기록: 선양자화/분해/store, 최종 decode.
분해는 공유 계산을 허용하되 각 행에 그 분해 비용을 청구. decode warmup1+3median.
분해 시간은 첫 실행이며 서로 다른 방법의 배포 성능 우위 주장 금지.
상태 저장 뒤 CPU decode와 GPU 비교, 입력/코드/상태 해시, finite/예산 준수,
synthetic fullrank·lowrank·직접 scalar pack reference 검사 후 실행.

개발상 다음 검사 후보: 두 프롬프트 모두 D<=0.9*plain이며 bytes<=plain.
또는 동일 rank에서 bytes<=0.5*plain이고 D<=1.1*plain인 압축 후보.
이는 방법 간 탐색 기준이며 통계적 유의성/신규성/영상 품질 확증이 아님.
실패한 행이나 seed를 빼지 않는다. 오류/수치 실패는 명시하고 수정 시 버전 기록.

## 출처 (독립 구현, 외부 코드 복사/설치 없음)
- T-ARP: https://arxiv.org/html/2606.26688v2 (Algorithm8)
- QR/무작위 range: https://arxiv.org/abs/2002.01387
- Tucker: https://tensorly.org/stable/modules/generated/tensorly.decomposition.tucker.html
  (일반 Tucker 참고; 본 구현은 단순 HOSVD이며 해당 라이브러리 HOOI와 구분)
- TT-SVD: https://doi.org/10.1137/090752286
