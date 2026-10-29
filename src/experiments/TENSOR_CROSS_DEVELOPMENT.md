# 방법 1: 공통 인덱스 텐서 압축 — 개발 타당성 검사

2026-10-30. 새 보류 실험이 아니라 이미 사용한 개발 자료의 진단이다.
측정 전에 이 문서와 실행 코드를 커밋한다. 다른 방법은 아직 실행하지 않는다.

## 질문과 범위
청크 3프레임의 V를 head별 공간(1560) × 채널(128) × 시간(3)으로
표현하면, 같은 실제 payload 예산의 직접 RTN보다 미래 읽기 오차가 작은가?
K는 BF16로 고정한다. 이번 결과로 K 압축/전체 생성 품질을 주장하지 않는다.
사용 자료: p0/p4, seed101, layers0/14/29, chunk6 V; chunk7 첫 호출의
기존 32개 query. fitting에는 V만 사용하고 미래 query는 평가에만 사용한다.
과거 V 전체가 없어 새 V가 만드는 post-Wo 오차만 평가한다.

## 1차 비교 (T-ARP 완전 재현 전의 표현 검사)
각 RTN 2/3/4비트의 실제 payload(코드+FP16 maxima)를 예산으로 삼는다.
- RTN: 기존 실제 pack 구현.
- independent: 프레임마다 다른 실제 채널 열을 선택하고 least-squares 계수 저장.
- common_fourier: 동일 채널 인덱스를 모든 Fourier 주파수에 사용하고
  주파수별 least-squares. 실제 선택 열과 inverse-FFT 계수를 FP16 실수로 저장.
  decode는 FFT → 행렬곱 → inverse FFT. conjugate symmetry를 rfft로 보존.
두 cross 팔 모두 SVD 우측 부분공간의 leverage score 상위 열을 선택한다.
이는 간단한 deterministic screening이며 논문의 adaptive randomized pivoting이 아니다.
선택 열/계수 FP16, 인덱스 int32의 실제 tensor storage 바이트를 센다.
각 팔에서 예산 이하의 최대 정수 rank 사용. 공통 인덱스는 head별 공유한다.
추가로 같은 rank의 독립 SVD, Fourier SVD 잘림 오차를 계산한다.
이는 FP32 Frobenius 최적 rank 잔차이며 출력 오차 하한/실제 배포 codec이 아니다.
단순 인덱스 선택 실패와 표현 자체의 rank 부족을 구분하기 위한 진단이다.

## 지표와 판단
두 고정 BF16 cache 창(전체와 가장 오래된 청크 제거)에서 새 V 기여의
post-Wo squared error 평균 D; 층 평균 후 프롬프트별 비교.
V 상대 L2, 모든 tensor payload 바이트, CUDA wall encode/decode 시간,
전체 진단 peak allocated/reserved도 보고한다. 파일 컨테이너/allocator overhead는
payload와 구분하며, 전체 BF16 복원 workspace를 포함하므로 배포 메모리 이득 아님.
각 timing은 첫 실행 결과이며 벤치마크 확정값이 아니다.
공통 표현이 동일 비트 예산의 RTN보다 D를 양 프롬프트에서 10% 이상 줄이면
정식 T-ARP 구현/선택 안정성 검사를 우선한다. 여러 예산 탐색은 개발 탐색이다.
실패하면 SVD 잔차·열 선택 손실을 확인한다. 선택 손실이 크면 안정화된 선택을
후속 계획으로 분리하고, 단 한 번의 실패로 전체 tensor 압축을 폐기하지 않는다.
미래 query로 인덱스/랭크/계수를 맞추지 않는다. 출력 원시 행을 모두 보존한다.

## 실행 게이트와 출처
소형 synthetic rank-full 복원, rfft 실수 복원, payload 예산 준수, 입력 SHA 확인,
기존 capture와 V 일치, 모든 결과 finite. 원본/동결 결과는 수정하지 않는다.
https://arxiv.org/abs/2606.26688v2 (T-ARP, 가정부 이론; 본 screening과 구별)
https://arxiv.org/abs/2509.15038 (CurDKV: CUR 자체는 신규 아님)
