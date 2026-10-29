# 方法 1: 2비트 예산 적응적 열 선택 (개발 검사)

2026-10-30. 기존 결과를 본 뒤 설계, 측정 전 코드/계획 커밋. 개발 데이터
p0/p4 seed101 layers0/14/29 chunk6 V, 미래 chunk7 query32, BF16 K 고정.
이전 진단과 동일한 두 창 평균 새 V 기여 post-Wo D. 생성/보류 평가 아님.

## 방법
https://arxiv.org/html/2606.26688v2 Algorithm 8의 공통 열 인덱스 선택.
rFFT 상위 우측 singular basis에서 residual row norm으로 표본을 선택한다.
DC/쌍 주파수 가중치 1/2, 명시적으로 확률 정규화. 선택된 행의 QR로
nullspace projector를 재계산한다(논문의 compact identity). 선택마다 각
주파수 rank 증가와 선택 행 annihilation을 확인한다. 보장 가정인 전체
frequency alignment를 검증한 것은 아니므로 이론적 attention 보장 없음.
열 선택은 Algorithm 8, 복원은 이전과 같은 least-squares 계수 FP16 저장.
행까지 선택하는 Algorithm 9 full cross/CUR를 재현했다고 하지 않는다.

## 비교와 예산
rank22, 2비트 RTN 실제 payload 이하. static common, SVD FP16, RTN2 기준선.
T-ARP seed9101..9105 전부 보고, 좋은 seed만 고르지 않는다.
추가 deterministic adaptive argmax 팔은 원 논문 Osinsky 알고리즘 아님.
모든 선택/계수 fitting은 V만 사용. query는 평가에만 사용.
FP16 열/계수와 int32 인덱스의 실제 바이트, serialize/CPU 복원검사.
encode/decode 1회 warmup + 3회 반복 median; encode에는 SVD·선택·fit 전부 포함.
GPU 동기화 포함. 순서 고정 및 작은 개발 표본이므로 배포 속도 주장 금지.

## 판단
랜덤 5개 선택 seed의 D 평균을 3층 평균 후 프롬프트별 비교한다.
RTN 대비 양 프롬프트 10% 이상 개선이면 직접 양자화 대비 gate 통과.
SVD 대비 양 프롬프트 5% 이상 개선이면 추가 품질 가능성.
SVD에 품질 이점이 없으면, 비용도 함께 보고 기여 가능성을 제한한다.
argmax는 별도 탐색 팔로 보고하며 랜덤 실패를 대신 통과시키지 않는다.
입력 및 기존 코드 SHA, synthetic projector/fullrank/lowrank gates,
이전 RTN/static/SVD 결과 재현, finite 및 예산 준수 확인 후 결과 저장.
