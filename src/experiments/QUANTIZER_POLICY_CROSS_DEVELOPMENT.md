# 양자화기 × 저장 정책 교차 비교: 개발 단계

2026-10-29. 새 수치 측정 전 고정한 개발 계획이다. 기존 본 실험·메커니즘 등록 파일과 구현을 수정하지 않는다.

## 질문과 범위

RTN과 현재 독립 재구현 KIVI 계열의 품질 차이가 rolling과 C-repeat에서 달라지는가?
이는 원본 KIVI 논문 전체 방법의 재현, 최적 양자화기 선정, 동일 메모리 예산 비교가 아니다.

4팔은 `RTN/L`, `RTN/C-repeat`, `KIVI-style/L`, `KIVI-style/C-repeat`다.
Wan2.1-T2V-1.3B / Self-Forcing DMD, K4V4, g=16, sink=0, residual=0,
문맥 21잠재프레임(32760토큰), 출력 126잠재프레임/42청크/501픽셀프레임,
청크당 4 denoise + 1 refresh, 기존 chunk RNG를 유지한다. 모든 팔의 현재 청크는 attention에서 BF16이다.
L은 attention 뒤 전체 물리 버퍼 encode, C-repeat는 기존 과거를 attention 전에 encode하며
새 pending을 최초 한 번만 봉인한다. C-once를 주 비교에 섞지 않는다.

## 양자화기의 고유 동작

- RTN: 그룹별 토큰 축 대칭 absmax, signed byte codes, FP16 scales.
- KIVI 계열: key는 같은 토큰 그룹에서 channel별 min/max, value는 각 토큰의 head×channel 축 min/max.
  코드의 실제 구현 축은 `[B,groups,g,H,D]`에서 key reduce=(2), value reduce=(3,4)다.
  uint8 codes, FP16 scale과 min offset을 쓴다. legacy signed wraparound는 끈다.
- 원본의 token/channel 그룹 정의나 residual 정책을 모두 복제했다고 부르지 않는다.
  per-head value grouping으로 사후 교체하지 않는다. 그런 구현은 다른 실험이다.
- L의 빈 물리 영역과 padding, C-repeat의 청크별 부분 그룹은 기존 정책 동작으로 유지한다.
- 유효/패딩 코드, scale, offset, BF16 pending/residual, 실제 합계와 가상 packed 추정치를 모두 기록한다.
  같은 비트폭은 같은 저장량이 아니다. 현재 경로는 packed 저비트 커널이 아니다.

## 입력과 실행 순서

이미 개발/이전 연구에 사용한 `policy_study_prompts.txt`의 p0(고양이), p4(자전거), seed101만 쓴다.
새 보류 텍스트는 이 단계에서 만들거나 사용하지 않는다. 2프롬프트×4팔=8개 개발 생성이다.

1. 합성 두 양자화기×두 정책: eviction/refresh, 부분 그룹, identity, KIVI offset 포함 바이트,
   RTN 기존 adapter 대응과 interaction 부호 검사를 통과한다.
2. 전체 모델 no-quant L/C-repeat를 각각 42청크 기존 p0 seed101 BF16과 비트 단위로 비교한다.
3. 같은 BF16 쓰기를 6개 층 `{0,5,10,15,20,29}`에서 4팔에 재생한다.
   p0/p4 각각 teacher 출력 42청크가 기존 BF16과 일치해야 한다. attention 직전 과거 K/V를
   삽입 BF16 원본과 비교한다. 이는 개발 진단이며 전층 일반화 또는 FR 품질 대리 검정이 아니다.
4. 8개 FR을 실행하고 원본 float 디코딩 후반 프레임 249–500의 MUSIQ 252개를 평균한다.
   RTN p0 두 팔은 기존 개발 최종 latent와도 비트 단위로 비교한다.
   경계/내부 MUSIQ와 기존 raw DINO/RAFT 진단은 탐색적으로 함께 기록한다.
5. 관측 수를 프레임/청크 수로 부풀리지 않고 두 프롬프트의 개별 차이를 공개한다. p값은 계산하지 않는다.

## 결과를 보기 전에 정하는 진행 기준

각 프롬프트에서

`I = (KIVI_C-repeat − RTN_C-repeat) − (KIVI_L − RTN_L)`

를 계산한다. 양수는 KIVI의 RTN 대비 이득이 C-repeat에서 커진다는 뜻이다.
아래 모두 만족할 때만 새 보류 확인의 사전 등록을 준비한다.

- 모든 구현/재현 게이트 통과, 8개 전체 생성·평가 완료, 주요 수치 모두 finite.
- 두 프롬프트에서 I의 부호가 같고 **각각 |I|≥1.0 MUSIQ**.
- 케이스당 생성+평가가 10분 이내, GPU peak allocated<22GiB, 프로세스 CPU high-water<48GiB.

1.0점은 통계적 유의성 또는 품질 유지 기준이 아니라 작은 후속 투자용 효과 기준이다.
두 사례가 개발 판단에 충분한 일반화 표본이라는 주장을 하지 않는다.
불충족이면 결과와 부정 판단을 정리하고 새 보류 생성으로 자동 확대하지 않는다.
순위 역전은 GO의 필수 조건이 아니며, 발생해도 두 사례의 탐색 관측으로만 보고한다.

구현 오류·비유한 값·자원 초과는 가설의 NO-GO와 구분해 중지한다.
고친 소스로 다시 시작하려면 이전 산출물을 별도로 보존하고 새 source hash로 게이트부터 실행한다.
표본 추가, INT2로 교체, 한 프롬프트 제외, 기준 완화로 개발 GO를 만들지 않는다.

## 비용과 보존

생성 시간과 로딩·텍스트·VAE·평가를 분리한다. 진단은 별도 실행이므로 생성 성능에 섞지 않는다.
최종 잠재 텐서와 per-frame 지표는 보존하며 대형 자료는 `~/sfq_data/quantizer_policy_cross`에 둔다.
작은 결과·source/weight/artifact 해시는 Git에 기록한다. 개발 결과가 GO여도 새 보류 입력의
표본·주지표·검정·실행 소스를 별도 봉인하기 전에는 보류 생성하지 않는다.

CLI: `bash ~/Quantization/tools/watch_progress.sh 0`

Log: `tail -F ~/sfq_data/quantizer_policy_cross.log`
