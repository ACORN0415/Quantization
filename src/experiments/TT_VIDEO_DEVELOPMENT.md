# TT의 실제 영상 4경로 비교 — 개발 계획

2026-10-30. 사용자가 2/3/4비트 모두 네 경로 생성을 요청했다.
현재 positive evidence는 V-only이므로 30개 층의 모든 과거 V를 압축하고
K/현재 pending 청크는 모든 팔에서 BF16으로 고정한다. 전체 KV 저비트 주장 금지.
기존 개발 p0/p4, 생성 seed101, 126 latent frames=42chunks=501 pixel frames.
BF16 기준2개 + 2prompts*3budgets*4arms=24개, 총26개 영상.

## 네 팔과 예산
- tt: 원본 V→TT→FP16 인자
- quant: 원본 V→실제 packed RTN2/3/4(g16)
- quant_tt: 원본 V→RTN→복원→TT→FP16 인자
- tt_quant: 원본 V→TT→인자 RTN2/3/4(연속16원소 실제패킹)
각 b비트 직접 V양자화의 실제 bytes 이하 최대 TT rank; TT_quant는 같은 예산
비교라 rank를 높임. rank60/77/94(FP16),299/299/299(인자 저비트)를 shape/byte
식으로 결정. 같은 rank 순서 비교는 앞의 450행 진단에서 별도 제공했다.
고정 codec_order_core의 공간→시간→채널 TT-SVD와 gauge 사용.
새 청크가 시작되면 직전 완료 청크를 한번 encode, 최근6개 압축/현재1개 BF16.
rolling 재양자화·sink없음. 기존 ChunkStore를 사용하고 기존 동결 소스 수정 안 함.

## 실행/게이트/순서
구현 identity codec에서 양 프롬프트8청크가 기존 BF16 latent와 정확히 같아야 함.
2비트 각4팔의 첫8청크로 기능/finite/초기동일성/실행시간/메모리를 확인.
이 검사는 개발 가능성 검사이며 결과로 팔/프롬프트/평가를 바꾸지 않음.
본실행: BF16 2영상 먼저 재생성/기존126frame reference와 정확히 대조.
이후 2비트→3비트→4비트, 각 prompt와4팔의 고정순서. 각 사례를 독립 초기화.
모델 로딩시간 별도. 모든 사례의 잠재텐서/프리뷰 영상/원시 지표 보존.
중단 시 source/input hash가 일치하는 완료 사례만 재사용하는 resume 지원.

## 오차/품질
- latent normalized squared L2 vs 같은 BF16, chunk별/전체/후반21..41.
  BF16과의 궤적 거리이며 이것만으로 시각적 품질을 판단하지 않는다.
- 삽입 당시 BF16 V 대비 저장 복원 relative L2, 층/segment norm합, 청크별.
  BF16 기준 생성의 V가 아니라 각 FR 자체 삽입값 기준임을 명시.
- MUSIQ: MP4 아닌 raw decoded RGB8 전501프레임. 전체/후반(249..500)/
  경계(c>=1의12c-3)와 내부 분리. 선행 실험 가운데1프레임과 직접 합치지 않는다.
- DINO frame feature consistency: raw 전프레임 특징, 주체 동일성의 proxy.
- RAFT flow magnitude/warp residual: 경계쌍(12c-4→12c-3)과
  내부쌍(12c+2→12c+3,12c+3→12c+4), 전체/후반. 빈 대응은undefined,
  motion이 줄어서 일관성이 좋아 보이는 위험을 함께 판단.
- BF16 raw RGB8 대비 pixel MSE/PSNR. 생성 궤적 차이의 보조 지표.
MP4는 육안 확인용; 지표 계산에는 사용하지 않는다.

## 속도/메모리
각42청크 sampling wall time, latent-frame throughput, chunk별 시간.
CUDA events로 encode/restore 시간(ChunkStore의 오차계산 복원 포함)을 기록.
첫컴파일/초기실행 영향/반복 없는 개발 timing 한계를 명시. optimized kernel 아님.
실제 cache tensor payload(K/V/pending 분리), full-window byte plateau,
PyTorch generation peak allocated/reserved와 decode/eval peak 구분.
FP32 TT 분해/복원 workspace와 모델을 포함한 peak를 compression ratio라 하지 않음.
전체BF16참조나진단용sourcecache는생성중GPU에상주하지않음(reference latentCPU).
품질평가/영상쓰기 시간은 sampling과 분리.

## 분석과 제한
모든 비트/팔을 공개. p0/p4 각각의 paired 차이와 개발 평균 보고, n=2·seed1
유의성/저널기여/일반화 주장 없음. TT의 인자 저장 비용/복원비용을 숨기지 않는다.
TT단독이좋아도TT→양자화가좋다고하지않고, 메모리/시간/움직임 trade-off를 유지.
출처: TT https://doi.org/10.1137/090752286 ; MUSIQ/DINO/RAFT는 저장소 기존
평가 코드/공식 가중치 사용. 외부코드 신규복사 없음.

## 구현 세부 명시
FP32 TT 분해 후 상태를 FP16/실제 packed code로 보관하고 attention에는 BF16으로
복원한다. cache_bytes는 K/V tensor payload이며 scalar end index/profiler
event/Python header를 제외한다. 전체 VRAM peak는 별도다. 본 queue 전에
기존 BF16501프레임의 full decode·raw evaluator preflight를 실행한다.

주 생성시간은 각 run_chunk의 CUDA동기화 wall time 합(sum_chunk_seconds /
generation_seconds)으로 계산한다. sampling_seconds는 CPU 오차 집계/진행기록까지
포함한 instrumented trajectory elapsed로 별도 표시. 같은 이유로 주 throughput은
generation_seconds 기준이며 품질평가/모델로딩은 포함하지 않는다.
