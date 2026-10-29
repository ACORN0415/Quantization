# 방법 2: 이력 기반 정밀 읽기 배분 — 작은 실제 생성 검사

2026-10-30. 방법1의 6개 codec 양자화 결합 실패 후 사용자 지시에 따라 진행.
기존에 query는 chunk6 refresh와 chunk7 첫 호출만 저장되어 있어 4회 denoise
이력 효과를 기존 캡처로 확정할 수 없다. 작은 전체 모델 실행으로 검사한다.

## 질문/범위
같은 과거 V에 매 호출 25%의 BF16 읽기 예산을 줄 때, 최근 정밀 처리 위치를
피하는 스케줄이 독립 선택보다 한 청크의 최종 latent 오차를 줄이는가?
분산 최적화 수렴 정리는 이 생성 실험에 직접 적용하지 않는다.
원천: https://arxiv.org/html/2601.05398v1 BanLast(K,m).
본 실험은 1/p 불편 보정(방법4)이 아니라 실제 고정밀 값으로 교체하는 읽기 배분.

## 고정 조건
p0/p4, 생성 seed101, BF16 prefix chunks0..6, target chunk7(3 latent frames).
30층 중 0/14/29의 직전 chunk6 V만 RTN2로 압축하고 나머지 KV는 BF16 유지.
K·현재 chunk7 V는 BF16. 4번 denoise 호출에서 chunk6의 4680토큰 중 1170개를
BF16로 읽는다. 선택 마스크는 head/channel/layer 공통. 5번째 refresh는 그대로.
write 상태를 수정하지 않고 attention 읽기용 clone에만 적용. 저장 재양자화 없음.
선택하지 않은 과거 V는 고정된 RTN2 복원값; 선택 위치만 원래 BF16 값.

## 팔
- fixed: 첫 무작위 25%를 모든 호출에서 재사용
- independent: 호출마다 독립 25%
- banlast1: 직전 호출에서 선택한 토큰을 제외하고 다음 25% 선택
- roundrobin: 한 permutation을 네 부분으로 나눠 4호출에 각각 배정
선택 seed9101/9102/9103. 모든 팔의 첫 마스크 일치, 각 호출 high count 동일.
2프롬프트*4정책*3선택seed=24개의 target chunk 실행. RTN2-only와 BF16 기준
각 프롬프트 추가. 전체 생성을24번 처음부터 실행하지 않고 같은 BF16 cache를 복원.
고정 generation noise/re-noise seed. 미래 query/출력에 접근해 스케줄 고르지 않음.
roundrobin은 최대 배제 극한의 별도 탐색 팔이며 ergodic Markov 가정 주장 없음.

## 게이트/분석
먼저 prefix 7청크와 BF16 target이 기존 reference latent와 정확히 일치해야 함.
전체 토큰을 BF16로 선택하는 hook 경로도 target latent와 정확히 같아야 함.
각 읽기에서 대상 chunk 위치/값, 3층*4호출 개수, 원본 cache 불변을 검증.
최종 target latent normalized squared L2를 주 개발 지표로 기록. 전 단계 오차
합을 solver 결과라고 바꾸어 부르지 않는다. latent와 모든 마스크/해시 보존.
primary banlast1의 3seed 평균이 independent보다 두 프롬프트 모두10% 낮으면
확장 검사 후보. fixed/roundrobin 별도 보고. 여러 선택 seed는 생성 seed가 아님.
finite 전체 및 latency/peak 기록. decoded 영상 품질/장기 FR 개선 주장 없음.

## 메모리/효율 한계
이것은 읽기 배분의 가능성 검사이다. 어느 위치든 다음 호출에서 고정밀이 될 수
있으므로 전체 BF16 참조(또는 refinement)를 저장해야 한다. low-bit code와
전체 BF16 V를 함께 가진 비용을 기록한다. 25% 읽는다고 저장이25%인 것이 아님.
실제 hook은 BF16 복원 buffer를 사용하며 fused selective kernel이 아니므로
bandwidth/speedup 증거가 아니다. 모든 비교 팔의 저장/논리적 읽기 예산은 같다.
초기 단계에서는3층·1과거청크만 개입; 실패해도 모든 층/정밀도/예산 배분의
불가능성을 뜻하지 않는다. 결과 이후 범위를 몰래 늘리지 않는다.
