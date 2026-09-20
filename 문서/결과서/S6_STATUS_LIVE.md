# 세션 6 실시간 상태 (2026-09-18 20:25 갱신)

리로드하더라도 이 파일 + RESULTS_session6.md + NOTES_session6.md만 읽으면 이어갈 수 있다.
**배치는 nohup으로 detach되어 있으므로 세션을 새로 열어도 계속 돈다.**

## 진행률 (수정본 §7 우선순위 기준 14항목)

| 항목 | 상태 |
|---|---|
| K1 격자 원인 | ✅ 완료 |
| K2 격자 정렬 구현 | ✅ 완료 |
| K3 정렬 ON 재측정 | ✅ 완료 — **격자 가설 확정** |
| M1-a 곱셈 검정 | 🔄 4실행 중 3개 완료, 마지막 진행 중 |
| P0 원본 대비 | ⏳ 체인 대기 (스크립트 완성, worktree 준비됨) |
| L1 스칼라 sweep | ⏳ 체인 대기 |
| L3 집중 유도 | ✅ 완료 |
| N 그림 | ✅ 완료 (10장) |
| O1 잔여 분해 | ✅ 완료 |
| O2 잔여 후보 | ⏳ 체인 대기 |
| M1-b LongLive 핵심표 | ⏳ 체인 대기 |
| L2 TF에서 c* | ❌ 미착수 |
| L4 flatten 일반성 ×4 | ❌ 미착수 (수정본에서 확대됨) |
| M2 SkyReels | ❌ 미착수 |

완료 6 / 진행 1 / 대기 4 / 미착수 3.

## 실행 중인 체인 (detached)
`run_s6_chain.sh` → M1-a → P0 → L1(driverB) → O2(driverC) → K1b → M1-b
별도로 `vbench_dims.py`가 CPU에서 정렬 ON 런 5개를 채점 중.

로그: `results/session6/{m1a,p0_driver,driverB,driverC,k1b_driver,m1b,chain}.log`

## GPU 실측 (중요 — 인계서 추정과 5배 차이)
- 63f × 10프롬프트 일반 경로 = **약 11분** (프롬프트당 56초).
  인계서의 "56분"은 실제로 프롬프트당 56초였던 것으로 보인다.
- 명시적(TUM) 경로 = **약 37분** (일반의 3.4배).
- 21f × 3프롬프트 무디코드 = 약 1분.
→ 남은 체인 전체가 6~8시간이면 끝난다. 39시간 예산은 5배 여유.

## 미착수 3항목의 준비 상태
- **L2**: `--tum_scalar_c`가 teacher_forced.py에 이미 배선됨. L1의 c*가 나와야 시작 가능.
- **L4**: 수정본에서 4실행으로 확대(RTN g32 / KIVI / LongLive / SkyReels).
  앞의 셋은 코드 준비 완료(`--kv_block_size 32`, `--kv_quant KIVI`, LongLive 포팅 완료).
  SkyReels는 M2와 공유하며 코드 이식 미착수.
- **M2**: 미착수. TUM 저장소 SkyReels-V2 설정 필요.

## 아직 주장하면 안 되는 것
1. ~~정렬 ON 레시피가 BF16 수준~~ → **해소됨.** VBench 확인 완료
   (`vbench_k3_paired.csv`): MUSIQ·subject_consistency·imaging_quality 셋 다 n.s.
   **단 temporal_flickering만 유의하게 나쁘다(t=−3.3)** — 그리고 측정한 모든
   양자화 설정이 이 차원에서 BF16과 다르다. "BF16과 구별 불가"는 세 지표에
   한정해서만 쓸 것. 정렬은 flickering을 오히려 악화시킨다(−0.0024 n.s. →
   −0.0034 t=−3.3). 원인 미규명 = 세션 7 후보.
2. A2의 정렬 이득이 A1의 5배인 이유(292번째 블록 가설) — `run_s6_K1b_sink.sh` 대기.
3. "상수 하나로 보정 대체 가능" — L1 대기. L3는 head 간 10,527배 편차를 보였으므로
   전역 상수로는 부족할 가능성이 크다.


## 재시작 후 확인 (20:22)
세션 리로드에도 전부 생존: 마스터 체인(pid 1806754), M1-a, 독립 관문 감시자,
VBench. nohup detach가 의도대로 동작했다.

## M1-a 보정 경로 관문 (리로드 직전 추가)
무보정 경로만 관문(210/210)을 통과시키고 보정 경로는 관문 없이 M1-a를 돌리고
있었다 — 프로젝트 자체 규칙 위반. `run_s6_M1a_gate.sh` 추가:
LongLive sink=3이 정확히 한 chunk이므로 chunk 0은 전부 self → bias 전부 0 →
λ=0과 λ=1이 **bit-exact여야** 하고 chunk 1+는 **달라야** 한다. 둘 다 명시적
경로라 경로 교체가 상쇄되고 self 제외 배선과 delta_sq 정렬만 검사한다.
**이 관문이 실패하면 M1-a를 다시 돌려야 한다.**
