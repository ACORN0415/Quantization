# 세션 7 이후 시작 상태 (2026-09-19 작성)

세션 6 실험 + 세션 7 정정(`S7_corrections_v2_2.md`)을 대부분 수행한 시점의 인계.

## 읽는 순서
1. **`문서/결과서/S6_FINDINGS.md`** — 확정/철회/미확정/할 일. 여기부터.
2. `문서/리뷰/CORRECTIONS_2026-09-19.md` — 철회·축소된 주장 claim ID별 대조표
3. `문서/지시서/S7_corrections_v2_2.md` — 최신 지시서 (남은 항목 확인용)
4. 이 문서 — 코드 상태·관문 기록·함정

---

## 1. 코드 상태 ⚠ 커밋 안 됨

| 저장소 | 커밋 | 상태 |
|---|---|---|
| Self-Forcing | `cfc8584` | **미커밋 변경 다수** (아래) |
| kv-quant-longhorizon | `650ce90` | 변경 없음 |
| LongLive | — | **미커밋 변경 7개 파일** (TUM 보정 이식) |
| sf-pristine | `33593df` (worktree) | P0 비교용 원본 |

미커밋 변경 (Self-Forcing):
```
수정: inference.py  tum_correction.py  wan/modules/causal_model.py
      oracle_anchor.py  figs_make.py  gateL3_concentration.py  gateO1_residual.py
신규: memreport.py  gateC2_taylor.py  gateK3_interaction.py  gateK1_report.py
      Makefile  tools/render_tables.py  tools/runlib.sh  run_s6_*.sh  run_s7_*.sh
```
**첫 할 일: 커밋.** `git add -A`는 results의 비디오·latent 수 GB를 스테이징하려다
멈춘다 — `.gitignore`에 `results/` 산출물을 넣고 코드만 스테이징할 것.
`final_recipe_manifest.json`이 diff 해시로 추적 중이므로 그 전 수치의 출처는 남아 있다.

## 2. 통과한 관문 (재검증 불필요)

| 관문 | 결과 |
|---|---|
| 플래그 전부 OFF, **63프레임**×3p(퇴출 포함) vs 세션 3 latent | INT4 63/63, INT2 63/63 bit-exact |
| 정렬 ON + BF16 vs A1_bf16 | 21/21 bit-exact |
| TUM taylor: 세션 6판 vs HEAD판 | 21/21 bit-exact ×2 |
| 보정 no-op: λ0 vs λ1 chunk 0 | 3/3 일치, chunk 1+ 60/60 상이 |
| 재양자화 카운터: 정렬 ON | 0 / 47,250 |
| fused 패킹·attention 등가 관문 | 통과 |
| LongLive 무보정 경로 flags-OFF | 210/210 bit-exact |
| LongLive 보정 경로 λ0 vs λ1 | chunk 0 10/10, chunk 1+ 200/200 — **부분 검증** |

**21프레임 회귀는 퇴출을 거치지 않는다**(7슬롯 캐시에 7 chunk). 회귀는 반드시 63프레임으로.

## 3. 새 플래그·도구

```
inference.py
  --kv_grid_align {off,segment}     슬롯별 독립 양자화 격자
  --tum_mode scalar --tum_scalar_c  캐시 logit에서 상수 하나
  --tum_scalar_fast                 FA2 2회 + lse 블렌드 (명시 경로 6배 → 2배)
  --tum_bias_diag / --tum_bias_dump L3 계측 (느림, 옵트인)
  --kv_requant_audit <path>         살아남은 슬롯의 코드 변화 카운터
  --kv_bytes_report <path>          storage 단위 실측 바이트 + peak 2종 + diff 해시
teacher_forced.py  --kv_grid_align, --kv_key_bits/--kv_value_bits, --tum_scalar_c
oracle_anchor.py   --policy {oracle,oldest,fixed} --fixed_anchors N...  + 선택 이력 로그
LongLive/inference.py  --tum_correct --tum_lambda

tools/render_tables.py   본문 표를 CSV에서 생성. make tables / make check-tables
tools/runlib.sh          드라이버 종료코드 집계 (부분 실패를 DONE으로 못 찍음)
memreport.py             storage 단위 메모리 + 반례 self-test
~/gpu/Quantization/tools/s6stats.py   t(n−1,.975) 기반 CI, TOST
~/gpu/vbench-venv        VBench 0.1.5 (sf-venv와 분리)
```

## 4. 절대 하면 안 되는 것

1. **H2 `exact_uniform` 본런** — 출하 8항 급수가 층 29 꼬리에서 발산(수치안정형 대비
   최대 절대차 2.3e+27). 먼저 `gateC2_taylor.py:log_sinh_over_x`로 교체할 것.
2. **비유의를 "동등"으로 쓰기** — TOST 미실시. 예외는 오라클뿐(선택 이력 140/140 동일).
3. **"8× 압축"** — 최종 경로 K4V2의 **실측은 4.00×**. 8×는 K3V2(커널 미구현) 이론값.
4. **INT2에서 요소별 기여를 더하기 / 쌓인 막대** — 상호작용 +9.22 [5.85, 12.59].
5. **LongLive 메모리 절감 인용** — BF16 master를 유지한 채 quant_state를 추가한다.
6. **세션 5 §4 emp ε 결론 인용** — 항등적으로 0인 양을 측정했다(보류).
7. **실행 중인 bash 스크립트 편집** — 바이트 오프셋 재읽기로 실행이 어긋난다.

## 5. 남은 일 (상세는 `S6_FINDINGS.md` §5)

**고칠 것**: F1 P0 정면 비교 부재(~25분, 가장 중요) · F2 커널 기준선 통일(~10분) ·
F3 exact_uniform 수치 안정화 · F4 커밋 + .gitignore + pre-commit 훅 검증
**넓힐 것**: F5 L3·C2 다중 프롬프트(현재 prompt 1개) · F6 격자 두 경로 분리(실측으로만)
**미실행**: TOST · L1 세 변형 · O2 · M1-b · L2 · L4 · M2 · 원본 키 진단 · 사람 평가
**문서**: `결과서/THESIS_by_section.md`가 세션 6·7 정정 **미반영**

## 6. 실측 소요 (인계 노트의 추정과 5배 차이났던 값)

- 63프레임 × 10프롬프트, 일반 경로: **약 11분** (프롬프트당 56초)
- 명시적(TUM) 경로: 약 37분 (일반의 3.4배)
- 21프레임 × 3프롬프트, 무디코드: 약 1분
- VBench 3차원, CPU, 10비디오: 약 7분/런

**세션 5 노트의 "63f×10p ≈ 56분"은 실제로 프롬프트당 56초였던 것으로 보인다.**
계획을 세우기 전에 첫 런으로 교정할 것.

## 7. 함정 (세션 6에서 실제로 겪은 것)

- `pgrep -f <스크립트명>` 대기 루프는 **자기를 띄운 셸을 매칭**해 영구 대기한다.
  마커 파일로 대기할 것.
- 새 계측은 **알려진 답이 있는 경우로 먼저 돌려볼 것**. 이번에 도구 버그 6건이 났고,
  합성 데이터로 검증한 L3 분석만 처음부터 옳았다.
- 두 점추정을 비교할 때는 **직접 대비의 CI를 먼저 계산**할 것. 유의성 차이로 차이를
  주장하면 안 된다(λ0.5 vs λ1, flickering ON vs OFF 둘 다 이 오류를 냈다).
- 새 비교표를 만들기 전에 `ls results/*_per_prompt.json`부터 볼 것. 교차모델 목표선을
  n=3 자료로 계산했는데 10프롬프트 자료가 이미 있었다.
- `tail -3`으로 배치 결과를 판단하지 말 것. 5행 중 4행 실패를 1행 실패로 보고했다.
