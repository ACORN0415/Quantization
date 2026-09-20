# 문서 색인

세 갈래로 나눈다. **지시서**(무엇을 하라고 했는가) · **리뷰**(외부가 무엇을 지적했는가) ·
**결과서**(무엇이 나왔는가).

**중복을 만들지 않았다.** 같은 문서가 여러 곳에서 필요한 경우 실제 파일은 여기
한 곳에만 두고 원래 위치에는 **심볼릭 링크**를 걸었다. 사본을 두면 한쪽만 고쳐져
갈라진다 — 실제로 이번에 `RESULTS_session5*.md`가 그렇게 갈라져 있었다(정정 헤더가
리뷰팩 사본에만 달려 있었다).

---

## 지금 읽어야 할 것

| | 파일 |
|---|---|
| **결과 진입점** | `결과서/S6_FINDINGS.md` — 확정된 것 / 철회한 것 / 확정되지 않은 것 / **다음에 할 것** |
| 가장 최근 지시 | `지시서/S7_corrections_v2_2.md` |
| 리뷰 지적 대조표 | `리뷰/CORRECTIONS_2026-09-19.md` — claim ID별 철회·축소 |

---

## 지시서/

| 파일 | 내용 |
|---|---|
| `S1_HANDOFF_4090.md` | 세션 1 |
| `S2_HANDOFF.md` / `S3_HANDOFF.md` | 세션 2·3 |
| `S5_START.md` | 세션 5 시작 상태 |
| `S6_START.md` | 세션 6 시작 상태 (기준 커밋·관문 통과 기록) |
| `S6_HANDOFF_rev0.md` | 세션 6 지시서 **초판** (142줄) |
| `S6_HANDOFF_rev1_M1a_P0.md` | 세션 6 지시서 **개정판** (191줄, M1-a·P0 추가) — **세션 중 이쪽으로 전환** |
| `S7_corrections_v1.md` → `v2` → `v2_1` → `v2_2` | 세션 7 정정 지시, v2_2가 최신 |
| `HANDOFF_kernel_track.md` | 커널 트랙 |

**주의**: `S6_HANDOFF_rev0`과 `rev1`은 같은 이름으로 존재하던 **다른 개정판**이다.
세션 6은 rev0으로 시작해 도중에 rev1(M1-a·P0 우선)으로 전환했다.

## 리뷰/

| 파일 | 내용 |
|---|---|
| `AUDIT_REPORT.md` | 세션 5 감사 |
| `CODEX_REVIEW_HANDOFF_2026-09-19.md` | 코덱스 리뷰 인계 |
| `rw_D_novelty_check.md` | 선행연구 대비 신규성 점검 |
| **`CORRECTIONS_2026-09-19.md`** | **세션 6·7이 철회·축소한 주장 대조표 (claim ID별)** |
| `codex_review_pack/` | 외부 리뷰어에게 보낸 **스냅샷**. 각 문서 상단에 경고 헤더, `CODEX_BRIEF.md`의 철회된 주장 10개에 인라인 ⚠ 표시 |

## 결과서/

| 파일 | 내용 |
|---|---|
| `S1_RESULTS_4090.md` ~ `S4_RESULTS.md` | 세션 1~4 |
| `S5_RESULTS.md`, `S5_RESULTS_gates.md`, `S5_NOTES.md`, `S5_STATUS_2026-09-18.md` | 세션 5 |
| **`S6_FINDINGS.md`** | **세션 6 진입점** |
| `S6_RESULTS_session6.md` | 세션 6 진행 기록 (표는 CSV에서 자동 생성) |
| `S6_NOTES_session6.md` | 함정·정정 목록·재사용 안내 |
| `S6_CORRECTIONS_log.md` | 무엇을 왜 바꿨는가 |
| `S6_gateK_bitdiff.md` | 격자 메커니즘 |
| `S6_gateL_concentration.md` | 집중 유도 + 실측 |
| `S6_gateC2_taylor.md` | Taylor 대 정확형, 구현 발산 |
| `S6_storage_semantics.md` | 두 코드베이스의 저장 모사 차이 |
| `S6_C3_predictions.md` | 사전 가설과 보고 방식 |
| `S6_STATUS_LIVE.md` | 세션 중 실시간 상태 (일부 낡음) |
| **`L1_attr_diagnostic.md`** | **L1 진단 (2026-09-20)** — P3 경로의 양자화기·attention 두 구성요소 대응 확인. 시험한 상태 한정 |
| **`L1_storage_design.md`** | **저장 설계 P1·P2·P3·P4 단독 비교 (2026-09-20)** — 품질·메모리. G1 조사가 남긴 항목 |
| **`THESIS_position_2026-09-19.md`** | **절별 입장 (현행)** — 절마다 ①쓸 수 있는 주제문 ②근거 ③쓰면 안 되는 문장 ④빈 자리. **집필은 이 문서로 시작한다** |
| `THESIS_by_section.md` | 절별 주제문 (세션 5) — **대체됨. 이력용** |

---

## 데이터는 옮기지 않았다

CSV·JSON·latent·비디오는 코드가 읽는 경로에 그대로 있다.

```
~/gpu/Self-Forcing/results/session6/     세션 6 CSV·JSON (문서는 심볼릭 링크)
~/gpu/Self-Forcing/results/              세션 1~5 산출물
~/gpu/LongLive/results/                  M1-a
```

## 도구

```
~/gpu/Self-Forcing/tools/render_tables.py   본문 표를 CSV에서 생성 (make tables / check-tables)
~/gpu/Self-Forcing/tools/runlib.sh          드라이버 종료코드 집계
~/gpu/Self-Forcing/memreport.py             storage 단위 메모리 집계 + 반례 self-test
~/gpu/Quantization/tools/{s6stats,analyze_s6,analyze_vbench,vbench_dims}.py
```

**미커밋**: 위 도구와 세션 6 코드 변경은 아직 커밋되지 않았다
(`결과서/S6_FINDINGS.md` §5.1 F4). `final_recipe_manifest.json`이 diff 해시로 추적 중.
