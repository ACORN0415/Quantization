# §4 Regridding · §5 The Interaction — 초고 v1 (영문)

작성 2026-09-20. 근거: `claim_table.md` G1~G6·Q1~Q6, `S6_FINDINGS.md` §1.1~§1.4.
번호는 **안 A(분석 9쪽)** 기준이다(구 문서의 "§4.5 격자"와 "§6 상호작용"에 해당).

> **집필 메모.** §4는 메커니즘, §5는 그 메커니즘이 평가 결론에 미치는 영향이다. **§4에 품질 인과를 넣지 않고**, §5에 메커니즘 설명을 되풀이하지 않는다.

---

## 4 Regridding in Rolling Caches

### 4.1 Where the grid moves

The cache is quantized in groups of 16 tokens along the token axis, each group carrying its own scale. A chunk is 4,680 tokens, which is 292.5 groups. Eviction therefore does not remove a whole number of groups: surviving content is displaced by 8 tokens relative to the group boundaries, and every group re-forms with different members.

Round-to-nearest is **idempotent on a fixed grid**: re-quantizing already-quantized content reproduces it bit for bit (relative difference 0.000e+00). Once the grid is displaced by 8 tokens, it is not: the relative difference is 5.47e−2 at INT4 and 2.01e−1 at INT2. These are storage-only figures — a clean BF16 chunk put through the storage operations alone, with no generation feedback — and they are measured by a different instrument from the run-time counter in §4.3.

In this diagnostic the relevant lifetime is the window's: in the sink-free A1 configuration the window holds 7 slots, so a given slot is re-quantized at most 6 times before it is evicted, and the storage error saturates there. **That ceiling comes from the window length, not from the mechanism** — it does not apply where a fixed sink keeps content resident, and it is not comparable to the change counts of §4.3, which aggregate over prompts, layers and repeated generation calls rather than tracking one slot's lifetime.

*Scope.* This idempotence is a property of the RTN implementation and numeric conditions we verified — symmetric scaling, group size 16 along tokens, the accumulation order and dtype of our path. It is not asserted of RTN in general; a different clipping rule, scale formula or accumulation order can behave differently.

### 4.2 Five quantities, kept separate

Throughout we distinguish (i) whether a **re-quantization call** occurs, (ii) whether the stored **integer codes** change, (iii) whether the **dequantized value** changes, (iv) whether the **error against the original** grows, shrinks or stays, and (v) whether any of this **affects output quality**. None of these implies the next. In particular a re-quantization call on a fixed grid changes nothing (§4.1), and a change in dequantized value does not by itself mean a larger error, since the error norm may decrease while the error's direction and correlation structure change.

This section establishes (i)–(iii) directly. Quality effects are the subject of §4.4 and §5; (iv) is measured only for the storage-error diagnostic described below.

### 4.3 How often codes actually change

A counter in the generation path tracks content — not slot indices — across an INT4 run and records, for each surviving slot, whether its stored **key** codes differ from what they were at the previous write, over 3 prompts × 63 frames × 30 layers. The unit is a **slot comparison**, not a call: one re-quantization call compares many surviving slots. Across 47,250 comparisons of surviving slots between successive writes, stored key codes changed in 8,910 comparisons.

Those comparisons partition into two populations according to whether the grid the slot sits on has been displaced; the two change rates use separate denominators:

| | comparisons | comparisons in which key codes changed | rate |
|---|---|---|---|
| grid alignment OFF, no displacement | 39,690 | 1,350 | 3.40% |
| grid alignment OFF, displacement | 7,560 | **7,560** | **100%** |
| grid alignment OFF, all | 47,250 | 8,910 | 18.86% |
| grid alignment ON | 47,250 | **0** | 0% |

The axis here is whether the grid moved, not when in the run the comparison occurred. Read this way the result is sharper than the pooled rate suggests: on a displaced grid **every** comparison found the key codes changed, while on an undisplaced grid a change was detected in 1,350 of 39,690 comparisons.

Two distinct paths produce the changes, and they are not the same mechanism. **(A) Boundary-group rescaling at write time**: where a chunk boundary falls inside a group, the group's later members are filled by the next write; the group's extent does not move, but its shared scale is recomputed from the new members, so the codes of the earlier members can change. **(B) Grid displacement at eviction**: when the evicted span is not a multiple of the group size, group extents shift relative to surviving content and the groups re-form with different members. (A) needs only a chunk size that is not a multiple of the group size; (B) needs an eviction to occur at all. Alignment closes both. The 1,350 changed comparisons in the undisplaced population are **consistent with** write-time boundary rescaling (A) — the counter records that key codes changed on an undisplaced grid, not which group or scale changed, and we did not link the changed positions to write boundaries; the 7,560 in the displaced population are consistent with (B) but **we have not established that (A) is absent there**, and **we have not separated how much either path contributes to quality.**

Position does not protect content. In a separate A2 INT2 generation audit, the positionally fixed sink slot's stored key codes changed in 804 of 9,000 comparisons with alignment OFF, versus 0 of 9,000 with alignment ON. These counts aggregate three prompts, 30 layers, and repeated generation calls. The mechanism is that 4,680 mod 16 = 8 leaves the 292nd block straddling the sink/rolling boundary, so a slot that never moves still sits in a group that re-forms.

*Scope.* These counts come from audits inside live generation runs, so they describe what the deployed path does, not an isolated storage experiment; the storage-only diagnostic reported in §4.1 is a separate instrument, applying the storage operations alone to a clean BF16 chunk. The counter compares **key** codes, so nothing here is asserted about values or about whole tokens. The two audits are also at different bit-widths — the 47,250/8,910 figures are an A1 INT4 run, the 804/9,000 figures an A2 INT2 run — and are not to be read as one series.

### 4.4 The quality effect of alignment

Turning grid alignment on, with everything else fixed (n=10 prompts, paired):

| configuration | alignment ON − OFF | 95% CI | t | sign |
|---|---|---|---|---|
| A1, INT2 | +2.28 | [+0.23, +4.33] | +2.52 | 8/10 |
| A2, INT2 | **+11.50** | [+8.36, +14.64] | +8.28 | 10/10 |
| A1, INT4 | +1.08 | [+0.51, +1.65] | +4.26 | 9/10 |
| A4, INT4 | +1.08 | [+0.23, +1.92] | +2.89 | 10/10 |

A control at 21 frames, where **no eviction occurs**, shows +0.06 (±0.46, t=0.29): no quality difference is detected. This condition excludes eviction-driven grid displacement (path B), but it does **not** exclude the difference in group partitioning or the write-time boundary rescaling of path A — alignment changes how groups are formed from the first write, before any eviction. What this result means is that no alignment ON−OFF quality difference was detected under the 21-frame condition. It does not isolate path A on its own, and it provides no bound on the quality contribution of path B in longer runs where eviction occurs — the interval is for the whole ON−OFF contrast at 21 frames, not for the eviction path.

---

## 5 The Interaction

### 5.1 A complete 2×2

Grid alignment and the content sink were varied in a full factorial — {alignment ON, OFF} × {sink 0 frames, sink 3 frames} — with all four cells measured and prompts paired (INT2, n=10). The sink here is **3 frames, which is one chunk**; it is not the anchor budget of §6, which is counted in chunks.

| term | INT2 | INT4 |
|---|---|---|
| main effect, alignment | +6.89 [+4.84, +8.94] | +0.83 [+0.32, +1.35] |
| main effect, sink | +3.07 [+1.52, +4.61] | +1.28 [+0.47, +2.08] |
| **interaction** | **+9.22 [+5.85, +12.59]** | −0.50 [−1.12, +0.13] |
| sum of simple effects | +0.74 [−2.72, +4.20] | +2.60 [+1.58, +3.62] |
| combined effect | +9.96 [+6.54, +13.37] | +2.11 [+0.98, +3.23] |

At INT2 the two interventions measured one at a time, on the baseline configuration, sum to +0.74 with an interval spanning zero; applied together they give +9.96. The interaction term is larger than either main effect.

At INT4 **no interaction was detected in this sample**. We do not read this as additivity: the interval [−1.12, +0.13] is consistent with a small interaction of either sign, and we did not compare the statistical power of the two bit-widths.

### 5.2 The sink contrast depends on alignment

The clearest consequence is that the measured value of one intervention depends on the storage condition under which it is measured:

| | A2 − A1, INT2 | 95% CI | t | sign |
|---|---|---|---|---|
| alignment OFF | −1.54 | [−3.14, +0.06] | −2.18 | 8/10 |
| alignment ON | **+7.68** | [+4.86, +10.49] | +6.17 | 10/10 |

Under a displaced grid the sink shows a negative point estimate whose interval includes zero — **it cannot be called a degradation**, and neither is there evidence of benefit. Under an aligned grid the same sink yields +7.68 with the interval well clear of zero.

The consequence for reporting is that a measurement of the sink is not interpretable without the storage condition it was taken under. A measurement on a displaced grid and one on an aligned grid are not of the same quantity, even with the same model, prompts, bit-width and sink size.

### 5.3 What may and may not be presented

At INT2, reading each intervention as an independent causal contribution is not supported: that reading predicts +0.74 where +9.96 is observed. What remains available is (i) **ordered conditional improvements**, with the order stated — for example, on top of alignment, adding the sink gives +7.68 — and (ii) any presentation that **carries the interaction term explicitly**. A stacked decomposition that hides both the order and the interaction is what the data rules out, not stacked presentation as such.

### 5.4 Scope

All figures here are for the models, prompts (n=10), frame count (63) and quantizer (RTN, group 16 along tokens) stated in §3, on a single implementation. We have not tested whether the interaction persists under a different quantizer, group axis, or chunk size. The interaction is an estimate on this sample; §10 states what would change if it does not reproduce.

---

### 집필 메모 — 쓰지 않은 것

| 뺀 것 | 이유 |
|---|---|
| 앵커 분해(+7.68 / +3.50 / +0.00) | **§6**의 내용이다. 여기 끌어오면 sink 효과와 앵커 예산 효과가 섞인다 |
| "널리 쓰이는 구현", "기존 평가들" | 범위 주장이므로 §2에서만 다룬다. 양방향 모두 막혀 있다 — 공개 harness(`b4c0936`)의 패치는 같은 **구조**를 가지므로 "우리 패치뿐"이라고 쓸 수 없고, 그 드라이버는 캐시를 전체 출력 길이로 확장해 **퇴출 분기가 실행되지 않으므로** "공개 코드도 같은 문제를 겪는다"도 쓸 수 없다(인용 문서 §5.1–5.2) |
| 다른 구현과의 비교에서 **캐시 용량 정책** | 뺀 것이 아니라 **넣어야 할 것.** 이동 코드의 유무만큼 드라이버의 용량 정책이 조건을 결정한다(인용 문서 §5.6) |
| 노출량 비율(토큰당 재양자화 횟수) | 장부 계산으로 세 번 틀렸다. 실측 카운터만 쓴다 |
| 경로 A·B의 품질 기여 크기 | **미분리.** §4.3에 그대로 명시 |
| 자기 감사 서사("이전 감사가…") | 비유의를 효과 없음으로 쓴 과거 진술을 인용해 변호하는 꼴이 된다. **§5.2는 현재의 조건부 대비만 쓴다** |

### 확인 필요

1. ~~**A1 INT4 / A4 INT4의 CI**~~ — **확인됨** (2026-09-21). `gateK_aligned.csv`: A1 INT4 1.0809±0.5740 → [+0.51, +1.65], t 4.260, 9/10; A4 INT4 1.0769±0.8432 → [+0.23, +1.92], t 2.889, 10/10. 본문 A4 하한 +0.24 는 반올림 경계(0.2337) — **+0.23** 이 맞다 `[확인]`.
2. ~~§4.3 카운터의 단위·대상~~ — **확인됨.** `_requant_audit` 는 `causal_model.py:89` 에 정의되고 `state["k"]["q"]` 만 비교하며 단위는 `n_cmp`(생존 슬롯 비교). 804 = 이동 없음 306 + 이동 있음 498 도 원자료와 일치 `[확인]`.
3. ~~A2의 804 단위~~ — **해결.** 같은 계측기의 sink 슬롯 K 코드 변화 검출 횟수이고 분모는 9,000(정렬 ON은 0/9,000). 이동 없음 306 + 이동 있음 498. **A2는 INT2 실행**이므로 INT4인 47,250 집계와 한 계열로 묶지 않는다 `[확인]`.
4. 용어: "displaced grid" vs "misaligned grid" 중 하나로 통일.

### 검증 기록 (2026-09-21, 2차)
- §4.4: "퇴출 없음 → 효과 없음이 예측" 삭제. 21프레임 대조는 경로 B(퇴출 격자 이동)만 제외하고 경로 A(쓰기 시 경계 재스케일)와 그룹 분할 차이는 제외하지 못한다 — §4.3 과 모순이었다.
- §4.3: 1,350건을 "(A)에서만 올 수 있다" → "(A)와 부합한다". 카운터는 위치·scale·쓰기 경계를 연결하지 않는다.
- (3차) §4.4 의 "bounds what the eviction path contributes" 삭제 — 제가 2차 수정에서 새로 넣은 과장이었다. 21프레임 CI 는 그 조건의 ON−OFF 전체 대비 구간이고 퇴출 경로의 상한이 아니다. §4.3 41행의 "only where a boundary group was re-formed" 를 관측(1,350/39,690)만으로. §5.2 제목을 "The sink contrast depends on alignment" 로.
