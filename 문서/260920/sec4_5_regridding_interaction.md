# §4 Regridding · §5 The Interaction — 초고 v1 (영문)

작성 2026-09-20. 근거: `claim_table.md` G1~G6·Q1~Q6, `S6_FINDINGS.md` §1.1~§1.4.
번호는 **안 A(분석 9쪽)** 기준이다(구 문서의 "§4.5 격자"와 "§6 상호작용"에 해당).

> **집필 메모.** §4는 메커니즘, §5는 그 메커니즘이 평가 결론에 미치는 영향이다. **§4에 품질 인과를 넣지 않고**, §5에 메커니즘 설명을 되풀이하지 않는다.

---

## 4 Regridding in Rolling Caches

### 4.1 Where the grid moves

The cache is quantized in groups of 16 tokens along the token axis, each group carrying its own scale. A chunk is 4,680 tokens, which is 292.5 groups. Eviction therefore does not remove a whole number of groups: surviving content is displaced by 8 tokens relative to the group boundaries, and every group re-forms with different members.

Round-to-nearest is **idempotent on a fixed grid**: re-quantizing already-quantized content reproduces it bit for bit (relative difference 0.000e+00). Once the grid is displaced by 8 tokens, it is not: the relative difference is 5.47e−2 at INT4 and 2.01e−1 at INT2.

*Scope.* This idempotence is a property of the RTN implementation and numeric conditions we verified — symmetric scaling, group size 16 along tokens, the accumulation order and dtype of our path. It is not asserted of RTN in general; a different clipping rule, scale formula or accumulation order can behave differently.

### 4.2 Five quantities, kept separate

Throughout we distinguish (i) whether a **re-quantization call** occurs, (ii) whether the stored **integer codes** change, (iii) whether the **dequantized value** changes, (iv) whether the **error against the original** grows, shrinks or stays, and (v) whether any of this **affects output quality**. None of these implies the next. In particular a re-quantization call on a fixed grid changes nothing (§4.1), and a change in dequantized value does not by itself mean a larger error, since the error norm may decrease while the error's direction and correlation structure change.

This section establishes (i)–(iii) directly. Quality effects are the subject of §4.4 and §5; (iv) is measured only for the storage-error diagnostic described below.

### 4.3 How often codes actually change

We track content — not slot indices — across the run and count how often a surviving token's stored codes differ from what they were, over 3 prompts × 63 frames × 30 layers. The 47,250 comparisons divide into two populations with **different denominators**, according to whether the re-quantization call in question acts on a grid that has been displaced:

| | re-quantization calls | calls in which stored codes changed | rate |
|---|---|---|---|
| grid alignment OFF, no displacement | 39,690 | 1,350 | 3.40% |
| grid alignment OFF, displacement | 7,560 | **7,560** | **100%** |
| grid alignment OFF, all calls | 47,250 | 8,910 | 18.86% |
| grid alignment ON | 47,250 | **0** | 0% |

The axis of this table is whether the grid moved, not when in the run the call occurred; 1,350 and 7,560 are counts out of different totals and must not be read as parts of 47,250. Read this way the result is sharper than a pooled rate suggests: **every** call on a displaced grid changed the stored codes, and calls on an undisplaced grid changed them only where a partially filled boundary group was re-formed.

Two distinct paths produce the changes. **(A) Boundary-block rescaling at write time**: a partially filled group at the tail is re-formed when the next write fills it. **(B) Grid displacement at eviction**: when the evicted span is not a multiple of the group size, the surviving groups in the rolling region re-form. Alignment closes both. The 1,350 changes in the undisplaced population can only come from (A); the 7,560 in the displaced population are consistent with (B) but **we have not established that (A) is absent there**, and **we have not separated how much either path contributes to quality.**

Position does not protect content. In the A2 configuration (sink of one chunk), the **positionally fixed sink slot is re-quantized 804 times**, because 4,680 mod 16 = 8 leaves the 292nd block straddling the sink/rolling boundary.

*Scope.* These counts come from a storage-only measurement: a clean BF16 chunk is passed through the storage regime alone, so generation feedback is excluded. In the sink-free A1 configuration the window holds 7 slots, so a slot is re-quantized at most 6 times and the storage error saturates there — **that ceiling comes from the window length, not from the mechanism**, and it does not apply when a fixed sink is present (A2: 804) or when repeated denoising calls are counted.

### 4.4 The quality effect of alignment

Turning grid alignment on, with everything else fixed (n=10 prompts, paired):

| configuration | alignment ON − OFF | 95% CI | t | sign |
|---|---|---|---|---|
| A1, INT2 | +2.28 | [+0.23, +4.33] | +2.52 | 8/10 |
| A2, INT2 | **+11.50** | [+8.36, +14.64] | +8.28 | 10/10 |
| A1, INT4 | +1.08 | [+0.51, +1.65] | +4.26 | 9/10 |
| A4, INT4 | +1.08 | [+0.24, +1.92] | +2.89 | 10/10 |

A control at 21 frames, where **no eviction occurs**, shows +0.06 (±0.46, t=0.29): no effect is detected. This is what the mechanism predicts — with no eviction there is no grid displacement, and the difference between the two settings is only how blocks are partitioned.

---

## 5 The Interaction

### 5.1 A complete 2×2

Grid alignment and the content sink were varied in a full factorial — {alignment ON, OFF} × {sink 0, 3} — with all four cells measured and prompts paired (INT2, n=10):

| term | INT2 | INT4 |
|---|---|---|
| main effect, alignment | +6.89 [+4.84, +8.94] | +0.83 [+0.32, +1.35] |
| main effect, sink | +3.07 [+1.52, +4.61] | +1.28 [+0.47, +2.08] |
| **interaction** | **+9.22 [+5.85, +12.59]** | −0.50 [−1.12, +0.13] |
| sum of simple effects | +0.74 [−2.72, +4.20] | +2.60 [+1.58, +3.62] |
| combined effect | +9.96 [+6.54, +13.37] | +2.11 [+0.98, +3.23] |

At INT2 the two interventions measured one at a time, on the baseline configuration, sum to +0.74 with an interval spanning zero; applied together they give +9.96. The interaction term is larger than either main effect.

At INT4 **no interaction was detected in this sample**. We do not read this as additivity: the interval [−1.12, +0.13] is consistent with a small interaction of either sign, and we did not compare the statistical power of the two bit-widths.

### 5.2 The sink effect changes sign

The clearest consequence is that the measured value of one intervention depends on the storage condition under which it is measured:

| | A2 − A1, INT2 | 95% CI | t | sign |
|---|---|---|---|---|
| alignment OFF | −1.54 | [−3.14, +0.06] | −2.18 | 8/10 |
| alignment ON | **+7.68** | [+4.86, +10.49] | +6.17 | 10/10 |

Under a displaced grid the sink shows a negative point estimate whose interval includes zero — **it cannot be called a degradation**, but neither is there evidence of benefit. Under an aligned grid the same sink yields +7.68 with the interval well clear of zero.

An earlier audit of this work recorded "the sink has no effect at INT2" as a finding. That conclusion was measured on a displaced grid. It is not that the audit erred in its analysis; the quantity it measured was conditioned on a storage property that was not part of the reported configuration.

### 5.3 What may and may not be presented

At INT2, reading each intervention as an independent causal contribution is not supported: that reading predicts +0.74 where +9.96 is observed. What remains available is (i) **ordered conditional improvements**, with the order stated — for example, on top of alignment, adding the sink gives +7.68 — and (ii) any presentation that **carries the interaction term explicitly**. A stacked decomposition that hides both the order and the interaction is what the data rules out, not stacked presentation as such.

### 5.4 Scope

All figures here are for the models, prompts (n=10), frame count (63) and quantizer (RTN, group 16 along tokens) stated in §3, on a single implementation. We have not tested whether the interaction persists under a different quantizer, group axis, or chunk size. The interaction is an estimate on this sample; §10 states what would change if it does not reproduce.

---

### 집필 메모 — 쓰지 않은 것

| 뺀 것 | 이유 |
|---|---|
| 앵커 분해(+7.68 / +3.50 / +0.00) | **§6**의 내용이다. 여기 끌어오면 sink 효과와 앵커 예산 효과가 섞인다 |
| "널리 쓰이는 구현", "기존 평가들" | 조사에서 P1형 경로는 우리 패치 하나였다(§2에서 다룬다) |
| 노출량 비율(토큰당 재양자화 횟수) | 장부 계산으로 세 번 틀렸다. 실측 카운터만 쓴다 |
| 경로 A·B의 품질 기여 크기 | **미분리.** §4.3에 그대로 명시 |

### 확인 필요

1. **A1 INT4 / A4 INT4의 CI**를 ±0.57·±0.84에서 구간 표기로 바꿔 넣었다 — `gateK_aligned.csv` 원본과 대조할 것 `[간접]`.
2. §5.2의 "audit" 서술 — 감사 결론 번호(C7·C13)를 본문에 넣을지 부록으로 뺄지. 자기 정정 서사를 본문에서 길게 끌지 않는다는 원칙과 맞출 것.
3. §5.1 표의 INT4 열은 상호작용 비검출을 보이기 위한 것이다. **INT4 주효과를 독립 결과로 읽히지 않게** 캡션에서 못 박을 것.
4. 용어: "displaced grid" vs "misaligned grid" 중 하나로 통일.
