# §1 · §2 · §6 · §7 · §8 · §9 · §10 · §11 — 초고 v1 (영문)

작성 2026-09-20. 안 A(분석 9쪽) 번호 기준. §3·§4·§5는 별도 파일.
근거: `claim_table.md`, `S6_FINDINGS.md`, `L1_storage_design.md`, `G1_storage_survey.md`.

> **집필 원칙(본문 아님).** 정정 과정 전체를 본문에 옮기지 않는다. 최종적으로 검증된 주장과 그 한계만 적는다. `claim_table`에서 **불가**인 행의 내용은 어떤 문장에도 넣지 않고, **조건부**인 행은 조건 열의 문구를 함께 쓴다.

---

## 1 Introduction

In streaming video diffusion, the key–value cache grows with the generated horizon and soon exceeds the size of the model itself, so compressing it is not optional. The question this paper asks is not how much quality a given bit-width costs, but **how much that answer depends on the implementation that stores the cache**.

We study a rolling cache quantized with round-to-nearest in groups of 16 tokens. Because a chunk is 4,680 tokens — 292.5 groups — eviction displaces surviving content relative to the group boundaries, and the content is re-quantized on a grid it was not quantized on. Round-to-nearest is idempotent on a fixed grid and is not idempotent once the grid moves.

The consequence is not a small numerical residue. Measured under a displaced grid, a content sink shows a negative point estimate at INT2 whose interval includes zero; measured under an aligned grid, the same sink gives **+7.68 [+4.86, +10.49]** MUSIQ. In a full 2×2, grid alignment and the sink interact with a term of **+9.22 [+5.85, +12.59]**, larger than either main effect, while the two interventions measured one at a time sum to +0.74 with an interval spanning zero.

**Contributions.**
1. We identify grid displacement in rolling group-quantized caches and characterise it at the level of stored codes, separating the re-quantization call, the code change, the change in dequantized value, and the change in error against the original (§4).
2. We show that this storage condition changes what an evaluation concludes: the sink effect's sign, and an interaction larger than either main effect (§5).
3. We separate the size of the Jensen bias correction's benefit from the contribution of its per-key structure, and report where its derivation's premise fails (§7).
4. We give a measurement protocol — random-number convention, storage path, memory accounting, statistical unit, provenance — with the scope of each verification stated (§3).

**We do not claim that this failure mode is unknown.** A survey of public implementations (§2) found that the two we could read avoid it, and one names it explicitly. Our scope is the dependence itself and its size, not its discovery.

*Figure 1.* The 2×2 at INT2 with intervals on every bar: alignment alone +2.28 [+0.23, +4.33]; sink alone −1.54 [−3.14, +0.06], interval crossing zero; both +9.96 [+6.54, +13.37]. INT4 panel alongside, where no interaction was detected.

---

## 2 Background and Related Work

**Streaming video diffusion.** Self-Forcing and its successors generate chunk by chunk with a rolling KV cache; LongLive retrains for a short cache; Deep Forcing adjusts sink and window without retraining. These works set the cache configurations we vary.

**KV-cache quantization.** KIVI quantizes keys per channel and values per token with a full-precision residual window. Quant VideoGen (QVG) quantizes a streaming video cache to 2 bits using learned codebooks with a fused dequantizing attention kernel. A recent study surveys 33 KV-compression methods and reports the transient BF16 reconstruction buffer as their dominant practical cost.

**Correction and bit allocation.** Prior work derives a bias correction for the softmax under key quantization and applies it in a score-modification kernel; separate work in the LLM setting argues keys require more bits than values. We use the former as the correction under study (§7) and the latter as background for our bit allocation.

### 2.1 Storage behaviour of public implementations

Because our results turn on a storage property, we read the public code rather than relying on the papers' descriptions. The unit of the survey is a **path** — repository, commit, configuration, backend and call site — not a project name, since paths within one project differ.

| path | storage behaviour | verification |
|---|---|---|
| KIVI, `876b4d2` | when the residual buffer fills, that block is quantized once and concatenated to the existing packed tensor. No re-quantization; the design has no eviction (append-only) | static code |
| QVG, `0601468` | `ChunkedKVCache` quantizes once per span. Eviction moves no data and re-quantizes nothing; chunk positions are fixed | static code |
| QVG, paper §5.1 | states the design intent: quantize the KV cache once per chunk and **avoid re-compression drift** | paper |
| the Self-Forcing patch used for our measurements | re-quantizes surviving content on a displaced grid at every eviction | static code and **execution** |

Two conclusions follow, and the second is a limit on the first. **(a)** The failure mode is neither unknown nor universal: QVG names it and both implementations we could read avoid it. We therefore make no priority or novelty claim about identifying it. **(b)** Our survey verified the two implementations **statically**; we did not run them. For the correction work we could not locate a public implementation at all, and Deep Forcing was not examined.

**Positioning.** What we contribute is not the observation that storage can drift, but a measurement of **how much a storage condition changes an evaluation's conclusion** in a setting where it was not controlled — including a sign change — together with the protocol needed to keep such conditions visible. We do not assert that published evaluations elsewhere are affected: within our survey the only path exhibiting the condition is the patch we used ourselves.

---

## 6 Decomposing the Anchor Effect

An earlier phase of this work reported a **+11.20** MUSIQ gain from an oracle that selects anchor chunks adaptively by a contamination criterion, and attributed it to the selection policy. Re-examined, the oracle chose the **same set as the fixed policy {0,1,2} in all 140 decisions**, and its per-prompt scores are identical to the fixed policy's to the decimal — the two are the same run.

| component | value | 95% CI |
|---|---|---|
| anchor presence (one fixed chunk) | +7.68 | [+4.86, +10.49] |
| anchor budget, 1 → 3 chunks | +3.50 | [+1.51, +5.50] |
| **adaptive selection** | **+0.00** | 140/140 identical selections |

The components sum to +11.18 against the +11.20 originally reported. Here "identical" is a stronger statement than "not detected": the selection histories match exactly, so for this run the two arms are the same computation. We use the word only in this case.

Two consequences. The deployable form is a **fixed** anchor set, since nothing is lost by dropping the oracle. And because total capacity is held fixed, increasing the anchor budget necessarily shortens the recent context — so the +3.50 is an **allocation effect between anchors and recent context under a fixed budget**, not an effect of anchor count per se.

---

## 7 The Correction and Its Scope

### 7.1 Effect size

The Jensen bias correction recovers **+10.04 [+3.09, +16.99]** of the INT2 loss, about a third of the total +29.23. Two independent metrics agree on the fraction recovered: 34% by MUSIQ and 32% by `subject_consistency`.

### 7.2 What the per-key structure contributes

Replacing the per-key bias with a flattened control produced no detected difference, in MUSIQ or in `subject_consistency`. The claim this supports is narrow: the flattened control is constant **per (query, head)** and therefore removes only the per-key structure while preserving the scale differences between heads. **Whether a single global constant would suffice is a different claim and is untested** — we did not run that sweep.

### 7.3 Concentration

Within a head, the dispersion of the per-key bias is small: σ_b/μ_b has median **0.165** against an independent-channel prediction of **0.153**, with Pearson r = +0.732 over 10 prompts and 90 records, and an effective dimension of **d_eff ≈ 11.7** rather than the nominal 128 — the contribution is carried by roughly a dozen outlier channels.

The prediction is nevertheless **systematically low**. Treating prompts as the unit of replication, the residual (observed − predicted) is **+0.0205 [+0.0134, +0.0276]**, t = +6.52, positive in 10 of 10 prompts. An earlier single-prompt measurement reported 0.150 against 0.151 and described the agreement as near-exact; that closeness was a property of that prompt, whose residual is the second smallest of the ten. We retract the characterisation and keep the finding it was attached to: dispersion is small and concentration is real, but **the prediction is not accurate per observation** — its residual is the same order as the predicted value.

### 7.4 Correction strength

λ = 1 is the theoretical default and no evidence places the optimum below it: λ = 0.5 and λ = 1 are not distinguished ([−0.0057, +0.0095]). Above λ = 1 the metrics disagree in **order**, not merely in magnitude: MUSIQ places λ = 4 at its point-estimate maximum while `subject_consistency` puts it **below the uncorrected baseline** (−0.1602, t = −18.27). A single metric is therefore not sufficient to select λ. We record the disagreement and its size and do not adjudicate between the metrics.

### 7.5 Where the derivation's premise fails

Per channel, the second-order approximation is adequate: |a| has median 0.003–0.107 with relative error below 1e−2 at the 95th percentile. **The quantity actually subtracted from the logit is the sum over 128 channels**, and there the approximation is not adequate in outlier heads: in layer 29 its relative error reaches a 95th percentile of 12.2–25.7 (median across prompts; full range 8.7–32.0), reproduced in all 10 prompts, while layers 0 and 15 stay at 0.02–0.17. This is a property of outlier layers, not of the approximation in general.

In those heads the correction does not re-weight the cache — it removes it. For layer 29 head 9 the exact bias is 162–328, and the cache attention mass falls from 0.89–0.99 to effectively zero; this is **directly observed** in the correction run, where that head's self-attention mass is 1.0. The counterfactual used for the exact form was first validated against the approximate run, reproducing its measured mass with a median absolute error of 0.024. The scope is narrow: **3 of the 12 heads in layer 29**; the remaining heads and layers 0 and 15 have biases below 1 and barely move.

### 7.6 Implementation status

The shipped series implementation of the exact form diverges beyond its radius of convergence, reaching an absolute difference of 2.3e+27 in layer 29. A numerically stable replacement is accurate (≤ 7.9e−6 over the full range) but at least 25× slower — in a three-prompt trial it did not finish the first prompt in 8 hours 23 minutes. **At present neither form is usable**: the shipped one is wrong in the tail and the stable one cannot be run. We therefore report the approximate form as the one under study and treat the exact form as a diagnostic only.

---

## 8 What the Correction Does Not Reach

After correction, the residual loss is +19.18, which splits as +4.36 against a teacher-forced control and +14.82 between that control and free-running generation. **This is an arithmetic decomposition of metric differences, not a causal one.** We write the second term as the **additional rollout loss relative to a BF16-history control** and do not call it contamination: the teacher-forced contrast cannot separate the contributions of the history, the query and the current chunk's quantization, nor their interactions.

Allocating the residual among candidate causes requires a presentation consistent with §5.3 — ordered and conditional, or carrying the interaction explicitly — and we have not chosen one. Separately, the diagnostics that measured only the re-quantization error are suspended until the original keys are preserved for comparison; the quantity they measured is identically zero on a fixed grid (§4.1) and so cannot speak to the error against the original.

---

## 9 Measured Cost

### 9.1 Storage and execution paths

Holding the model, prompts (n = 10), frame count, seed and quantizer (RTN INT2) fixed and varying the storage policy and execution path:

| arm | re-quantization input | grid | MUSIQ | vs P1 | 95% CI | cache residency |
|---|---|---|---|---|---|---|
| P1 rolling | dequantized value | displaced | 41.98 | — | | 3.397 GB |
| P2 aligned | dequantized value | fixed | 44.26 | +2.28 | [+0.24, +4.33] | 3.402 GB |
| P3 packed fused | (none) | fixed | 44.63 | +2.65 | [+1.09, +4.21] | **1.834 GB** |
| P4 BF16 master | original BF16 | displaced | 43.96 | +1.99 | [−0.07, +4.04] | 9.436 GB |

**P2 and P3 showed higher MUSIQ than P1. P4's improvement was not detected, and the differences among P2, P3 and P4 were not detected** (P2−P3 [−1.58, +0.85]; P2−P4 [−0.37, +0.96]; P3−P4 [−0.73, +2.05]).

Two limits on reading this table. The arms are **not a complete factorial**: the cell (original BF16 input, fixed grid) was not run, and P3 lies outside that design, having no re-quantization input and additionally using a different quantization kernel and attention path. Accordingly **P3's +2.65 is attributed to the P3 path as a whole**, not to its storage policy; a tensor-level comparison bounded the attention-path difference on the inputs it tested, but that is not a bound over all calls or on the final metric.

On memory the arms separate along a different axis. Relative to P1, only P3 reduces residency further (P2 differs from P1 by 0.999×); relative to BF16 (6.038 GB), P3 is 3.292× and P2 is 1.775× smaller. **These are different comparisons from the quality contrasts above and must not be summarised as one pair.**

P4 is larger than BF16 alone because it retains a BF16 master alongside the quantized state (6.038 + 3.397 = 9.436 GB). This is a property of **our implementation of that design inside Self-Forcing**; we did not measure the original system that uses it.

### 9.2 One final configuration

For a single configuration — A4 window, packed fused path, RTN K4V2, in-kernel bias at λ = 1 — measured against our BF16 baseline on a fixed source, 10 prompts, 63 frames:

| | BF16 | final | change |
|---|---|---|---|
| cache residency | 6.038 GB | 1.510 GB | **4.00×** |
| peak allocated | 15.33 GB | 10.81 GB | |
| `inference()` total | 465.6 s | 545.2 s | **+17.1%** |
| MUSIQ | 71.204 | 71.381 | +0.177 [−1.046, +1.400] |

Four qualifications belong with this table. **(i)** The MUSIQ difference is **not detected, which is not equivalence** — no equivalence test was run, and the sign of a point estimate carried by 4 of 10 prompts should not be read. **(ii)** The two arms differ in cache configuration (A4 versus A1), so this is not the effect of quantization alone: a window change and a sink are included. **(iii)** MUSIQ is the only metric measured for this arm. **(iv)** The latency includes VAE decoding, which is common to both arms and therefore dilutes the relative overhead; the attention path's own overhead is larger and we did not separate it.

The residency decomposes as 0.647 GB of packed segments plus **0.863 GB of BF16 pending chunk — 57% of the total** — because the fused path does not pack the chunk currently being written. The remaining 276,960 B is accounted for: 276,480 B from rounding the scale-block count (292.5 → 293) across three segments and both tensors, and 480 B of index tensors. Note that the two large components are closed-form quantities compared against the measurement, not a decomposition derived from it.

Measured in isolation on a single harness, the fused kernel costs **+26.6%** against FlashAttention-2 (8.325 ms versus 6.575 ms). This measures a different thing from the end-to-end figure and the two should not be conflated; we quote the end-to-end +17.1% as the headline and the kernel figure as its basis.

---

## 10 Limitations and Discussion

Our results come from the Wan 1.3B family, 10 to 20 prompts, and RTN-family quantizers, on one implementation. Three diagnostics rest on narrower ground still: the head-level bias statistics of §7.5 come from a single prompt in part, the two re-quantization paths of §4.3 were not separated in their quality contribution, and the attribution of §9.1's +2.65 between storage policy and execution path remains open.

Two storage emulations differ between the codebases we touched — one retains a BF16 master and does not accumulate, the other accumulates — so cross-model comparisons carry that factor.

During this work we produced seven instrumentation defects and four reporting errors; both lists appear in the appendix with what each affected. Only one instrumentation defect changed a result, and it did so by silently preventing a multi-prompt run from generalising. The common cause in nearly every case was the same: a new instrument was used on the quantity of interest before it was run on a case whose answer was already known.

The recommendation we can support is narrow and practical. **Report the storage and re-quantization conditions of a quantized KV cache alongside its bit-width, and where feasible report results under both aligned and unaligned storage.** A bit-width alone does not identify the configuration that was measured.

---

## 11 Conclusion

The quality cost of a low-bit KV cache is not a function of bit-width alone. In the implementation we studied, whether stored content is re-quantized on a moving grid changes the measured effect of a cache-configuration choice — including its sign — and produces an interaction larger than either intervention's individual effect. Storage conditions are part of the configuration being evaluated, and results that do not state them describe an experiment that cannot be identified.

---

### 확인 필요 (집필 세션)

1. **§2의 인용** — 각 선행연구를 arXiv 번호로 확정하고, KIVI·QVG 커밋 해시를 실제 조사표와 대조할 것. 본문의 `876b4d2`·`0601468`은 조사표 인용이다 `[간접]`.
2. **§7.1의 34%** — 10.04/29.23 = 34.3% `[계산]`. subject_consistency 32%는 원자료 확인 필요 `[간접]`.
3. **§9.2의 peak reserved**를 표에서 뺐다(BF16 22.01 / 최종 16.55). 넣을지 결정할 것.
4. **§10의 "일곱·넷"** — 부록 목록과 개수가 맞는지 대조.
5. **§1 Figure 1 캡션**에 INT4 패널이 "상호작용 비검출"을 보이기 위한 것이며 INT4 주효과를 독립 결과로 읽지 말라는 문구를 넣을 것.
6. 전체 용어 통일: displaced grid / storage path / cache residency / not detected vs equivalent / re-quantization.
