# §1 · §2 · §6 · §7 · §8 · §9 · §10 · §11 — 초고 v2 (영문)

> **v2 (2026-09-21).** §7·§9 를 **P3(packed, once-per-chunk) + λ1 기준선** 중심으로 재구성. P1(rolling) 결과는 경로 의존성을 보이는 대조로 남긴다. 근거: `지시서/S8_PLAN_iso_memory_v1.md` §4·§6, `results/session8/`.

작성 2026-09-20. 안 A(분석 9쪽) 번호 기준. §3·§4·§5는 별도 파일.
근거: `claim_table.md`, `S6_FINDINGS.md`, `L1_storage_design.md`, `G1_storage_survey.md`.

> **집필 원칙(본문 아님).** 정정 과정 전체를 본문에 옮기지 않는다. 최종적으로 검증된 주장과 그 한계만 적는다. `claim_table`에서 **불가**인 행의 내용은 어떤 문장에도 넣지 않고, **조건부**인 행은 조건 열의 문구를 함께 쓴다.

---

## 1 Introduction

In streaming video diffusion the key–value cache holds the context retained for generation, and its size scales with that retained length; at the context lengths used here it is a substantial share of the memory budget, which is what motivates compressing it. The question this paper asks is not how much quality a given bit-width costs, but **how much that answer depends on the implementation that stores the cache**.

We study a rolling cache quantized with round-to-nearest in groups of 16 tokens. Because a chunk is 4,680 tokens — 292.5 groups — eviction displaces surviving content relative to the group boundaries, and the content is re-quantized on a grid it was not quantized on. In the implementation, dtype and grid configuration we tested, re-quantization on a fixed grid reproduces the stored content exactly, and re-quantization after the grid moves does not (§4.1).

The consequence is not a small numerical residue. Measured under a displaced grid, a content sink shows a negative point estimate at INT2 (−1.54) whose interval includes zero — no benefit and no degradation is detected; measured under an aligned grid, the same sink gives a detected benefit of **+7.68 [+4.86, +10.49]** MUSIQ. What changes between the two conditions is the sign of the point estimate and whether a benefit is detected, not a demonstrated reversal from harm to benefit. In a full 2×2, grid alignment and the sink interact with a term of **+9.22 [+5.85, +12.59]**, larger than either main effect, while the two interventions measured one at a time sum to +0.74 with an interval spanning zero.

**Contributions.**
1. We identify grid displacement in rolling group-quantized caches and characterise it at the level of stored codes, separating the re-quantization call, the code change, the change in dequantized value, and the change in error against the original (§4).
2. We show that this storage condition changes what an evaluation concludes: the sign of the sink's point estimate and whether its benefit is detected, and an interaction larger than either main effect (§5).
3. We measure the Jensen bias correction on a storage path that does not re-quantize (P3) and on one that does (P1): its benefit at INT2 is **+20.3** on the former and **+10.0** on the latter, so a correction's measured value depends on the path it is evaluated on. We separate that benefit from the contribution of its per-key structure and report where its derivation's premise fails (§7).
4. We give a measurement protocol — random-number convention, storage path, memory accounting, statistical unit, provenance — with the scope of each verification stated (§3).

**We do not claim that this failure mode is unknown.** A survey of public implementations (§2) found that the two we could read avoid it, and one names it explicitly. Our scope is the dependence itself and its size, not its discovery.

*Figure 1.* The 2×2 at INT2 with intervals on every bar: alignment alone +2.28 [+0.23, +4.33]; sink alone −1.54 [−3.14, +0.06], interval crossing zero; both +9.96 [+6.54, +13.37]. INT4 panel alongside, where no interaction was detected.

---

## 2 Background and Related Work

> **인용 상태 (CITATIONS_verified_v1 기준).** `[확인]` 원문·저장소 직접 확인: QVG(2602.02958, `0601468`), UCSD 33-method(2603.27469, harness `b4c0936`), Tuncer et al.(2605.26266, 코드는 찾지 못함), KIVI(2402.02750, `876b4d2`). `[미확인]` 원문을 열지 않아 근거 위치가 없음: Self-Forcing, LongLive, Deep Forcing(2512.05081), KV-AdaQuant(2502.15075), SkyReels-V2, MAGI-1, VBench, MUSIQ, FlashAttention-2. 미확인 항목에 대한 진술("LongLive retrains for a short cache", "Deep Forcing adjusts sink and window without retraining")은 **제출용 문장으로 확정하지 않는다.**

**Streaming video diffusion.** Self-Forcing and its successors generate chunk by chunk with a rolling KV cache; LongLive retrains for a short cache; Deep Forcing adjusts sink and window without retraining. These works set the cache configurations we vary.

**KV-cache quantization.** KIVI quantizes keys per channel and values per token with a full-precision residual window. Quant VideoGen (QVG) quantizes a streaming video cache to 2 bits with k-means centroids (K = 256, uint8 indices) and progressive residual quantization; its two settings use blocks of 64 (QVG) or 16 (QVG-Pro), and a fused kernel dequantizes and adds back the assigned centroids. It is not scalar RTN, so its quantizer differs from ours in kind, not only in bit-width. A 33-method empirical study on Self-Forcing reports that several methods compress the cache substantially yet still exceed BF16 peak VRAM, because the integration reconstructs dense BF16 tensors during attention reads and refresh.

**Correction and bit allocation.** Tuncer et al. derive a bias correction for the softmax under key quantization and apply it in a score-modification kernel; separate work in the LLM setting argues keys require more bits than values. We use the former as the correction under study (§7) and the latter as background for our bit allocation. One contrast bears directly on this paper's condition: Tuncer et al. group channels within a token (g = 32), whereas we group along the token axis (16 tokens). A group that does not span tokens cannot be re-formed by eviction. This implication is our reading of their description; we located no public implementation to confirm it.

### 2.1 Storage behaviour of public implementations

Because our results turn on a storage property, we read the public code rather than relying on the papers' descriptions. The unit of the survey is a **path** — repository, commit, configuration, backend and call site — not a project name, since paths within one project differ.

| path | storage behaviour | verification |
|---|---|---|
| KIVI, `876b4d2` | when the residual buffer fills, that block is quantized once and concatenated to the existing packed tensor. No re-quantization; the design has no eviction (append-only) | static code |
| QVG, `0601468` | `ChunkedKVCache` quantizes once per span. Eviction moves no data and re-quantizes nothing; chunk positions are fixed | static code |
| QVG, paper §5.1 | states the design intent: quantize the KV cache once per chunk and **avoid re-compression drift** | paper |
| the Self-Forcing patch used for our measurements | re-quantizes surviving content on a displaced grid at every eviction | static code and **execution** |

Two conclusions follow, and the second is a limit on the first. **(a)** The failure mode is neither unknown nor universal: QVG names it and both implementations we could read avoid it. We therefore make no priority or novelty claim about identifying it. **(b)** Our survey verified the two implementations **statically**; we did not run them. For the correction work we could not locate a public implementation at all, and Deep Forcing was not examined.

**Positioning.** What we contribute is not the observation that storage can drift, but a measurement of **how much a storage condition changes an evaluation's conclusion** in a setting where it was not controlled — including the sign of a point estimate and whether a benefit is detected — together with the protocol needed to keep such conditions visible.

The public harness of the 33-method study (`b4c0936`) applies a patch whose path restores the whole cache, shifts it, and re-quantizes it, with no group-alignment code — the same **structure** as the P1 path we measured; our code differs from that patch by 498 added and 7 removed lines, so we neither use it unchanged nor introduced the structure. Whether the structure produces the condition depends on the driver as much as on the patch: that harness pre-extends the cache to at least the full output length before generation, so on the dependency we compared (`33593df`) its RTN path is statically traced **not to evict within the configured generation length**, and eviction-driven grid displacement does not occur there. What can occur is the write-time path (§4.3 A): a 4,680-token boundary falls inside a 16-token group on alternate chunks, and re-quantizing the whole cache after the next write can change the earlier members' codes — shown by a constructed CPU case with the public quantizer, not measured in a model run. We could not identify from the available materials which Self-Forcing commit the published runs used. We therefore assert neither that published evaluations are affected nor that the condition is unique to our patch; when comparing implementations, the **cache-capacity policy of the driver** must be recorded alongside the quantizer.

---

## 6 Decomposing the Anchor Effect

An oracle that selects anchor chunks adaptively by a contamination criterion yields **+11.18 [+7.59, +14.76]** MUSIQ over the no-anchor (`oldest`) baseline in the four-arm control run. In that run the oracle selected **the same set as the fixed policy {0,1,2} in all 140 decisions**, and the two arms' per-prompt scores agree to the decimal.

The table is an **ordered sequence of conditional contrasts** — each row is measured on top of the rows above it, in this order, and the values are not interchangeable main effects.

| component | value | 95% CI / supporting evidence |
|---|---|---|
| anchor presence (one fixed chunk) | +7.68 | [+4.86, +10.49] |
| anchor budget, 1 → 3 chunks | +3.50 | [+1.51, +5.50] |
| **adaptive selection** | **+0.00** | not an interval: the two arms made identical selections on all 140 decisions observed |

The components sum exactly to the +11.18 total of the same run, because the third component is identically zero. (An earlier oracle run against a different baseline gave +11.20; it is a separate comparison with a different baseline arm and is not the total of this decomposition.) The zero entry is not a non-detection: the selection histories match exactly on the 140 decisions we observed, so on these inputs the adaptive rule returned what the fixed rule returns. **This does not establish that adaptive selection is unnecessary on other inputs**, and we did not test inputs on which the two rules diverge.

Because total capacity is held fixed, increasing the anchor budget necessarily shortens the recent context — so the +3.50 is an **allocation effect between anchors and recent context under a fixed budget**, not an effect of anchor count per se.

---

## 7 The Correction and Its Scope

### 7.1 Effect size — and its dependence on the path

We evaluate the correction (Tuncer et al., λ = 1) on two paths that store the same content differently: **P3**, the packed once-per-chunk path (fused kernel, no re-quantization), and **P1**, the rolling path that re-quantizes on a displaced grid (dequantize→FA2). Same model, prompts (n = 10), 63 frames, seed, A1 window; all arms in one batch per path.

| INT2, A1 | λ = 0 | λ = 1 | λ1 − λ0 (95% CI) | fraction of BF16 gap recovered |
|---|---|---|---|---|
| **P3 packed** | 44.63 | **64.95** | **+20.32 [+13.57, +27.07]**, 10/10 | 76% of 26.58 (MUSIQ); 72% of 0.225 (`subject_consistency`: +0.162 [+0.138, +0.186]) |
| P1 rolling | 41.98 | 52.02 | +10.04 [+3.10, +16.99], 8/10 | 34% of 29.23 (MUSIQ); 32% (`subject_consistency`) |

On the properly built path the correction recovers about three quarters of the INT2 gap; on the rolling path, about a third. The difference of the two effects is **+10.28 [+6.07, +14.49]**, positive in 10/10 prompts. We write this as a **correction × path** interaction: P1 and P3 differ in storage policy *and* in attention kernel (explicit dequantize→FA2 versus fused), and we have not attributed the difference between them to storage alone (§9.1). What the table establishes is narrower and sufficient: **the benefit measured for a correction on one storage/execution path did not represent its benefit on another.** The P1 figure is the one we reported first; it understated the correction by half.

At K4V2 on P3 the correction adds **+3.59 [+0.85, +6.33]** MUSIQ and **+0.027 [+0.015, +0.039]** `subject_consistency`, 10/10 on both.

### 7.2 What the per-key structure contributes

Replacing the per-key bias with a flattened control produced no detected difference, in MUSIQ or in `subject_consistency` (measured on P1; not repeated on P3). The claim this supports is narrow: the flattened control is constant **per (query, head)** and therefore removes only the per-key structure while preserving the scale differences between heads. **Whether a single global constant would suffice is a different claim and is untested** — we did not run that sweep.

### 7.3 Concentration

Within a head, the dispersion of the per-key bias is small: σ_b/μ_b has median **0.165** against an independent-channel prediction of **0.153**, Across **1,080 head-level observations drawn from 90 captured records over 10 prompts**, the descriptive Pearson correlation between prediction and measurement is **+0.732**. The participation ratio is **d_eff ≈ 11.7** against a nominal 128; it is not a count of channels, and indicates only that the contribution is distributed unevenly across them.

The prediction is nevertheless **systematically low**, and its mean bias and its per-observation scatter are different quantities. **Confidence intervals for the mean residual use prompts as the replication unit** (n = 10): the mean residual (observed − predicted) is **+0.0205 [+0.0134, +0.0276]**, t = +6.52, positive in 10 of 10 prompts. Separately, the **row-level residual standard deviation is 0.0712**, against a predicted median of 0.153 — substantial scatter around individual predictions. We therefore report the concentration as reproduced across prompts and the prediction as biased low, and we make no claim about the accuracy of an individual predicted value.

### 7.4 Correction strength

λ = 1 is the theoretical default. In the direct contrast we ran, λ = 0.5 and λ = 1 were **not distinguished on `subject_consistency`** (+0.0019 [−0.0057, +0.0095]) but **were on MUSIQ**, where λ = 1 is higher (λ0.5 − λ1 = −2.96 [−4.69, −1.22]); **the location of the optimum is undetermined** — we did not sweep finely enough to place it, and the two metrics do not agree even on this pair. Above λ = 1 the disagreement is in **order**: the MUSIQ point estimate is highest at λ = 4, while `subject_consistency` at the same setting is **below the uncorrected baseline** (−0.1602, t = −18.27). A single metric is therefore not sufficient to select λ. We record the disagreement and its size and do not adjudicate between the metrics.

### 7.5 Where the derivation's premise fails

In a **single-prompt diagnostic**, most channelwise terms had small approximation errors (|a| median 0.002–0.107; relative error below 1e−2 at the 95th percentile). **That percentile summary does not characterise the error in the summed correction** — the quantity actually subtracted from the logit is the sum over 128 channels, and a rare large term can dominate a sum whose 95th percentile is small.

Measured on the summed quantity across **all ten prompts**, the error is large in the sampled outlier heads: in layer 29 the relative error reaches a 95th percentile of 12.2–25.7 (median across prompts; full range 8.7–32.0), while layers 0 and 15 stay at 0.02–0.17. The two measurements have different scopes and we do not present them as one verification.

In those heads the correction does not re-weight the cache — it removes it. Two separate observations support this.

**Directly measured, in the approximate-form run**: for layer 29 head 9 the cache attention mass falls from 0.89–0.99 without correction to effectively zero with it, and that head's self-attention mass is 1.0.

**Computed as a counterfactual on fixed logits, for the exact form**: the exact bias for that head is 162–328, which the counterfactual maps to a cache mass below 1e−68. The exact form was not run in generation (§7.6), so this is a calculation on a fixed state, not an observation. The counterfactual expression was first checked against the approximate-form run, where it reproduces the measured mass with a median absolute error of 0.024.

The scope is narrow in both cases: **3 of the 12 heads in layer 29**; the remaining heads and layers 0 and 15 have biases below 1 and barely move.

### 7.6 Implementation status

The shipped series implementation of the exact form diverges beyond its radius of convergence, reaching an absolute difference of 2.3e+27 in layer 29. A numerically stable replacement is accurate (≤ 7.9e−6 over the full range), but in our trial **it did not complete the first of three prompts within 8 hours 23 minutes**; we did not establish a like-for-like throughput comparison, so we quote no speed ratio. **At present we have no exact-form implementation we can run in generation**: the shipped one is wrong in the tail and the stable one did not finish. We therefore report the approximate form as the one under study and treat the exact form as a diagnostic computed on fixed states.

---

## 8 What the Correction Does Not Reach

On the rolling path, the residual loss after correction is +19.18, which splits as +4.36 against a teacher-forced control and +14.82 between that control and free-running generation. On the packed path the post-correction residual is much smaller — **−6.26 [−9.35, −3.17]** MUSIQ and **−0.063 [−0.080, −0.046]** `subject_consistency` at INT2 (§9.2) — and we did not repeat the teacher-forced decomposition there. **This is an arithmetic decomposition of metric differences, not a causal one.** We write the second term as the **additional rollout loss relative to a BF16-history control** and do not call it contamination: the teacher-forced contrast cannot separate the contributions of the history, the query and the current chunk's quantization, nor their interactions.

Allocating the residual among candidate causes would require a presentation consistent with §5.3 — ordered and conditional, or carrying the interaction explicitly — and we do not attempt it here. A separate diagnostic measured only the re-quantization error; on a fixed grid that quantity is identically zero (§4.1), so it does not bear on the error against the original, which requires preserving the original keys for comparison.

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

Two limits on reading this table. The arms are **not a complete factorial**: the cell (original BF16 input, fixed grid) was not run, and P3 lies outside that design, having no re-quantization input and additionally using a different quantization kernel and attention path. Accordingly **P3's +2.65 is attributed to the P3 path as a whole**, not to its storage policy; on two reconstructed cache states derived from one generation run, the quantizers agreed, while the attention outputs were not bitwise identical; their effect on rollout quality remains unmeasured.

On memory the arms separate along a different axis. Relative to P1, only P3 reduces residency further (P2 differs from P1 by 0.999×); relative to BF16 (6.038 GB), P3 is 3.292× and P2 is 1.775× smaller. Quality and memory may be reported together **for the same pair** — P3 against P1 is +2.65 MUSIQ at **54.0% of P1's cache residency** — but figures drawn from different pairs must not be combined: the 5.145× between P4 and P3 does not belong with the +2.65 between P3 and P1.

P4 is larger than BF16 alone because it retains a BF16 master alongside the quantized state (6.038 + 3.397 = 9.436 GB). This is a property of **our implementation of that design inside Self-Forcing**; we did not measure the original system that uses it.

### 9.2 Quality on the packed path against BF16 — same window

The cleanest comparison holds the cache configuration fixed: A1 window (21 frames, no sink) for every arm, packed path with the correction on, against our BF16 baseline (which equals the released BF16 path bit-for-bit under a matched RNG convention, §3). n = 10, 63 frames, one batch.

| A1, P3 + λ1 vs BF16 | MUSIQ (BF16 71.20) | `subject_consistency` (BF16 0.9448) | `temporal_flickering` (BF16 0.9812) | residency |
|---|---|---|---|---|
| **K4V2** | 70.80, **−0.41 [−1.24, +0.43]** | 0.9336, **−0.011 [−0.024, +0.001]** | +0.001 [−0.001, +0.003] | 2.157 GB (2.80× smaller) |
| INT2 | 64.95, **−6.26 [−9.35, −3.17]** | 0.8819, **−0.063 [−0.080, −0.046]** | −0.006 [−0.010, −0.003] | 1.834 GB (3.29× smaller) |

In the conditions tested, **the INT2 residual loss is detected on both metrics; the K4V2 difference is not detected on either.** The K4V2 interval on `subject_consistency` reaches only +0.001 and the point estimate is below BF16 in 8 of 10 prompts, so a loss within [−0.024, +0.001] remains possible; **not detected is not equivalence** — no equivalence test was run.

### 9.3 One final configuration (A4 window)

For the configuration we first reported — A4 window (12 frames + 3-frame sink), packed fused path, RTN K4V2, in-kernel bias at λ = 1 — measured against the A1 BF16 baseline on a fixed source, 10 prompts, 63 frames:

| | BF16 | final | change |
|---|---|---|---|
| cache residency | 6.038 GB | 1.510 GB | **4.00×** |
| peak allocated | 15.33 GB | 10.81 GB | |
| `inference()` total | 465.6 s | 545.2 s | **+17.1%** |
| MUSIQ | 71.204 | 71.381 | +0.177 [−1.046, +1.400] |

Four qualifications belong with this table. **(i)** The MUSIQ difference is **not detected, which is not equivalence** — no equivalence test was run, and the sign of a point estimate carried by 4 of 10 prompts should not be read. **(ii)** The two arms differ in cache configuration (A4 versus A1), so this is not the effect of quantization alone: a window change and a sink are included — the same-window comparison in §9.2 is the one to read for the quantization path itself. **(iii)** MUSIQ is the only metric measured for this arm. **(iv)** The reported time is end-to-end and **includes VAE decoding**. We did not measure the decode cost separately or verify that it is identical across arms, so we do not infer what the attention path's overhead would be with decoding excluded.

The residency decomposes as 0.647 GB of packed segments plus **0.863 GB of BF16 pending chunk — 57% of the total** — because the fused path does not pack the chunk currently being written. The remaining 276,960 B is accounted for: 276,480 B from rounding the scale-block count (292.5 → 293) across three segments and both tensors, and 480 B of index tensors. Note that the two large components are closed-form quantities compared against the measurement, not a decomposition derived from it.

Measured in isolation on a single harness, the fused kernel costs **+26.6%** against FlashAttention-2 (8.325 ms versus 6.575 ms). We report **+17.1% for the measured inference workload and +26.6% separately for the kernel microbenchmark**. The microbenchmark does not decompose or explain the end-to-end figure, and the two are not combined.

---

### 9.4 What the INT2 saving buys

Relative to K4V2 at the same window, INT2 saves **323 MB (15% of the K4V2 cache)** and, in the conditions tested, costs a detected loss on both metrics (§9.2). Whether that trade is worth making depends on what 323 MB enables. From the closed-form residency model (validated to 0.000 MB on three measured points): at the K4V2 W=7 budget INT2 fits **one more chunk** (8 versus 7); larger gains — +4 chunks at the BF16 W=4 budget, +8 at the BF16 W=7 budget — lie **beyond the 7-chunk training window** of this model. On a 24 GB device with ≈ 9.3 GB of non-cache memory per stream, both K4V2 and INT2 admit **two** concurrent streams to BF16's one; the 323 MB does not change the count. We therefore do not claim a use for INT2 over K4V2 in this setting. The regime in which INT2 would matter — a device where the cache is the binding constraint, a model with a longer training window, or a larger batch — was not measured.

## 10 Limitations and Discussion

Our results come from the Wan 1.3B family, 10 to 20 prompts, and RTN-family quantizers, on one implementation. Four diagnostics rest on narrower ground still: the head-level bias statistics of §7.5 come from a single prompt in part, the two re-quantization paths of §4.3 were not separated in their quality contribution, the attribution of §9.1's +2.65 between storage policy and execution path remains open, and the correction × path interaction of §7.1 (+10.28) likewise cannot be attributed to storage alone because P1 and P3 also differ in attention kernel. The comparisons in §9.2 hold the window fixed; we did not compare INT2 and K4V2 at equal memory, and the context (longer window at equal budget) and throughput (concurrent streams) questions remain unanswered — a 126-frame BF16 comparison of 12- versus 21-frame windows (n = 5, sink not controlled) detected no `subject_consistency` difference (+0.024 [−0.019, +0.066]), which bounds the effect size to look for rather than closing the question.

Two storage emulations differ between the codebases we touched — one retains a BF16 master and does not accumulate, the other accumulates — so cross-model comparisons carry that factor.

During this work we produced seven instrumentation defects and four reporting errors; both lists appear in the appendix with what each affected. Only one instrumentation defect changed a result, and it did so by silently preventing a multi-prompt run from generalising. The common cause in nearly every case was the same: a new instrument was used on the quantity of interest before it was run on a case whose answer was already known.

The recommendation we can support is narrow and practical. **Report the storage and re-quantization conditions of a quantized KV cache alongside its bit-width, and where feasible report results under both aligned and unaligned storage.** A bit-width alone does not identify the configuration that was measured.

---

## 11 Conclusion

The quality cost of a low-bit KV cache is not a function of bit-width alone. In the implementation we studied, whether stored content is re-quantized on a moving grid changes the measured effect of a cache-configuration choice — the sign of its point estimate and whether a benefit is detected — and produces an interaction larger than either intervention's individual effect. Storage conditions are part of the configuration being evaluated, and results that do not state them describe an experiment that cannot be identified.

---


### 변경 기록 v1 → v2 (2026-09-21)

- **§7.1** 재구성: P3(packed, 재양자화 없음) 위 보정 **+20.32 [+13.57, +27.07]**(INT2 갭의 76%), P1 위 +10.04(34%). 차 +10.28 [+6.07, +14.49] 은 **보정 × 경로**(저장 정책 + attention 커널) 상호작용으로만 쓴다. K4V2 +3.59.
- **§9.2 신설**: 같은 창(A1)에서 P3+λ1 vs BF16 — K4V2 두 지표 비검출(subj CI 상한 +0.001, 동등 아님), INT2 두 지표 손실 검출. 구 §9.2(A4 최종 구성)는 §9.3 으로.
- **§9.4 신설**: INT2 − K4V2 = 323 MB 가 사는 것 — 분포 안 +1 chunk, 24 GB 에서 스트림 2 vs 2. INT2 의 용도를 주장하지 않는다.
- **§1 기여 3, §7.2, §8, §10** 을 위와 맞춤. §10 에 문맥·처리량 미답과 126f 사전 결과(효과 크기 한정, 종료 아님) 추가.
- 근거: `results/session8/p3bias_musiq_contrasts.txt`, `p3bias_vbench_contrasts.txt`, `p3bias_vs_bf16_and_interaction.txt`, `what_323mb_buys.txt`; 드라이버 `run_s8_P3bias_main.sh`, `run_s8_P1lam1_rerun.sh`.
