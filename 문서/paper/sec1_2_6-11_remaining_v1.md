# §1 · §2 · §6 · §7 · §8 · §9 · §10 · §11 — 초고 v1 (영문)

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
3. We separate the size of the Jensen bias correction's benefit from the contribution of its per-key structure, and report where its derivation's premise fails (§7).
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

**Positioning.** What we contribute is not the observation that storage can drift, but a measurement of **how much a storage condition changes an evaluation's conclusion** in a setting where it was not controlled — including a sign change — together with the protocol needed to keep such conditions visible.

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

### 7.1 Effect size

The Jensen bias correction recovers **+10.04 [+3.09, +16.99]** of the INT2 loss, about a third of the total +29.23. Two independent metrics agree on the fraction recovered: 34% by MUSIQ and 32% by `subject_consistency`.

### 7.2 What the per-key structure contributes

Replacing the per-key bias with a flattened control produced no detected difference, in MUSIQ or in `subject_consistency`. The claim this supports is narrow: the flattened control is constant **per (query, head)** and therefore removes only the per-key structure while preserving the scale differences between heads. **Whether a single global constant would suffice is a different claim and is untested** — we did not run that sweep.

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

After correction, the residual loss is +19.18, which splits as +4.36 against a teacher-forced control and +14.82 between that control and free-running generation. **This is an arithmetic decomposition of metric differences, not a causal one.** We write the second term as the **additional rollout loss relative to a BF16-history control** and do not call it contamination: the teacher-forced contrast cannot separate the contributions of the history, the query and the current chunk's quantization, nor their interactions.

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

### 9.2 One final configuration

For a single configuration — A4 window, packed fused path, RTN K4V2, in-kernel bias at λ = 1 — measured against our BF16 baseline on a fixed source, 10 prompts, 63 frames:

| | BF16 | final | change |
|---|---|---|---|
| cache residency | 6.038 GB | 1.510 GB | **4.00×** |
| peak allocated | 15.33 GB | 10.81 GB | |
| `inference()` total | 465.6 s | 545.2 s | **+17.1%** |
| MUSIQ | 71.204 | 71.381 | +0.177 [−1.046, +1.400] |

Four qualifications belong with this table. **(i)** The MUSIQ difference is **not detected, which is not equivalence** — no equivalence test was run, and the sign of a point estimate carried by 4 of 10 prompts should not be read. **(ii)** The two arms differ in cache configuration (A4 versus A1), so this is not the effect of quantization alone: a window change and a sink are included. **(iii)** MUSIQ is the only metric measured for this arm. **(iv)** The reported time is end-to-end and **includes VAE decoding**. We did not measure the decode cost separately or verify that it is identical across arms, so we do not infer what the attention path's overhead would be with decoding excluded.

The residency decomposes as 0.647 GB of packed segments plus **0.863 GB of BF16 pending chunk — 57% of the total** — because the fused path does not pack the chunk currently being written. The remaining 276,960 B is accounted for: 276,480 B from rounding the scale-block count (292.5 → 293) across three segments and both tensors, and 480 B of index tensors. Note that the two large components are closed-form quantities compared against the measurement, not a decomposition derived from it.

Measured in isolation on a single harness, the fused kernel costs **+26.6%** against FlashAttention-2 (8.325 ms versus 6.575 ms). We report **+17.1% for the measured inference workload and +26.6% separately for the kernel microbenchmark**. The microbenchmark does not decompose or explain the end-to-end figure, and the two are not combined.

---

## 10 Limitations and Discussion

Our results come from the Wan 1.3B family, 10 to 20 prompts, and RTN-family quantizers, on one implementation. Three diagnostics rest on narrower ground still: the head-level bias statistics of §7.5 come from a single prompt in part, the two re-quantization paths of §4.3 were not separated in their quality contribution, and the attribution of §9.1's +2.65 between storage policy and execution path remains open.

Two storage emulations differ between the codebases we touched — one retains a BF16 master and does not accumulate, the other accumulates — so cross-model comparisons carry that factor.

During this work we produced seven instrumentation defects and four reporting errors; both lists appear in the appendix with what each affected. Only one instrumentation defect changed a result, and it did so by silently preventing a multi-prompt run from generalising. The common cause in nearly every case was the same: a new instrument was used on the quantity of interest before it was run on a case whose answer was already known.

The recommendation we can support is narrow and practical. **Report the storage and re-quantization conditions of a quantized KV cache alongside its bit-width, and where feasible report results under both aligned and unaligned storage.** A bit-width alone does not identify the configuration that was measured.

---

## 11 Conclusion

The quality cost of a low-bit KV cache is not a function of bit-width alone. In the implementation we studied, whether stored content is re-quantized on a moving grid changes the measured effect of a cache-configuration choice — the sign of its point estimate and whether a benefit is detected — and produces an interaction larger than either intervention's individual effect. Storage conditions are part of the configuration being evaluated, and results that do not state them describe an experiment that cannot be identified.

---

### 확인 필요 (집필 세션)

1. **§2의 인용** — 각 선행연구를 arXiv 번호로 확정하고, KIVI·QVG 커밋 해시를 실제 조사표와 대조할 것. 본문의 `876b4d2`·`0601468`은 조사표 인용이다 `[간접]`.
2. ~~**§7.1의 34%**~~ — **확인됨.** 10.044/29.229 = 34.4%; subject_consistency 0.0737/0.2284 = **32.3%** (`figure_data.json` fig5, bf16_ref·λ1) `[확인]`.
3. **§9.2의 peak reserved**를 표에서 뺐다(BF16 22.01 / 최종 16.55). 넣을지 결정할 것.
4. ~~**§10의 "일곱·넷"**~~ — **확인됨.** `S6_FINDINGS.md` §5.4 도구 버그 7항, §5.5 보고 오류 4항 `[확인]`.
5. **§1 Figure 1 캡션**에 INT4 패널이 "상호작용 비검출"을 보이기 위한 것이며 INT4 주효과를 독립 결과로 읽지 말라는 문구를 넣을 것.
6. 전체 용어 통일: displaced grid / storage path / cache residency / not detected vs equivalent / re-quantization.

### 검증 기록 (2026-09-21, 결과 세션)

본문 수치를 원자료와 대조했다. 위 본문에서 **고친 것**:
- **§2** — QVG 설명·33-method 문장·Tuncer et al. 호칭·그룹 축 대조를 `CITATIONS_verified_6` §7 대로 교체. **"within our survey the only path exhibiting the condition is the patch we used ourselves" 삭제** — 공개 harness `b4c0936`의 패치가 같은 구조를 가진다(§5.1). 대체 문단은 §5.2 결론 문장을 따랐다.
- **§6** — **+11.20 → +11.18 [+7.59, +14.76]**. +11.20은 `gateK_aligned`의 oracle(55.99) vs driver-oldest(44.79)이고, 분해 성분은 `gateK0` 네 팔(oracle3 55.44 vs k0_oldest 44.26)에서 나왔다. 다른 팔·다른 기준선이므로 "sum to +11.18 against the +11.20 measured"처럼 한 측정의 오차로 쓰면 안 된다.
- **§7.4** — λ0.5 vs λ1 은 **subject_consistency 한정** 비검출이고 **MUSIQ 에서는 λ1 이 높다**(−2.96 [−4.69, −1.22]). "λ≥2에서 순서가 반대"를 구체 문장으로 교체.
- **§7.5** — |a| 중앙 최솟값 0.003 → **0.002**(0.00229, chunk 1 layer 0).

**대조해서 맞은 것**: §1·§4·§5·§7.1·§7.3·§7.5(나머지)·§7.6·§8·§9 의 모든 수치, §10 의 7·4 개수.

### 검증 기록 (2026-09-21, 2차)
- §1·§11: "sink effect's sign / including its sign" → "점추정치의 부호와 이득 검출 여부". 정렬 OFF 의 −1.54 는 CI 가 0 을 포함하므로 악화의 증거가 아니다(§5.2 와 일치시킴).
- §2 머리말: "외부 원문 재검증 안 함" → 확인/미확인 상태표.
- §6: 표가 "순서를 정한 조건부 대비"임을 본문에 명시.
