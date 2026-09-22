# Storage Conditions Are Part of the Configuration: Re-quantization Grids and the Evaluation of Low-Bit KV Caches in Streaming Video Diffusion

*Draft v1 — 2026-09-22. Numbers are sourced from `claim_table.md`; figures are generated from raw data by `tools/make_figures.py`.*

## Abstract

The quality cost of a low-bit KV cache in streaming video diffusion is usually reported as a function of bit-width. We show that, in the implementation we studied, it is also a function of how the cache is stored. A chunk of 4,680 tokens is 292.5 groups of 16, so eviction displaces surviving content by 8 tokens relative to the quantization grid, and re-quantizing on the displaced grid changes the stored codes: in a run-time audit, key codes changed in 100% of displaced surviving-slot comparisons and 0% with the grid pinned. This storage condition changes what an evaluation concludes. In a full 2×2 at INT2, grid alignment and a content sink interact with a term of +9.22 [+5.85, +12.59] MUSIQ, larger than either main effect; the sink's contrast is −1.54 [−3.14, +0.06] under the displaced grid and +7.68 [+4.86, +10.49] under the aligned one. A published bias correction for quantized keys recovers +10.0 of the INT2 loss on the re-quantizing path but +20.3 [+13.6, +27.1] on a packed once-per-chunk path — the same correction, evaluated on two paths, gives benefits that differ by a factor of two. On the packed path with the correction, K4V2 is not distinguished from BF16 on MUSIQ or subject consistency (not equivalence; no equivalence test was run), while INT2 retains a detected loss on both. We do not claim the failure mode is new — one public implementation names and avoids it — nor that INT2 has a use over K4V2 in the setting measured. We report a protocol under which storage conditions stay visible, and recommend that KV-cache quantization results state the storage and re-quantization conditions alongside the bit-width.

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

> **Figure 1.** *Alignment × sink interaction, INT2 (left) and INT4 (right).* MUSIQ differences with 95% CIs (t(9, .975), n = 10, paired by prompt) from a full 2×2 in which all four cells were measured. The third bar is a **computed prediction under additivity** (hatched), not a measurement; the fifth is the interaction estimate itself. At INT2 the predicted sum is +0.74 [−2.72, +4.20] against a measured joint effect of +9.96 [+6.54, +13.37]; the interaction is **+9.22 [+5.85, +12.59]**. The purpose of the INT4 panel is the interaction comparison: −0.50 [−1.12, +0.13] is not detected in this sample, which is not a proof of additivity. Bars whose CI includes 0 are drawn in outline only; the "sink only" bar at INT2 (−1.54 [−3.14, +0.06]) is one of them and must not be read as a degradation.  
*(file: `figs/fig1_interaction.pdf`)*

## 2 Background and Related Work

> *Citation status.* Verified against the source paper and repository: QVG (arXiv 2602.02958, commit `0601468`), the UCSD 33-method study (arXiv 2603.27469, harness `b4c0936`), Tuncer et al. (arXiv 2605.26266; no public implementation located), KIVI (arXiv 2402.02750, commit `876b4d2`). Not yet verified against the source — statements about them are provisional: Self-Forcing, LongLive, Deep Forcing (2512.05081), KV-AdaQuant (2502.15075), SkyReels-V2, MAGI-1, VBench, MUSIQ, FlashAttention-2.

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

## 3 Measurement Protocol

Everything below was fixed before the comparisons in §§4–9 were run, except where a section says otherwise. Each item names what was verified and how far the verification reaches.

**Models, prompts, hardware.** Self-Forcing (Wan 1.3B family) with a rolling KV cache; LongLive (retrained for a short cache) for the cross-model checks of §7. Ten prompts (`prompts/quant3/prompts10.txt`), 63 output frames (21 chunks of 3 frames = 4,680 tokens each), seed 0, one RTX 4090 (24 GB). Cache configurations are named A1 (21-frame window, no sink), A3 (12-frame window, no sink), A4 (12-frame window, 3-frame sink); a chunk is one 3-frame block.

**Quantizer.** Round-to-nearest, symmetric, per-(block, head, channel) scale, groups of 16 tokens along the token axis (`kv_quant/rtn.py`, `block_size = 16`). Bit-widths INT4, INT2 and K4V2 (4-bit keys, 2-bit values). "Block" in this paper means the 16-token quantization group, not the 3-frame generation block.

**Storage and execution paths.** P1: dequantize the whole cache, shift on eviction, write the new chunk, re-quantize the whole cache (the path of the public Self-Forcing patch we measured). P2: P1 with each chunk slot quantized on its own grid. P3: packed segments, each chunk quantized once when it is finalized, read by a fused Triton attention kernel; the chunk being written stays in BF16. P4: P1 with a retained BF16 master as the re-quantization input. Paths are compared only when the section states which components differ (§9.1).

**Randomness.** Our BF16 path reproduces the released implementation bit-for-bit on 10 / 10 prompts once two conventions are matched: a single global noise draw and no per-chunk seed. Before matching them we observed a MUSIQ difference of −1.76 [−3.68, +0.16] that was entirely attributable to the RNG convention. All BF16 baselines here are therefore the released model's.

**Memory.** Residency is the byte count of the cache tensors at the end of the last prompt, deduplicated by `untyped_storage().data_ptr()` — keying on `data_ptr()` double-counts offset views and misses narrowed ones (verified on a constructed counter-example: 2,250 B versus 5,610 B). Residency is not peak memory; peak allocated is reported separately. A closed-form residency model (§9.4) matches three measured configurations to within 0.000 MB.

**Metrics.** MUSIQ (no-reference, per-frame image quality) as the primary metric; VBench `subject_consistency` (DINO feature similarity across frames) and `temporal_flickering` as independent axes. VBench `imaging_quality` is MUSIQ-based and is not treated as independent. Where both MUSIQ and `subject_consistency` are available they are reported together, and §7.4 records a case in which they disagree.

**Statistics.** All contrasts are paired by prompt; intervals are 95% with t(n − 1, .975) (2.262 at n = 10), not 1.96. A non-detected difference is reported as such and is never called equivalence; no equivalence (TOST) test was run in this work. Where head-level or row-level observations are pooled across prompts, the prompt is the unit of replication (§7.3). The sign of a point estimate whose interval includes zero is not interpreted.

**Provenance.** Every performance and quality artefact records the commit hash and the SHA-256 of the working-tree diff; a commit hash alone does not identify the source of a run with uncommitted changes. Verdicts are per artefact, not per batch: a driver cannot mark a batch complete while a step failed.

**Instruments.** Three instruments produce the storage measurements of §4: a storage-only probe that applies the quantizer to a clean BF16 chunk (§4.1); a run-time counter that follows content across evictions and compares stored key codes between successive writes (§4.3); and a capture hook that saves the BF16 cache and query immediately before quantization, keyed by prompt, seed, chunk, layer, step and call site (§9.1). Each was first run on a case with a known answer. Where that rule was not followed the instrument was wrong (Appendix A).

**Five quantities.** Throughout we keep separate: whether a re-quantization call occurs, whether stored codes change, whether the dequantized value changes, whether the error against the original changes, and whether quality changes (§4.2).

## 4 Regridding in Rolling Caches

### 4.1 Where the grid moves

The cache is quantized in groups of 16 tokens along the token axis, each group carrying its own scale. A chunk is 4,680 tokens, which is 292.5 groups. Eviction therefore does not remove a whole number of groups: surviving content is displaced by 8 tokens relative to the group boundaries, and every group re-forms with different members.

Round-to-nearest is **idempotent on a fixed grid**: re-quantizing already-quantized content reproduces it bit for bit (relative difference 0.000e+00). Once the grid is displaced by 8 tokens, it is not: the relative difference is 5.47e−2 at INT4 and 2.01e−1 at INT2. These are storage-only figures — a clean BF16 chunk put through the storage operations alone, with no generation feedback — and they are measured by a different instrument from the run-time counter in §4.3.

> **Figure 2.** *Idempotence and grid displacement.* *(file: `figs/fig3_idempotence.pdf`)* Relative L2 change of the reconstruction across re-quantization (storage operations alone on a clean BF16 chunk; `gateK1_lineage.py`). On the same grid the change is 0 for INT4 and INT2 — re-quantization did not alter the stored content; this is idempotence, not "bit-exact against the original". After an 8-token displacement (4,680 = 292.5 × 16) it is 0.055 (INT4) and 0.201 (INT2). The observation is specific to the RTN implementation, dtype and grid configuration we verified.

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

> **Figure 3.** *K-code changes across surviving-slot comparisons *(file: `figs/fig2_requant_counts.pdf`)*, A1 INT4 generation run.* Unit: one comparison of a surviving slot's stored **key** codes between successive writes (not calls, not tokens); 3 prompts × 63 frames × 30 layers. Rates use separate denominators: with alignment off, codes changed in 1,350 / 39,690 undisplaced comparisons (3.40%) and in **7,560 / 7,560 displaced ones (100%)**; with alignment on, 0 / 39,690 and 0 / 7,560. These are counts of code changes, not quality contributions. The A2 INT2 sink-slot audit (804 / 9,000) is a different run at a different bit-width and is not plotted on this axis.

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

Figure 1 (§1) shows the four cells, the additive prediction and the interaction estimate side by side for both bit-widths.

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

> **Figure 4.** *Anchor effect as ordered conditional contrasts.* *(file: `figs/fig4_anchor_decomposition.pdf`)* Applied top to bottom: anchor presence (one chunk pinned) +7.68 [+4.86, +10.49]; anchor budget 1 → 3 chunks +3.50 [+1.51, +5.50]; adaptive selection **+0.00** — the oracle and the fixed policy made identical selections in all 140 decisions observed, so no interval is drawn (it is not an estimate with uncertainty). The cumulative total, oracle − oldest = +11.18 [+7.59, +14.76], is recomputed from the per-prompt total and is **not** the sum of the increment CIs; it is drawn with a different marker. Because total capacity is fixed, a larger anchor budget shortens the recent context: this is an allocation effect. +0.00 is specific to the 140 decisions observed and does not show adaptive selection to be unnecessary on other inputs.

## 7 The Correction and Its Scope

### 7.1 Effect size — and its dependence on the path

We evaluate the correction (Tuncer et al., λ = 1) on two paths that store the same content differently: **P3**, the packed once-per-chunk path (fused kernel, no re-quantization), and **P1**, the rolling path that re-quantizes on a displaced grid. The execution paths differ in more than storage. On P3 the correction is a switch inside the same fused kernel (λ = 0 or 1). On P1, λ = 0 dequantizes and runs the default attention, whereas λ = 1 dequantizes and runs an **explicit corrected-attention path** — FlashAttention-2 exposes no per-key bias interface, so turning the correction on also changes the kernel. Same model, prompts (n = 10), 63 frames, seed, A1 window; the four P3 arms ran in one batch, the P1 arms in another.

| INT2, A1 | λ = 0 | λ = 1 | λ1 − λ0 (95% CI) | fraction of BF16 gap recovered |
|---|---|---|---|---|
| **P3 packed** | 44.63 | **64.95** | **+20.32 [+13.57, +27.07]**, 10/10 | 76% of 26.58 (MUSIQ); 72% of 0.225 (`subject_consistency`: +0.162 [+0.138, +0.186]) |
| P1 rolling | 41.98 | 52.02 | +10.04 [+3.10, +16.99], 8/10 | 34% of 29.23 (MUSIQ); 32% (`subject_consistency`) |

On the properly built path the correction recovers about three quarters of the INT2 gap; on the rolling path, about a third. The difference of the two effects is **+10.28 [+6.07, +14.49]**, positive in 10/10 prompts. We write this as a **correction × path** interaction: P1 and P3 differ in storage policy *and* in execution path — and on P1 the correction itself switches the kernel — so we have not attributed the difference between them to storage alone (§9.1). What the table establishes is narrower and sufficient: **the benefit measured for a correction on one storage/execution path did not represent its benefit on another.** The point estimate of the correction's gain observed on P1 was about half of that observed on P3; the P1 value is the one we reported first, and it is a correct measurement of P1.

At K4V2 on P3 the correction adds **+3.59 [+0.85, +6.33]** MUSIQ and **+0.027 [+0.015, +0.039]** `subject_consistency`, 10/10 on both.

### 7.2 What the per-key structure contributes

Replacing the per-key bias with a flattened control produced no detected difference, in MUSIQ or in `subject_consistency` (measured on P1; not repeated on P3). The claim this supports is narrow: the flattened control is constant **per (query, head)** and therefore removes only the per-key structure while preserving the scale differences between heads. **Whether a single global constant would suffice is a different claim and is untested** — we did not run that sweep.

### 7.3 Concentration

The diagnostics in §§7.3–7.6 were captured on the P1 path with the explicit corrected-attention kernel (λ = 1, Taylor form) unless stated otherwise; none has been repeated on P3.

Within a head, the dispersion of the per-key bias is small: σ_b/μ_b has median **0.165** against an independent-channel prediction of **0.153**, Across **1,080 head-level observations drawn from 90 captured records over 10 prompts**, the descriptive Pearson correlation between prediction and measurement is **+0.732**. The participation ratio is **d_eff ≈ 11.7** against a nominal 128; it is not a count of channels, and indicates only that the contribution is distributed unevenly across them.

The prediction is nevertheless **systematically low**, and its mean bias and its per-observation scatter are different quantities. **Confidence intervals for the mean residual use prompts as the replication unit** (n = 10): the mean residual (observed − predicted) is **+0.0205 [+0.0134, +0.0276]**, t = +6.52, positive in 10 of 10 prompts. Separately, the **row-level residual standard deviation is 0.0712**, against a predicted median of 0.153 — substantial scatter around individual predictions. We therefore report the concentration as reproduced across prompts and the prediction as biased low, and we make no claim about the accuracy of an individual predicted value.

### 7.4 Correction strength

The λ sweep below was run on P1 (explicit corrected attention). λ = 1 is the theoretical default. In the direct contrast we ran, λ = 0.5 and λ = 1 were **not distinguished on `subject_consistency`** (+0.0019 [−0.0057, +0.0095]) but **were on MUSIQ**, where λ = 1 is higher (λ0.5 − λ1 = −2.96 [−4.69, −1.22]); **the location of the optimum is undetermined** — we did not sweep finely enough to place it, and the two metrics do not agree even on this pair. Above λ = 1 the disagreement is in **order**: the MUSIQ point estimate is highest at λ = 4, while `subject_consistency` at the same setting is **below the uncorrected baseline** (−0.1602, t = −18.27). A single metric is therefore not sufficient to select λ. We record the disagreement and its size and do not adjudicate between the metrics.

> **Figure 5.** *λ sweep on two metrics.* *(file: `figs/fig5_lambda_two_metrics.pdf`)* Top: MUSIQ; bottom: VBench `subject_consistency`; both as differences from uncorrected INT2 (λ = 0), n = 10, same videos (MUSIQ re-evaluated on all eight arms in one pass). The MUSIQ point estimate is highest at λ = 4 (+12.12) while `subject_consistency` at the same setting is below the uncorrected baseline (−0.160 [−0.180, −0.140]). From λ = 1 to λ = 2 both point estimates fall, so the disagreement is not a uniform reversal above λ ≥ 2. λ = 0.5 versus λ = 1: no difference detected on `subject_consistency` (+0.0019 [−0.0057, +0.0095]); on MUSIQ λ = 1 is higher (λ0.5 − λ1 = −2.96 [−4.69, −1.22]). We do not adjudicate between the metrics. The flattened control is not a point on the λ axis and is shown in a separate panel sharing the y-axis.

### 7.5 Where the derivation's premise fails

In a **single-prompt diagnostic**, most channelwise terms had small approximation errors (|a| median 0.002–0.107; relative error below 1e−2 at the 95th percentile). **That percentile summary does not characterise the error in the summed correction** — the quantity actually subtracted from the logit is the sum over 128 channels, and a rare large term can dominate a sum whose 95th percentile is small.

Measured on the summed quantity across **all ten prompts**, the error is large in the sampled outlier heads: in layer 29 the relative error reaches a 95th percentile of 12.2–25.7 (median across prompts; full range 8.7–32.0), while layers 0 and 15 stay at 0.02–0.17. The two measurements have different scopes and we do not present them as one verification.

In those heads the correction does not re-weight the cache — it removes it. Two separate observations support this.

**Directly measured, in the approximate-form run**: for layer 29 head 9 the cache attention mass falls from 0.89–0.99 without correction to effectively zero with it, and that head's self-attention mass is 1.0.

**Computed as a counterfactual on fixed logits, for the exact form**: the exact bias for that head is 162–328, which the counterfactual maps to a cache mass below 1e−68. The exact form was not run in generation (§7.6), so this is a calculation on a fixed state, not an observation. The counterfactual expression was first checked against the approximate-form run, where it reproduces the measured mass with a median absolute error of 0.024.

The scope is narrow in both cases: **3 of the 12 heads in layer 29**; the remaining heads and layers 0 and 15 have biases below 1 and barely move.

### 7.6 Implementation status

The shipped series implementation of the exact form diverges beyond its radius of convergence, reaching an absolute difference of 2.3e+27 in layer 29. A numerically stable replacement is accurate (≤ 7.9e−6 over the full range), but in our trial **it did not complete the first of three prompts within 8 hours 23 minutes**; we did not establish a like-for-like throughput comparison, so we quote no speed ratio. **At present we have no exact-form implementation we can run in generation**: the shipped one is wrong in the tail and the stable one did not finish. We therefore report the approximate form as the one under study and treat the exact form as a diagnostic computed on fixed states.

## 8 What the Correction Does Not Reach

On the rolling path, the residual loss after correction is +19.18, which splits as +4.36 against a teacher-forced control and +14.82 between that control and free-running generation. On the packed path the post-correction residual is much smaller — **−6.26 [−9.35, −3.17]** MUSIQ and **−0.063 [−0.080, −0.046]** `subject_consistency` at INT2 (§9.2) — and we did not repeat the teacher-forced decomposition there. **This is an arithmetic decomposition of metric differences, not a causal one.** We write the second term as the **additional rollout loss relative to a BF16-history control** and do not call it contamination: the teacher-forced contrast cannot separate the contributions of the history, the query and the current chunk's quantization, nor their interactions.

Allocating the residual among candidate causes would require a presentation consistent with §5.3 — ordered and conditional, or carrying the interaction explicitly — and we do not attempt it here. A separate diagnostic measured only the re-quantization error; on a fixed grid that quantity is identically zero (§4.1), so it does not bear on the error against the original, which requires preserving the original keys for comparison.

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

The cleanest comparison holds the cache configuration fixed: A1 window (21 frames, no sink) for every arm, packed path with the correction on, against our BF16 baseline (which equals the released BF16 path bit-for-bit under a matched RNG convention, §3). n = 10, 63 frames. The four P3 arms ran in one batch; the BF16 arm is the existing A1 BF16 run from the baseline check (same source, prompts, seed and frame count), not part of that batch.

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

> **Figure 6.** *Cost of the A4 final configuration.* *(file: `figs/fig6_cost.pdf`)* Left: resident KV cache, BF16 6.038 GB versus 1.510 GB (4.00× smaller), of which 0.863 GB (57%) is the BF16 pending chunk that the fused path does not pack (hatched). Right: `inference()` total, 465.6 s versus 545.2 s (+17.1%), **including VAE decoding**. Residency is measured at the end of the last prompt with storage-keyed deduplication and is not a peak; peak allocated is 15.33 versus 10.81 GB. The arms differ in cache configuration (A1 window versus A4 window + 3-frame sink), so this is not the effect of quantization alone; the same-window comparison is Table §9.2. The 276,960 B decomposition residual is not drawn. The kernel-only +26.6% measures something else and is reported separately.

---

### 9.4 What the INT2 saving buys

Relative to K4V2 at the same window, INT2 saves **323 MB (15% of the K4V2 cache)** and, in the conditions tested, costs a detected loss on both metrics (§9.2). Whether that trade is worth making depends on what 323 MB enables. From the closed-form residency model (validated to 0.000 MB on three measured points): at the K4V2 W=7 budget, INT2 would fit an eighth chunk — but the model's training window is 7 chunks, so **within the training window both configurations reach the same W=7 and INT2 leaves 323 MB of headroom rather than buying context**; the larger gains (+4 chunks at the BF16 W=4 budget, +8 at the BF16 W=7 budget) all lie beyond the training window. In a simple calculation that assumes ≈ 9.3 GB of non-cache memory per stream (estimated from one BF16 run and applied to both paths), the two configurations admit the same number of concurrent streams on a 24 GB device; **whether they actually run concurrently, and at what throughput, was not measured**, so this calculation does not establish that the 323 MB leaves the stream count unchanged. We therefore do not claim a use for INT2 over K4V2 in this setting, and we do not rule one out. The regime in which INT2 would matter — a device where the cache is the binding constraint, a model with a longer training window, or a larger batch — was not measured.

## 10 Limitations and Discussion

Our results come from the Wan 1.3B family, 10 to 20 prompts, and RTN-family quantizers, on one implementation. Four diagnostics rest on narrower ground still: the head-level bias statistics of §7.5 come from a single prompt in part, the two re-quantization paths of §4.3 were not separated in their quality contribution, the attribution of §9.1's +2.65 between storage policy and execution path remains open, and the correction × path interaction of §7.1 (+10.28) likewise cannot be attributed to storage alone because P1 and P3 also differ in attention kernel. The comparisons in §9.2 hold the window fixed; we did not compare INT2 and K4V2 at equal memory, and the context (longer window at equal budget) and throughput (concurrent streams) questions remain unanswered — a 126-frame BF16 comparison of 12- versus 21-frame windows (n = 5, sink not controlled) detected no `subject_consistency` difference (+0.024 [−0.019, +0.066]); that interval is for the contrast of those two configurations and does not isolate the window-length effect, so it neither bounds it nor closes the question.

Two storage emulations differ between the codebases we touched — one retains a BF16 master and does not accumulate, the other accumulates — so cross-model comparisons carry that factor.

During this work we produced seven instrumentation defects and four reporting errors; both lists appear in the appendix with what each affected. Only one instrumentation defect changed a result, and it did so by silently preventing a multi-prompt run from generalising. The common cause in nearly every case was the same: a new instrument was used on the quantity of interest before it was run on a case whose answer was already known.

The recommendation we can support is narrow and practical. **Report the storage and re-quantization conditions of a quantized KV cache alongside its bit-width, and where feasible report results under both aligned and unaligned storage.** A bit-width alone does not identify the configuration that was measured.

## 11 Conclusion

The quality cost of a low-bit KV cache is not a function of bit-width alone. In the implementation we studied, whether stored content is re-quantized on a moving grid changes the measured effect of a cache-configuration choice — the sign of its point estimate and whether a benefit is detected — and produces an interaction larger than either intervention's individual effect. Storage conditions are part of the configuration being evaluated, and results that do not state them describe an experiment that cannot be identified.

## Appendix A — Instrumentation defects and reporting errors

Seven instrumentation defects and four reporting errors were found during this work. One defect changed a result; the rest were caught before they did. Their common cause was that a new instrument was applied to the quantity of interest before being run on a case whose answer was already known.

**Instrumentation.** (1) The re-quantization counter compared slots by index, so content moved by eviction was counted as re-quantized; fixed by following content. (2) A CSV written with an f-string shifted every column right of any label containing a comma; caught when tables were generated from the CSV. (3) A duplicate argparse registration killed four of five rows in a batch that still printed "done". (4) Memory was deduplicated by `data_ptr()`, double-counting offset views and missing narrowed ones; replaced by storage-keyed accounting with a counter-example test. (5) A wait loop matched its own launching shell and waited indefinitely; replaced by marker files. (6) A running bash script was edited in place; bash re-read it by byte offset and two GPU jobs overlapped. (7) The bias-dump deduplication key omitted the prompt, so a 10-prompt run wrote the same 9 records as a 1-prompt run — **this one affected a result**: the multi-prompt generalisation of §7.3 had not happened until it was re-run.

**Reporting.** (1) A batch with four of five rows failed was reported as "one row failed". (2) A cross-model target was computed from n = 3 data when n = 10 data of the same tool existed (21.68 → 14.67), and a "retraining-level" claim did not hold. (3) The effective scalar of the correction was reported as a uniform mean (70.08) dominated by one head; the attention-weighted value is ~0.6. (4) "The optimum λ may lie below 1" was inferred from two point estimates without a test; the direct contrast was not detected on the metric used.

