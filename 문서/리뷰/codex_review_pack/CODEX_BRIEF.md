# Independent Review Brief — Low-Bit KV Cache in Streaming Video Diffusion

> ⚠ **이 문서는 세션 5 시점 스냅샷이다.** 세션 6·7에서 아래 주장들이 철회·축소됐다:
> C-I(실측 ε 균일·ε=0) **철회** · C-D(구조는 INT4에서만) **철회** · C-R(오라클 정책) **분해** ·
> C-M(0.75GB 8×) **철회, 실측 4.00×** · C-J/C-B/C-H/C-L **범위 축소** · C-A **n=3 예비**.
> **먼저 `CORRECTIONS_2026-09-19.md`를 읽고 해당 항목을 대체해서 읽을 것.**


(Prepared 2026-09-18 for an independent reviewer/auditor. Written so that a reader with no prior context can evaluate the work. **Do not assume any claim below is true; your job is to find where it breaks.**)

---

## 0. Your task

You are Reviewer 2 and an independent code auditor at the same time. We want three things, in this order:

1. **Adversarial review of the claims** (§3–§6 below). For every claim, ask: is there an alternative explanation? is the evidence sufficient (n, metric, model)? is the statistic right? is it actually new relative to the prior work listed in §7? Produce a ranked list of the weakest points, each with the concrete experiment or check that would resolve it.
2. **Code audit of the attached patches** (`patches/`, especially anything touching `causal_model.py`: attention path, correction term, cache eviction, grid alignment; and `kernels/`). Look for: side effects that change generation when a "measurement" flag is on; cross-prompt state leakage; metric computations that reward the wrong thing (we already found one: a drop-from-chunk-0 metric that rewards destroying chunk 0); off-by-one at block/segment boundaries; anything that would make a "bit-exact" gate pass vacuously.
3. **Check the derivation and statistics**: the concentration argument in §5.3, and independent recomputation of paired t-tests / leave-one-out from the CSVs in `results/`.

Report format: numbered findings, each with severity (blocks a claim / weakens a claim / cosmetic), the claim it affects, evidence, and the cheapest way to verify or fix. Do not soften. Where you agree with a claim, say so in one line and move on.

---

## 1. Setting

- Model: **Self Forcing** (Wan2.1-T2V-1.3B distilled to a causal, chunk-wise autoregressive video diffusion model). Generates 3 latent frames per chunk in 4 denoising steps, attending to a rolling KV cache of previous chunks. Default cache: 21 latent frames, no sink ("A1", written w21s0). One chunk = 4,680 spatial tokens; 30 layers; head dim 128.
- Second model: **LongLive v1.0** (same base, fine-tuned for a cache of 3 sink frames + 9-frame window, w9s3).
- Hardware: single RTX 4090 24 GB (local server). Kernel development also on RTX 3080.
- Quantization: KV cache only. UCSD `kv-quant-longhorizon` codebase: RTN symmetric, group 16 along tokens, per-token; KIVI asymmetric (key per-channel, value per-token). Notation KaVb = keys a bits, values b bits.
- Prompts: first 10 of MovieGenBench, 63 latent frames (21 chunks) unless noted. Seeds fixed per chunk (two RNG streams).
- **Primary metric (since session 5): mean MUSIQ over the 21 chunks** (no-reference single-frame quality, ~70 for BF16). The earlier "drop" metric MUSIQ(c20) − MUSIQ(c0) is reported only alongside; see bug E. Also: RAFT flow error, LPIPS vs teacher-forced trajectory, CLIP, latent relative L2. All comparisons: paired t-test over 10 prompts (t(9,.05)=2.26), leave-one-prompt-out for every significant claim.

Cache-structure names: A1 = w21s0 (default), A2 = w21s3, A3 = w12s0, A4 = w12s3, A5 = w21s6.

---

## 2. Measurement tools

- **Chaos floor.** Perturbing the initial noise by 1e-3 makes the BF16 run diverge to relative latent L2 ≈ 0.74 within 8 chunks (three seeds: 0.736–0.760). So reference-based latent distance saturates and cannot measure quality; the perturbed control's MUSIQ is +0.16 vs unperturbed (10 prompts).
- **Teacher-forced / free-running (TF/FR) lockstep.** Two caches in memory. Before chunk t the *reference* cache holds the BF16 trajectory's K/V; the TF pass copies it, quantizes once, generates chunk t → error is pure *injection*. The FR pass generates from its own quantized history → injection + *propagation*. FR − TF = propagation. Gates: with quantization off, TF equals reference bit-exactly (63/63 chunks); reference equals standalone BF16 run (21/21).
- **Attention-mass hook.** FlashAttention-2 does not expose probabilities, so a hook recomputes online softmax from post-RoPE Q and cached K, accumulating mass per source chunk. Verified vs full softmax (3e-8), vs FA2 output (allclose 1e-2), vs an independent CPU fp32 reimplementation from raw dumps (18/18, max 3.9e-6); hook on/off leaves generation bit-exact. The current chunk ("self") is read in BF16 in this codebase (re-quantization happens after attention).

---

## 3. Claims about *where* the loss comes from (sessions 1–5, audited)

C-A. ⚠[n=3 예비·낙폭지표] **Propagation dominates injection.** A1, drop metric: INT4 TF −2.6 vs FR −11.0 → propagation ≈ 5:1; INT2 TF −14.6 vs FR −42.8 → ≈ 2:1. Injection is constant across chunks (RTN re-quantization on the same grid is idempotent, verified bit-exact).
C-B. ⚠[key "단독"→병목, 3비트 하한은 RTN 한정] **Keys alone cause the collapse.** Mean-MUSIQ loss vs BF16, A1, no correction: K2V2 −29.2, K2V3 −29.6, K2V4 −29.6 (value bits irrelevant); K2V2 −29.2 → K3V2 −16.0 → K4V2 −7.2 (key bits monotone); K4V4 −4.6. Explanation offered: softmax convexity — zero-mean key noise inflates E[exp] of cached (quantized) tokens relative to the unquantized current chunk.
C-C. **Attention inverts under INT2.** BF16 at chunk 20: self 47.8%, most recent 19%, oldest slot 7% (chunk 0 gets 8.5% < its 14.3% token share → no implicit attention sink). INT2: self 19.2%, oldest 29.8%. Inversion is already −16 pt on self at chunk 1, before any eviction (eviction starts at chunk 7), and grows −22 pt more by chunk 20. K4V2 keeps self at 47.1%; K2V4 collapses to 21.3%.
C-D. ⚠[철회 — 정렬하면 INT2에서 sink +7.68] **Cache structure helps at INT4 only.** Drop metric, INT4: A1 −10.97, A2 −5.62, A3 −6.04, A4 −3.28 (t=5.4), A5 −5.54; LongLive INT4 +0.11 (n.s. vs its BF16). Mean metric (session 5): A4 vs A1 at INT4 +2.79 (t=3.26) but at **INT2 −2.81 (t=−2.95, harmful)**. At INT2 (drop metric) sinks did nothing (A2, A5 n.s.), shorter window helped (A3 −46.8 vs A1 −52.6).
C-E. **Sinks are content anchors, not attention magnets.** Per-slot deviation from the BF16 trajectory *decreases* with slot age (A1 oldest slot 0.19@c8 → 0.55@c20; pinned sink in A2 stays 0.064). Replacing sink content: another prompt's chunk 0 or INT2-noised → n.s.; **matched-norm Gaussian noise → A2 −5.62 → −17.99, worse than no sink (A1 −10.97)**. Noise anchor also hurts BF16 (−3.67, t=−2.51). A "freeze" variant was excluded after the audit traced it to a cross-prompt tensor leak (bug C).
C-F. **Long horizon.** 126 frames, 5 prompts: BF16 flat (68.6 → 67.9); A1 INT4 falls linearly ≈ −0.6/chunk, −24 at chunk 41, no plateau; A4 INT4 falls at less than half the rate (−12.8 pt better at chunk 41, 5/5).
C-G. **KIVI (after fixing two bugs in the UCSD code: zero-point clamp, fp16 EPS underflow) beats RTN at equal bits**: KIVI4 vs RTN4 +2.60 (10/10), KIVI2 vs RTN2 +13.32 (10/10). KIVI's residual window ≈ no effect here because self is already BF16.

---

## 4. Claims about the *correction* (session 5, after bug E)

Background: TUM (arXiv 2605.26266) subtracts from each cached logit b_i = (1/24d)Σ_c q_c²Δ_ic² (2nd-order Taylor of the exact uniform-noise bias Σ_c log(sinh x_c/x_c), x_c = q_cΔ_ic/(2√d)).

**Bug E (found by session 5, overturned all earlier correction numbers):** our implementation also subtracted the bias on the BF16 *self* block (Δ should be 0 there), destroying chunk 0 in proportion to λ (70.6 → 61.1 @λ=1 → 45.6 @λ=4); and the drop metric c20−c0 mechanically improves when c0 is lowered. Earlier "recoveries" (+10.4 paper-spec, +28.3 at 4× scale, "4× significantly better") were artifacts. Confirmed artifact is confined to correction runs: chunk-0 MUSIQ is 70.3–70.9 across all non-correction configs.

After the fix (self excluded, mean metric), A1 INT2, 10 prompts:
C-H. ⚠[λ*=1은 "이론 기본값"] λ sweep (FR, vs no correction): λ=0.5 +7.09, **λ=1 +10.04 (t=3.27, 8/10)**, λ=2 +9.33, λ=4 +12.12, λ=8 +11.16; λ4−λ1 = +2.08 (t=0.69, n.s.); λ8−λ4 = −0.96 (t=−4.04). **TF**: λ=1 +4.69 (t=2.82), λ=4 +0.22; TF λ4−λ1 = −4.46 (t=−3.02). Conclusion drawn: λ*=1 in both; propagation roughly doubles the effective key noise (λ≈2 restores the attention distribution but quality is flat over λ=1–4).
C-I. ⚠⚠[철회 — 항등적으로 0인 양을 측정] **Measured quantization error is uniform**: σ²_emp/(Δ²/12) = 0.94–0.98 across INT4/INT3/INT2; |mean|/σ ≤ 0.003. Non-self slots have ε exactly 0 (storage is idempotent). An "empirical" correction built from measured ε at cache slots therefore does nothing (−1.0, −1.1) — the session's stated reason is that the realized error at cached slots is zero because what is stored is what is read.
C-J. ⚠[per-key 한정, head 간 스케일은 유지] **Per-key structure contributes nothing.** "Flatten" control: keep the *mean* down-weight of the cache region, remove per-key differences → +10.49 (t=3.37, 9/10) vs Taylor +10.04; difference +0.45 (t=0.62, n.s.). Conclusion drawn: the whole effect is a scalar down-weight of cached logits relative to self; TUM's q²Δ² structure is unnecessary in this setting. (Proposed explanation to be derived: b_i is a sum over 128 channels and concentrates, σ_b/μ_b ~ CV/√d.)
C-K. Correction restores the attention profile toward BF16: self mass at chunk 20: INT2 16.3% → λ=1 37.1% → λ=2 51.1% (BF16 50.1%) → λ=4 73.7% (over-corrected). Quality insensitive between λ=1 and 4.
C-L. ⚠[인과 주장 철회 — 산술 분해] Recovery is 34% of INT2 damage (−29.2 → −19.2). The remaining 66% is hypothesized to be *content* contamination of cached keys (propagation), which a logit offset cannot fix. Not yet decomposed (planned: TF/FR split after correction).

---

## 5. Claims about the *recipe* and *systems* (session 5)

C-M. ⚠⚠[철회 — K3 패킹 없음, 실측 4.00×] **Recipe.** 30-layer KV, BF16 6.04 GB (21 f). A1 INT4 packed 1.89 GB, −4.61. **A4 + λ=1 + K4V2: 0.86 GB (7.0×), −0.55. A4 + λ=1 + K3V2 or KIVI-K2/V2: 0.75 GB (8.0×), −1.11 / −0.72.** A4 + λ=1 + K2V2: −17.63 (key 3-bit is the floor). Increments all significant (e.g. A4+corr K4V2 vs A1 INT4: +4.07, t=3.83, 10/10). Note: "8×" = per-token 4.6× (1344 vs 6144 B/token) × window 21→12 (1.75×); the window part reduces context and is not compression per se.
C-N. **Effective bits**: group-16 fp16 scales add +1 bit/element (INT2 → 3.0 effective bits; TUM g=32 → 2.75; QVG VQ → ~2.3).
C-O. ⚠[표현 교정 — 재구성 비용] **Systems.** Attention arithmetic intensity = number of queries = 4,680 FLOP/byte (cache-length independent), 28× the 4090 balance point → compute-bound; KV compression buys capacity, not attention speed. Existing INT4 path: 3 chunks BF16 2,817 ms → INT4 4,427 ms; attention unchanged (906→900); of +1,611 ms, quantize 1,261 + dequantize 343; eviction copy_/cat ≈ 554 ms. Existing code stores 4-bit codes in int8 (3.40 GB, only 1.8×).
C-P. **Kernels.** Packed quantize kernel bit-exact with UCSD path, 2.81 → 0.31 ms. Fused attention v2 (Triton) reads packed K/V, dequantizes in registers, optional in-kernel bias, passes equivalence gates 1–3 (max|diff| ≤ 9.8e-4); **8.17 ms vs FA2 6.9 ms (+13%, target ≤5% not met)**; end-to-end 70.1 s → 51.3 s per prompt (−27% vs existing INT4 path), peak 10.69 → 9.84 GB. Gate 4 (quality equivalence ±1 vs dequant path) "failed" because fused was *better* by +1.5 (6/6 prompts).
C-Q. **Grid misalignment ("lineage").** 4,680 tokens = 292.5 blocks of 16. The model's rolling eviction re-quantizes cached content on a grid shifted by 8 tokens each eviction; RTN is not idempotent across shifted grids, so error accumulates. A driver that quantizes each chunk once on a fixed grid is better than the model path by INT4 +1.29 (t=4.03, 10/10), INT2 +2.81 (t=3.35). This is also why the fused path beat the dequant path.
C-R. ⚠[분해 — 적응 선택 기여 +0.00] **Anchor oracle.** A driver that keeps the 3 least-contaminated chunks as anchors (oracle contamination from lockstep) degenerates to chunks {0,1,2} (contamination increases with age) — i.e. *the same chunks as the positional sink A2* — yet beats A2 at INT2 by +15.55 (t=8.93, 10/10): "lineage" component +2.81, "policy" component +11.20. **The +11.2 is unexplained**: selection is identical to A2, so the mechanism must be in how the cache is rebuilt/re-quantized. Current hypothesis: in the model path the sink content itself is re-quantized on the shifted grid every eviction and is destroyed at INT2, which would mean "sinks are useless at INT2" (C-D) is an artifact. **Untested.**

---

## 6. Known bugs found so far (all in *our* experimental patches, none in the base model)

A. Correction Δ = 2×step (4× bias) — fixed. B. A "Δ=0 path gate" that never exercised the explicit attention path — replaced by a real control. C. Sink-manipulation tensors leaked across prompts — fixed. D. Stale slot labels in two archived JSON files — values unaffected. E. Correction applied to BF16 self block + drop-metric artifact — fixed, all correction numbers re-derived. Also: two bugs in the upstream UCSD KIVI implementation (zero-point clamp, fp16 EPS underflow) — fixed, reverses their KIVI-vs-RTN conclusion. We assume there is a sixth.

---

## 7. Prior work we position against (check our novelty claims against these)

- TUM, "Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion", arXiv 2605.26266 — the Jensen bias, exact and Taylor forms, FlexAttention implementation (~5% overhead), MAGI-1/SkyReels-V2/HY-WorldPlay. Reports Taylor *over*-estimates at large noise.
- Quant VideoGen, arXiv 2602.02958 — INT2 KV via vector quantization, fused dequant kernel (1.5–4.3% overhead), 6.9–7.1× per-token, includes Self-Forcing-1.3B, 700-frame protocol.
- UCSD 33-method study, arXiv 2603.27469 — benchmarks on Self Forcing; notes transient BF16 buffer reconstruction erases nominal savings.
- KV-AdaQuant "More for Keys, Less for Values", arXiv 2502.15075; AsymKV — key-heavier bit allocation in LLMs. TurboQuant-inspired statistical analysis, arXiv 2605.08114 — softmax convexity makes keys more sensitive than values (theory, LLM).
- LongLive (retrained sink3+window9); Deep Forcing, arXiv 2512.05081 (training-free deep sink + pruning, no quantization); Forcing-KV, arXiv 2605.09681 (head-wise KV pruning); VideoMLA (low-rank latent KV cache).

Our claimed novelty: (i) propagation/injection decomposition and its magnitude; (ii) key causality by intervention and the attention-inversion trajectory in video diffusion; (iii) sinks as content anchors; (iv) TUM structure ≡ scalar; (v) grid-misalignment lineage; (vi) additive recipe; (vii) packed/fused kernels on a consumer GPU.

---

## 8. What is planned next (so you can judge whether it addresses the right weaknesses)

Session 6: grid-alignment hypothesis (bit-level diff of oracle vs positional; aligned model path; re-measure C-D at INT2); scalar correction as a one-parameter method + concentration derivation + robustness on 4 cells (RTN g32, KIVI, LongLive, SkyReels-V2); decomposition of the remaining 66% after correction (TF/FR, then alignment / anchor / key-bits each on top); LongLive stacking test (LongLive + K3V2 + λ=1 vs LongLive INT4 at equal quality with −30% cache); head-to-head vs pristine Self-Forcing on the same GPU; VBench added to all key tables. Kernel track: ncu roofline, fused ≤5%, segment ring buffer (no copy, no re-quantization), quantize fused into projection epilogue, INT8 QKᵀ, multi-stream capacity.

---

## 9. Files attached

- `STATUS_2026-09-18.md` — status, claim-by-claim verdict table, to-do.
- `THESIS_by_section.md` — the intended section-by-section claims of the paper (12-page unified draft).
- `RESULTS_session5.md`, `RESULTS_session5_gates.md` — raw session-5 reports with all numbers.
- `AUDIT_REPORT.md` — the independent audit (bugs A–D; bug E is in RESULTS_session5_gates).
- `rw_D_novelty_check.md` — our own novelty check against the papers in §7.
- `HANDOFF_session6.md`, `HANDOFF_kernel_track.md` — planned experiments.
- `patches/`, `kernels/`, `results/*.csv` — (to be attached from the GPU server by the author).

Questions we most want answered: Is C-A's 5:1 confounded by the lineage effect in C-Q? Is C-J trivially true (softmax shift-invariance) rather than a finding? What explains the +11.2 in C-R if not the grid hypothesis? Are the paired t-tests on 10 prompts and the leave-one-out claims computed correctly from the CSVs? Which of our seven novelty claims does not survive §7?
