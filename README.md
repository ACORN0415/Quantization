# Quantization error propagation in causal video diffusion

Does low-precision quantization error **accumulate along the frame axis** through the KV cache of a streaming (causal) video diffusion model?

Two sessions of measurements on [Self-Forcing](https://github.com/guandeh17/Self-Forcing) and [LongLive](https://github.com/NVlabs/LongLive) (both Wan2.1-T2V-1.3B based), using the KV-cache quantizers from [kv-quant-longhorizon](https://github.com/suraj-ranganath/kv-quant-longhorizon). Single RTX 4090 (24 GB).

---

## Answer

**Yes, it accumulates — and an attention sink stops it.**

Getting there required discarding the first session's conclusion, because the metric it rested on cannot measure accumulation.

| | Self-Forcing (no sink) | LongLive (sink 3 + window 9) |
|---|---:|---:|
| eviction starts at chunk | 7 | 4 |
| KV INT4 — cache error (key) | 0.0278 | **0.0640** |
| KV INT4 — MUSIQ change over rollout | **−11.0** | **−0.3** |
| KV INT2 — damage above own chaos floor | **2.51×** | **1.12×** |

LongLive takes **2.3× more quantization error into its cache** and ends up essentially lossless at INT4. The sink does not reduce how much error is injected; it stops that error from compounding.

And the protection comes from the sink's **position being preserved**, not from its numerical precision — quantizing the sink slots to INT2 costs nothing measurable (1.11× vs 1.12×).

---

## Why the first session got it wrong

Session 1 concluded "no accumulation" from two observations, both of which turned out to be artifacts:

1. **In-situ cache error is flat across chunks.** True, but that measures *injection*, not *propagation*. RTN rounding is near-idempotent, so re-quantizing a cache every chunk does not compound — yet the downstream consequence still grows.
2. **The latent L2 curve against BF16 tracks the control.** The latent metric saturates: a 4-step distilled sampler nudged early lands on a *different but equally valid* sample, so the error rises to a chaotic-divergence ceiling and stops discriminating.

The second point is worth stating sharply, because it is easy to repeat:

> The chaos floor (BF16 + a 0.1 % noise perturbation) reaches rel_err **0.738** at chunk 20 while its MUSIQ **improves by +4.3**. Fixed KIVI INT4 sits at rel_err **0.753** — statistically the same distance from BF16 — with MUSIQ **−10.3**. Two runs equidistant from the reference, opposite quality outcomes.

The converse also happens: a *buggy* KIVI run reached rel_err 1.277 (far past "uncorrelated") with MUSIQ only −3.4, because it produced a sharp, high-contrast, entirely different scene. Neither metric alone is sufficient.

### What fixed the measurement

**Teacher-forced / free-running separation.** At every chunk, a reference BF16 pass and a quantized pass run over the *same* context in lockstep, so the quantized error at that chunk is the error injected there alone:

```
e_inj[t]       = teacher-forced error   (context re-anchored to BF16 each chunk)
e_total[t]     = free-running error
propagation[t] = e_total[t] − e_inj[t]
```

Injection turns out flat; propagation grows monotonically for **all five** configurations tested, INT4 included.

**A reference-free quality metric** (MUSIQ, via `pyiqa`) alongside it, since latent L2 saturates at the chaos floor.

Caveat carried through the results: `FR − TF` includes chaotic divergence as well as cache propagation. FP8 W8A8 has *larger* propagation than RTN INT4 (0.875 vs 0.591) yet **no** quality loss (MUSIQ +1.4 vs −11.0). Propagation and a reference-free metric must be read together.

---

## Other findings

### Key error costs ~9× more than value error

Controlled experiment, same quantizer, only the bit allocation differs:

| config | key error | value error | MUSIQ change |
|---|---:|---:|---:|
| INT4 baseline | 0.0278 | 0.0369 | −11.0 |
| K4V2 (good keys, bad values) | 0.0280 | **0.1788** | −13.8 |
| K2V4 (bad keys, good values) | **0.1291** | 0.0358 | −36.9 |

Keys pass through the softmax and are exponentially sensitive; values contribute linearly. Key error changes *which* tokens are attended to, value error only blurs *what* is retrieved.

This also explains Wan's design: `norm_q`/`norm_k` apply RMSNorm to queries and keys while **values have no normalization**, which keeps key error low at a given bit-width — fortunate, given keys are the sensitive side. Spend bits on keys.

### There is no collapse threshold

Session 1 reported a knee between "2.8 % cache error is safe" and "13 % collapses". There isn't one. Raising key error from 0.0278 to 0.0418 (via coarser block size) leaves latent rel_err unchanged at ~0.81 — but MUSIQ moves −11.0 → −14.5. The apparent knee was the latent metric's ceiling. Damage is continuous and near-monotonic in key error.

### Two bugs in the upstream KV-quantization code

Filed as `issue/ISSUE_zp_clamp.md` with a runnable reproduction. Not yet posted upstream.

**Bug 1 — the zero point is clamped into the quantized range.** `quantize_asym` does `zp = round(qmin − x_min/scale).clamp(qmin, qmax)`. The zero point is an *offset*, not a code; clamping collapses any group that does not straddle zero. Post-RoPE keys hit this constantly. On an all-positive group: asymmetric INT4 gives rel_err **0.9296** versus symmetric INT4's **0.0399** — 23× worse, with 100 % of groups affected. `RTNQuantizer` uses `quantize_sym` and is unaffected, which is exactly why RTN looked fine and KIVI looked broken.

**Bug 2 — revealed only by fixing the first.** `EPS = 1e-8` underflows to zero in the fp16 the scale is stored in, while `zp = −x_min/1e-8` overflows to `inf`, so dequantization yields `(q − inf) × 0 = NaN`. With the original clamp in place this never surfaced — the group silently dequantized to zero instead. Whole 63-frame runs came back NaN once the clamp was removed.

The fix stores the group minimum instead of a zero-point code (same storage, numerically safe). After it, KIVI INT4's key error drops **0.1841 → 0.0176**, now *better* than RTN INT4's 0.0278 — the expected ordering for asymmetric vs symmetric at equal bit-width. **The session-1 result "KIVI is worse than RTN" was entirely an implementation bug.**

A third, unrelated bug: `create_quantizer` references `key_bits`/`value_bits`/`name` without binding them from `**kwargs`, so every RTN and KIVI construction raises `NameError`.

---

## Repository layout

```
docs/     handoffs, results reports, environment notes
patches/  diffs against each upstream repo, plus the configs and sweep drivers
tools/    measurement and instrumentation scripts
results/  per-chunk CSVs and figures
issue/    upstream bug report draft + minimal reproduction
```

Large artifacts are not in git: per-chunk latents (~2.2 GB) and generated videos live outside the repo.

### Documents

| file | what |
|---|---|
| `docs/RESULTS_4090_session.md` | session 1 — reproduction, first error curves, the metric problem |
| `docs/RESULTS_session2.md` | session 2 — TF/FR separation, KIVI bugs, sink experiments, threshold sweep |
| `docs/NOTES_session1.md` | session 1 working notes (English) |
| `docs/SETUP.md` | environment reproduction (venv, not conda) |
| `docs/env.txt` | exact versions, checkpoint hashes, config values |
| `docs/HANDOFF_*.md` | the instructions each session worked from |

### Key figures

| file | shows |
|---|---|
| `results/propagation.png` | TF (injection) / FR / propagation, three panels |
| `results/longlive_vs_selfforcing.png` | both models plus damage above each one's own chaos floor |
| `results/threshold.png` | cache error vs rollout damage, keys and values separated |
| `results/kv_cache_error.png` | in-situ cache error is flat along the chunk axis |
| `results/iq_*.png` | MUSIQ curves |

---

## Reproducing

Full setup in `docs/SETUP.md`. Short version:

```bash
# Self-Forcing at 33593df, kv-quant-longhorizon at b4c0936, LongLive at v1.0 (e52d9ef)
git apply patches/self-forcing/0{1,2,3}_*.patch          # in the Self-Forcing checkout
git apply patches/kv-quant-longhorizon/0{5,8}_*.patch    # in the kv-quant checkout
git apply patches/longlive/07_longlive_port.patch        # in the LongLive checkout
cp tools/*.py <checkout>/                                # measurement scripts
```

Two things that will bite you:

- **`local_attn_size` defaults to `-1`**, which disables rolling eviction — but the cache buffer is sized to exactly 21 latent frames, so anything longer dies with a shape mismatch. `patches/self-forcing/self_forcing_dmd_long.yaml` sets it to 21. With 3 frames per chunk, eviction then begins at chunk 7.
- **`requirements.txt` pulls tensorrt, pycuda, onnx, CLIP, wandb and flask**, none of which the inference path imports; `docs/requirements-infer.txt` is the trimmed set. But `lmdb` *is* required (`utils/dataset.py` imports it unconditionally) and is easy to miss.

### Reproducibility gates

Every experiment is gated on bit-exactness before any quantized run:

| gate | why it matters |
|---|---|
| same seed, run twice | baseline determinism |
| same seed, global CUDA RNG drained by 7777 draws per chunk | the pipeline calls `torch.randn_like` on the **global** RNG once per denoising step; any quantizer that consumes RNG would silently shift the whole trajectory |
| before vs after applying the upstream KV-quant patch | the patch must be a no-op without a quantizer attached |
| lockstep reference trajectory vs the plain free-running BF16 run | otherwise `FR − TF` subtracts across different references |

The second gate is the one that matters. Plain re-run determinism does not prove what is needed.

Seeding uses two independent per-chunk generator streams so neither depends on execution history:

```
initial chunk noise :  seed * 1000003 + chunk_idx
re-noising          :  seed * 1000003 + 7919 + chunk_idx
```

---

## Open questions

1. **Sink size vs residual sensitivity.** Pinning chunk 0 removes 67–94 % of the chunk-7 sensitivity step, but a small residual remains. Sweeping sink size (6, 9) would separate what is left from the RoPE-position hypothesis.
2. **KIVI's residual window.** The upstream implementation has none, so these numbers are "KIVI-style", not KIVI. Implementing it would allow a real comparison.
3. **Longer rollouts.** Propagation had not flattened by chunk 20 in Self-Forcing.
4. **A reference-based perceptual metric** (LPIPS against BF16) to triangulate — MUSIQ can reward a divergent but punchy output, as the buggy-KIVI case showed.
5. **Wan 14B / LongLive-2.0-5B activation outliers** — needs an 80 GB GPU; 1.3B has none (row max/median 1.2–4.6×, versus the 50–1000× that signals a real outlier problem).
