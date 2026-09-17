# Session notes — Self Forcing reproduction + first frame-axis quantization error curves

Date: 2026-09-16 · RTX 4090 24GB · see `env.txt` for exact versions.

---

## 1. Did seed pinning pass bit-exactness?

**Yes, and it passes a stronger test too.**

| check | result |
|---|---|
| same seed, run twice, `torch.equal` on every chunk latent | **PASS** (21/21 tensors, 3 prompts x 7 chunks) |
| same seed, but the global CUDA RNG is deliberately drained by 7777 draws per chunk | **PASS** (21/21) |
| same seed, before vs after applying the UCSD `causal_model.py` patch | **PASS** (63/63) |

The second check is the one that matters. Plain re-run bit-exactness does *not*
prove what we need: the plan's worry was that a quantization method which
consumes RNG would silently shift every later draw and move the whole
trajectory. The original code sampled the initial noise up front (deterministic),
but `pipeline/causal_inference.py` also called `torch.randn_like()` on the
**global** CUDA RNG once per denoising step for re-noising. Any extra RNG
consumer anywhere would have desynchronized that stream.

Fix: two independent per-chunk `torch.Generator` streams, neither of which
depends on execution history.

- initial chunk noise: `seed * 1000003 + chunk_idx`
- re-noising inside the chunk: `seed * 1000003 + 7919 + chunk_idx`

They are separate streams so that a chunk's re-noise is not correlated with its
own initial noise.

## 2. How the UCSD code was integrated

Their `scripts/01_generate.py` expects Self-Forcing under `third_party/` and
drives it with its own harness. Going that route would have lost our seed
pinning and per-chunk latent dumping, which the error curves depend on — so
instead we attached **their quantizers to our patched pipeline**:

1. Applied their `docs/patches/self_forcing_kv_quant.patch` verbatim. It only
   touches `wan/modules/causal_model.py`, which we had not modified, so there is
   no conflict. Verified BF16 stays bit-exact after applying it (the patch is a
   no-op when no quantizer is attached).
2. `_initialize_kv_cache` now seeds each layer dict with
   `"quantizer": getattr(self, "kv_quantizer", None)` and `"quant_state": None`.
3. `inference.py --kv_quant RTN --kv_bits 4` builds the quantizer through their
   `kv_quant.factory.create_quantizer` and sets `pipeline.kv_quantizer`.

**Bug found and fixed in their code.** `kv_quant/factory.py` passed `key_bits`,
`value_bits` and `name` to `RTNQuantizer`/`KIVIQuantizer` without ever binding
them — they arrive through `**kwargs` but were never popped. Both paths raised
`NameError: name 'key_bits' is not defined`. That is exactly the RTN INT4 / RTN
INT2 / KIVI INT4 set this session was asked to run, so nothing in Step C could
have run without fixing it. Minimal fix: pop them from `kwargs`. See
`../patches/ucsd_factory_kwargs.patch`.

**Cache reuse across prompts.** Their patch replaces `kv_cache["k"]` with an
empty tensor once a quantizer is active, but the pipeline's between-prompt reset
only zeroes the index counters. We additionally clear `quant_state` and
reallocate the k/v buffers on that path.

## 3. A blocker in Self-Forcing itself: 63-frame rollout could not run

`WanDiffusionWrapper` defaults to `local_attn_size=-1` and neither shipped
config overrides it, so the rolling-eviction branch in
`wan/modules/causal_model.py` is **disabled**. But the KV cache buffer is sized
`32760` tokens = exactly 21 latent frames. At 21 frames it fits exactly; past
that the write slice and the source tensor disagree in shape and it dies.

So Step B as written was not runnable. Fix: `configs/self_forcing_dmd_long.yaml`
sets `model_kwargs.local_attn_size: 21`, which turns the eviction path on.

Consequence for reading the curves: with 3 latent frames per chunk, the cache
holds exactly 7 chunks, and chunk 6 fills it to the last token. **Eviction starts
at chunk 7** — that is where the vertical marker goes in `error_curves.png`.

The 21-frame runs are unaffected by this setting (verified: the eviction branch
never triggers within 21 frames), so the Step A reference stays valid.

## 4. Measured peak VRAM

| run | frames | mean s/video | peak allocated | peak reserved |
|---|---|---|---|---|
| bf16 | 21 | 13.9 | 12.74 GB | 15.68 GB |
| bf16 | 63 | 46.6 | 13.16 GB | 19.40 GB |
| RTN INT4 (KV cache) | 63 | 58.0 | **10.70 GB** | 15.94 GB |

Two things worth flagging against the plan's expectations:

- The plan expected roughly 19 GB peak for the 21-frame baseline; we see
  **12.74 GB allocated**. (Reserved does reach 15.7 GB.)
- Tripling the rollout from 21 to 63 frames barely moves peak memory
  (12.74 -> 13.16 GB), which is the rolling cache doing its job: KV residency is
  capped at 21 frames regardless of video length.
- **The "quantized yet higher peak VRAM" effect did not reproduce here.** RTN
  INT4 lands 19% *below* BF16, at 24% more wall-clock. Dropping ~6 GB of
  persistent BF16 cache outweighs the temporary per-layer dequantization buffer
  in this configuration.

## 5. The latent L2 metric does not measure what the plan assumes

This is the main methodological finding of the session and it affects how every
curve should be read.

The error metric the plan specifies is
`rel_err[c] = ||x_q[c] - x_bf16[c]|| / ||x_bf16[c]||`. Under quantization this
saturates near or above 1.0 within about 8 chunks. A value of 1.0 means the
quantized latent is essentially **uncorrelated** with the reference.

But the videos are fine. Decoded frame 40 of prompt 0:

| config | rel_err @ chunk 20 | what the video actually looks like |
|---|---|---|
| BF16 | — | matches the prompt |
| W8A8 (FP8) | 0.94 | same scene, same subject, coherent; different extras |
| W4A16 | 1.24 | fully coherent, different camera framing |
| W4A4 | — | coherent, slightly flatter detail, not broken |

So the metric is tracking **trajectory divergence in a chaotic sampler**, not
quality loss. A 4-step distilled model nudged early lands on a different but
equally valid sample. Latent L2 against BF16 cannot, on its own, distinguish
"error accumulated through the KV cache" from "the sampler went somewhere else".

Two consequences, both acted on:

1. **Added a control** the plan does not include: BF16 everywhere, with the
   initial noise multiplied by `(1 + eps*N(0,1))` for `eps = 1e-3` and `1e-5`.
   This measures how fast this model diverges from *any* perturbation. It is the
   baseline the quantization curves have to be read against — if they track the
   control, there is no cache-specific accumulation to report.
2. **Recorded the KV-cache quantization error directly** (the plan's optional
   item in section 6): relative error between the cache contents before storing
   and after reloading, averaged over layers, per chunk. This is immune to the
   divergence confound.

### Was the quantizer itself too crude?

Checked before blaming the model, because a bad quantizer would produce the same
symptom. It is not.

On real checkpoint weights (180 target tensors): FP8 per-tensor 0.0265 vs FP8
per-channel 0.0264 — identical, so there are no outlier channels. INT4 group-128
gives 0.1225, normal for uncalibrated RTN. INT8 group-128 gives 0.0061.

On real activations captured during a forward pass, per-tensor vs per-token FP8
came out 0.0263-0.0290 vs 0.0257-0.0265, with max/median row magnitude ratios of
only 1.2-4.6. Pathological activation outliers run 50-1000x. **This model does
not have them**, so the simple per-tensor FP8 scale is not costing anything and
was left alone.

## 6. Additions beyond the written plan

- `w8a16` (INT8 weights, BF16 activations). Every config the plan specifies
  pushes past the divergence ceiling within ~8 chunks; INT8 weights perturb
  about 20x less, keeping the early curve readable.
- The two noise-perturbation control runs described above.
- `--kv_err_log`, the in-situ KV cache quantization error.

Nothing specified by the plan was dropped or substituted.

## 7. Environment deviations

- No conda on this host; used `venv` at `~/gpu/sf-venv` (`python3.10-venv`
  installed via apt).
- `requirements.txt` pulls `nvidia-tensorrt`, `pycuda`, `onnx*`, `dashscope`,
  CLIP, wandb and flask, none of which the inference path imports — installed a
  trimmed set instead (`requirements-infer.txt`). `lmdb` *is* needed
  (`utils/dataset.py` imports it) and was added back after the first run failed.
- flash-attn 2.7.4.post1 installed from the prebuilt cu12/torch2.6/cxx11abiFALSE
  wheel; no source build was needed. (An SDPA fallback exists in
  `wan/modules/attention.py` and would have been used automatically otherwise.)
- `transformers` resolved to 5.17.0, well past the `>=4.49.0` the repo asks for.
  It is only used for `AutoTokenizer` (T5 is reimplemented in-repo) and works.

## 8. What the curves actually show

`error_curves.png` (latent L2 vs BF16) and `kv_cache_error.png` (in-situ cache
error). Dashed black in the first plot is the control.

### The control is a hard floor, and it is perturbation-size independent

A 10x larger nudge gives the same curve: mean difference between `eps=1e-2` and
`eps=1e-3` is **+0.024** over chunks >= 5. That is the signature of chaotic
saturation, and it means the control is a property of the sampler, not of the
perturbation. Read every other curve as a multiple of it.

(A third control at `eps=1e-5` returned exactly 0.0000 at every chunk. That is
not a divergence measurement — bf16 has ~0.4% relative resolution, so a 1e-5
perturbation rounds away entirely and the noise tensor is unchanged. It is a
fourth bit-exactness check by accident; it is excluded from the plot.)

| config | rel_err @ chunk 20 | vs floor | reading |
|---|---:|---:|---|
| control (1e-2 / 1e-3) | 0.74 - 0.75 | 1.00x | chaos floor |
| **RTN INT4 (KV cache)** | 0.81 | 1.09x | on the floor — no cache-specific accumulation |
| W8A16 | 0.84 | 1.13x | on the floor |
| W8A8 (FP8) | 0.94 | 1.27x | slightly above |
| K/V-projection-only W4A4 | 1.09 | 1.47x | above |
| W4A16 / W4A8 | 1.24 | 1.68x | above |
| KIVI INT4 (KV cache) | 1.28 | 1.73x | above |
| W4A4 | 1.34 | 1.81x | above |
| **RTN INT2 (KV cache)** | **1.85** | **2.51x** | real accumulation, and the video collapses |

### Answering the shape question

- **RTN INT4: flat.** Indistinguishable from the control from about chunk 8 on
  (chunk 9: 0.624 vs the control's 0.621). What earlier looked like "error
  accumulates then saturates" is just the divergence floor. Without the control
  this would have been misread as absorption-after-accumulation.
- **RTN INT2: step, then linear.** It tracks the others until the eviction
  boundary, kinks upward at **chunk 7 — exactly where the rolling cache starts
  discarding the oldest chunk** — and then climbs roughly linearly to 1.85,
  well past the sqrt(2) uncorrelated line. This is the one curve in the sweep
  where the plan's "step at the eviction point" prediction actually shows up.
- Everything else sits on a band between the two, rising in parallel with the
  control (same slope, higher offset). Parallel to the floor means the extra
  error is injected per chunk, not compounded across chunks.

### The direct cache measurement settles the mechanism

Relative error between what goes into the cache and what comes back out,
averaged over 30 layers x 3 prompts:

| method | chunk 8 (K) | chunk 20 (K) | change | chunk 20 (V) |
|---|---:|---:|---:|---:|
| RTN INT4 | 0.0280 | 0.0278 | **-0.5%** | 0.0371 |
| RTN INT2 | 0.1383 | 0.1298 | **-6.1%** | 0.1728 |
| KIVI INT4 | 0.1907 | 0.1841 | **-3.5%** | 0.0569 |

**The cache error does not accumulate — for any method.** It is flat, in fact
mildly decreasing, once the cache fills. The UCSD patch dequantizes and
re-quantizes the *entire* cache every chunk, so compounding was the obvious
thing to expect; it does not happen, because RTN-style rounding is near
idempotent (a value already on the quantization grid maps to itself).

Put together with the latent curves, the mechanism is:

> Error does not compound inside the KV cache. Bit-width sets the size of a
> **constant** per-chunk error injection (2.8% at INT4, 13% at INT2). Rollout
> quality collapses when that constant exceeds what the model's own dynamics can
> absorb — not because the cache degrades over time.

That is a negative result for the accumulation hypothesis as stated in section
0 of the plan, at least for RTN/KIVI-style cache quantization.

### KIVI INT4 is much worse than RTN INT4, and the cache measurement says why

KIVI is the more sophisticated method (per-channel keys, per-token values) and
should beat RTN at equal bits. Here it is nearly twice as far from the floor
(1.28 vs 0.81). The cache measurement localizes it: KIVI's **key** error is
**0.184 vs RTN's 0.0278 — 6.6x worse at the same nominal 4 bits**, and worse even
than RTN INT2's keys (0.130). Its value error (0.057) is the better half of the
trade but does not pay for it.

Also worth noting: RTN INT2 (k=0.130, v=0.173) ends up *worse* in latent terms
than KIVI (k=0.184, v=0.057) despite lower key error — which hints that **value
error costs more than key error** in this pipeline. Worth a targeted K-vs-V
bit-allocation sweep next session.

### The K/V-projection ablation is inconclusive at W4A4

Growth from chunk 0 to chunk 20: K/V-projections only **+0.61**, everything
except K/V **+0.68**, all layers **+0.71**, control **+0.67**. The *offsets*
differ (K/V-only starts at 0.475 from just 60 of 300 layers, so the cache path
is disproportionately damaging per layer) but the *slopes* are all the control's
slope. At W4A4 there is no measurable cache-specific accumulation; only the
INT2 cache run shows that.

## 9. Open questions for the next session

1. **Why is KIVI's key quantization so poor here?** 6.6x worse than plain RTN at
   the same bit-width is large enough to suspect the implementation rather than
   the method. Check `kv_quant/kivi.py` against the paper before drawing
   conclusions about KIVI as a technique.
2. **Values look more fragile than keys.** Run asymmetric allocations (K4V8,
   K2V8, K8V2) and see whether the latent curve tracks value error specifically.
3. **Where is the collapse threshold?** INT4 (2.8% cache error) is safe, INT2
   (13%) collapses. A 3-bit run, or RTN INT4 with a deliberately coarsened block
   size, would locate the knee.
4. **The metric needs replacing for quality claims.** Latent L2 vs BF16 saturates
   at the chaos floor and says nothing about quality above it — W4A16 sits at
   1.24 with a perfectly good video, RTN INT2 at 1.85 with a broken one. A
   reference-free or perceptual measure is needed before any quality ranking.
   (VBench was explicitly out of scope this session.)
5. **Longer rollouts.** 63 frames gives 14 chunks past the eviction boundary.
   The INT2 curve was still climbing at chunk 20; it is unclear whether it
   plateaus or keeps going.
6. **The plan expected ~19 GB peak at 21 frames; we measured 12.74 GB
   allocated.** Worth confirming which configuration the estimate came from.

