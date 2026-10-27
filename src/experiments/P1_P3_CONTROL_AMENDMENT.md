# Development control amendment — before held-out measurements

The v1 gates on implementation `e891179` failed at development prompt 0,
seed 0, P1 RTN2, chunk 19 (zero-indexed): the noise trajectory produced
nonfinite latents. Quantized FR and TF completed all 42 chunks with finite
latents. No held-out prompt was sampled; no main-study registration seal existed.
Raw failed-gate outputs and logs are retained under
`${HOME}/sfq_data/policy_study_v1_failed_control` and the corresponding log.

The v1 control recomputed Q(x)-x on its own noisy state at every encode.
This does not match the actual innovations of quantized FR: repeated rounding
on an already quantized state can be nearly idempotent, whereas randomized
states keep generating new large innovations. Norm-dependent noise can amplify
itself. A cache-only reproduction with fresh fixed-distribution writes
(42 chunks, 5 writes/chunk, 7-chunk capacity, seed 77) showed K norms:

| Chunk | Quantized | Own-state noise |
|---|---:|---:|
| 0 | 19.43 | 19.98 |
| 6 | 48.99 | 640.29 |
| 19 | 47.65 | 376977.19 |
| 41 | 48.92 | nonfinite |

This demonstrates amplification even without model feedback. It is not a
quality result and does not establish every cause of the model failure.

Version 2 records **absolute per-encode innovation norms from the paired
quantized FR** and replays that fixed schedule on the noise trajectory.
Norms are recorded without changing the quantized states. Random directions
still use a private RNG and the control input's would-be quantization-error
support, preserving unused zero coordinates. The support is not claimed to
match FR coordinate-for-coordinate. Verify event counts, actual rounded norms,
zero-error identity, frozen source and full-model compatibility again.

No amplitude clipping, favorable-seed selection, prompt exclusion or quality
threshold was used to make the control pass. The planned sample, primary
policy contrast, endpoint, time windows and analysis remain fixed. The new
control estimates the consequence of replacing actual FR storage-innovation
magnitudes with random directions along an independently evolving trajectory;
it is still not a matched read-error or chaos-baseline comparison.

Version 2 also fixes numerical-outcome handling before held-out sampling.
The secondary control is allowed to exhibit genuine numerical instability;
this is counted and retained rather than used to replace a seed, lower the
amplitude or select a finite-only sample. A cell's full-sample noise-quality
contrast is undefined if any of its 20 controls fails. Primary TF/FR and the
calibration schedule must remain finite, and zero-noise identity must pass.
This keeps failure visible without treating an unstable secondary arm as a
reason to silently discard valid primary policy comparisons.
