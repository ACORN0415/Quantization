# P1/P3 main study v2 — fixed before held-out measurements

Version 1 failed a development gate before any held-out sampling: own-state
INT2 P1 noise became nonfinite at chunk 19 (zero-indexed). Version 2 fixes the
control amplitude schedule to the paired quantized FR's actual innovations.
Primary endpoints, prompts, seeds and analysis are unchanged. See
[P1_P3_CONTROL_AMENDMENT.md](P1_P3_CONTROL_AMENDMENT.md).

## Question and scope

Compare complete P1 rolling and P3 chunk storage policies at the same nominal
RTN bit width on Self-Forcing / Wan2.1-T2V-1.3B. This is a policy evaluation,
not a new-method claim or an equal-memory comparison. Group boundaries,
padding, current-chunk retention and re-quantization all differ. Do not attribute
the policy contrast to re-quantization alone.

Parameters are fixed in [policy_study_plan.json](policy_study_plan.json).
Use the 10 original prompts in [policy_study_prompts.txt](policy_study_prompts.txt),
seeds 101 and 202, 126 latent frames / 42 chunks, 21-frame context, sink 0,
residual 0, RTN group 16, INT4 and INT2. These are convenience prompts spanning
several subjects, not a random sample from a video-prompt population. No prompt
replacement or sample-size change based on generated quality is permitted.
All previously generated prompts are development-only and excluded.

## Arms and teacher forcing

For each prompt/seed/bit/policy, run BF16 reference, replay-TF, quantized FR and
storage-noise FR. There are 80 policy cases, 240 TF/FR/noise tensor artifacts and
20 shared BF16 references. BF16 is recomputed for replay in each case and must
match the saved reference bit-exactly; its decoding/quality is reused.

Replay every reference K/V write through a persistent teacher store, including
all four denoising calls and context refresh. Clone before each current chunk;
TF writes affect the independent working clone only. Discard that working
clone afterwards. Use this definition for both policies. This matches the
storage schedule, not realized input-dependent error histories. TF video quality
is secondary: VAE decoding of stitched counterfactual chunks is not teacher-forced.

FR historical KV errors are recorded in a separate single-trajectory pass,
which is also the final FR generation pass. TF history diagnostics reuse the
aligned reference cache, excluding current-chunk tokens. Both measure actual
pre-attention historical K/V against insertion-time BF16 originals. Record all
layers/calls; report macro averages over the specified chunks, layers and calls.

## Noise control

First record the absolute FP32 innovation norm at every actual encode event
of the paired quantized FR, separately for K and V. On the noise arm's own
input x compute e = Q(x)-x to determine eligible coordinates only. Draw an
independent Gaussian direction, set entries with e=0 to zero, normalize to
the corresponding **recorded FR norm**, add to x and cast to BF16.
Keep the result under the same storage policy. This preserves the control
input's eligible error support and the paired FR's absolute innovation norm
before rounding; record the actual rounded norm.
The direction is isotropic on that support with fixed norm, not an unnormalized
Gaussian perturbation. Each layer uses a private RNG with seed
1000000 + prompt_index*10000 + generation_seed*100 + layer_index.

This differs from the earlier development control that perturbed every tensor
coordinate. A development gate must verify the new support mask, RNG isolation,
zero-error identity, end-to-end feasibility and rounded norms before held-out
measurements. The target is not a held-out quality-tuned quantity: Q and the norm
rule are fixed before measurements. The absolute norm schedule is calibrated
from the paired FR before its quality is inspected and never scaled using the
control's growing state. Eligible support is determined on the control input,
not matched coordinate-for-coordinate to FR. This is neither matched accumulated read-error nor an initial
BF16 perturbation chaos baseline. It cannot by itself establish a unique cause
for large FR distances.

Predeclared instability handling: a nonfinite noise latent, K/V input or
would-be support quantization is a secondary control instability outcome.
Retain its finite prefix and failure location. Do not replace the seed or
shrink its amplitude. Report the count out of all 20 prompt/seed controls per
policy/bit cell. If any control in that cell fails, its full-sample noise MUSIQ
mean and quantized-minus-noise quality contrast are undefined; do not report
a selected finite-only contrast or impute a quality score. This rule is fixed
before held-out sampling. Zero-noise identity and calibration validity are
still mandatory; any nonfinite primary TF/FR or invalid calibration halts.

## Endpoints and analysis

Primary endpoint: average MUSIQ on the middle raw decoded frame of each late
chunk 21–41 (zero-indexed), before lossy video encoding. Pixel frame index is
round((3*c+1.5-1)*4+1), clipped to the video length, matching the pilot.
For each bit width compute P3 FR minus P1 FR within each prompt/seed, then
average the two seeds within each prompt. Higher is better. BF16 subtraction
cancels in this policy contrast; report the BF16-relative arm scores too.

The independent analysis unit is the prompt (n=10), never individual chunks or
seeds. Report mean paired difference and a 97.5% prompt-cluster percentile
bootstrap CI (20,000 replicates, seed 872341). This provides Bonferroni coverage
for two intervals under the bootstrap approximation. Report an exact two-sided
paired sign-flip test over 2^10 assignments, assuming exchangeability/symmetry
of paired differences under the null. Apply Holm correction to the two bit-width
tests at family alpha .05. Report assumptions and the small prompt count. No
post-hoc power, equivalence, or generalization claim.

Secondary descriptive endpoints: early chunks 1–6, middle 7–20 and late 21–41
MUSIQ; TF/FR/noise latent relative L2; common historical K/V errors; quantized FR
minus noise-FR MUSIQ within policy; and mean adjacent-frame pixel difference.
Use 95% prompt-cluster bootstrap intervals for secondary quality contrasts,
without confirmatory p-value claims. MUSIQ measures frame quality, not semantic
faithfulness or temporal/subject consistency; the pixel difference is a motion
proxy, not a temporal quality score. No FR-minus-TF causal decomposition.

## Gates, execution and retention

Before any held-out case: pass CPU/reference tests, full-model replay/zero-noise
identity on development prompts, exact INT4 compatibility with the prior
feasibility tensors, and 126-frame INT2 smoke tests for both policies including
the final noise definition and numerical-outcome handling. Decode and score finite development outputs to measure
end-to-end cost. Commit the registration seal containing source, plan, prompt
and gate hashes before held-out sampling. Gates are prerequisites, not study data.

Run policies separately on the 24GB GPU; free TF/reference states before FR
and noise passes, and free caches before VAE decoding. Reuse BF16 quality only
after exact latent equality. Execute prompt, seed, bit, policy in the fixed order
listed in the JSON plan. Do not stop based on quality or significance. Expect
roughly 12–24 hours including gates, diagnostics, decoding and controls; refine
the displayed ETA from elapsed completed cases, without changing sample size.

Infrastructure failures may be retried twice with identical parameters and
frozen code. Never drop a numerically bad case or silently replace a prompt.
Record failures; nonfinite primary outputs or failure of a validity gate halt the study
as invalid pending an explicit documented amendment and new registration.
Resume only completed matching-source stages. Keep every final latent tensor,
raw per-call metrics, quality values and source fingerprints outside Git;
publish compact evidence, CSVs, plots and conclusions. Secondary noise instability
is reported under the rule above and never silently excluded. Keep licenses and source
attribution. Do not publish partial quality conclusions before gates pass.

## Visibility

`bash tools/watch_progress.sh` shows current case/stage and queue completion.
`tail -F ~/sfq_data/policy_study.log` shows the live log. Machine-readable state
is `~/sfq_data/policy_study/status.json`. The runner proceeds through all cases
and writes the final analysis without a manual handoff.
