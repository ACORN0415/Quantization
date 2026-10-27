# P3 TF/FR pilot results

Fixed protocol: [P3_TF_FR_PILOT.md](../P3_TF_FR_PILOT.md).
Two new prompts, seed 0, 126 latent frames, RTN P3 group 16, no sink/bias.
The original BF16/plain-snapshot gates passed before measurement. The actual
`anchor_store` identity path was validated retrospectively after the pilot,
passing all 42 TF and reference comparisons; see [anchor_validity.json](anchor_validity.json).

| Bits | Prompt | Late TF rel | Late FR rel | Late TF−BF16 MUSIQ | Late FR−BF16 MUSIQ |
|---|---|---|---|---|---|
| 4 | 0 | 0.1288 | 0.7328 | -1.35 | -8.11 |
| 4 | 1 | 0.1592 | 1.0455 | -1.49 | -12.30 |
| 2 | 0 | 0.3116 | 1.4813 | -7.15 | -40.86 |
| 2 | 1 | 0.3196 | 1.5516 | -3.58 | -49.14 |

Late = chunks 21–41. All window summaries exclude chunk 0.
These are descriptive observations from n=2, not evidence of equivalence or generalization.
FR−TF distances are not an additive causal decomposition; chaotic divergence is included.
MUSIQ and the motion proxy do not establish subject/identity preservation.
Sampling peak includes concurrent reference/TF/FR caches and is not a per-policy memory comparison.

![Latent distances](latent.png)
![MUSIQ](musiq.png)

Per-window values: [windows.csv](windows.csv). Full compact evidence: [evidence.json](evidence.json).
Videos remain outside Git at `${DATA_ROOT}/p3_pilot`. Derived latent tensors were removed only after successful video/metric export; hashes and removed byte counts are preserved.

## Interpretation

In all four pilot cases, late free-running latent distance exceeded the
teacher-forced distance, and late FR MUSIQ was lower than TF MUSIQ. This shows
that the distinction remains observable in this once-per-chunk simulation; it
is not limited to the reconstructed rolling/requantization path.

This is a descriptive result for two prompts and one seed. BF16 perturbation
controls, subject consistency, additional seeds/prompts and another model are
still needed before a general propagation or publication claim. TF videos are
assembled from chunkwise counterfactual outputs; VAE decoding itself was not
teacher-forced. Their MUSIQ is therefore an assembled-video diagnostic, not a
pure per-chunk denoiser-injection quality estimate.

Consistency checks passed: all four cases use the gated source fingerprint,
all values are finite, each case exports three 501-frame videos, and the two
bit-width arms have identical BF16 latent-file hashes and BF16 quality arrays
for each prompt. See [consistency.json](consistency.json).

The initial export removed **704,489,580 bytes**, including redundant gate
tensors. This historical cleanup is preserved in the record. Final research
latents have now been restored: all **12/12 SHA256 values** match, and all
**168/168 latent-distance rows** match. The 603,846,220 bytes of tensors are
retained; this is artifact recovery, not a memory-saving result.


## Follow-up validation and retention

The actual identity-quantizer anchoring gate passed on the unchanged original
pilot implementation: **42/42 teacher and reference comparisons**, with the
segment encoding path exercised. This retrospective validation does not alter
the registered analysis or turn the pilot into a confirmatory study.

All twelve original tensors were regenerated with the original source/inputs
and matched their archived SHA256 exactly. Quality/video records were preserved.
The published window CSV is byte-identical to `caf0b4b`. Final tensors now remain
local for lossless re-decoding and subsequent metrics. Current exporters retain
them by default and protect archived cases from another source version.

Twenty-four component tests passed after the fixes. New schema-2 KV diagnostics
measure historical tokens immediately before attention, excluding current BF16
tokens for both policies. Post-storage and historical injection values are
separate fields. These diagnostics were not retroactively substituted into the
original pilot measurements.

Evidence: [validity_followup.json](validity_followup.json),
[latent_restoration.json](latent_restoration.json), and the generated
[SUMMARY.md](SUMMARY.md). Original scope limitations still apply.
