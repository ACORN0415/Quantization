# P3 TF/FR pilot results

Fixed protocol: [P3_TF_FR_PILOT.md](../P3_TF_FR_PILOT.md).
Two new prompts, seed 0, 126 latent frames, RTN P3 group 16, no sink/bias.
All source-bound no-quantization and reference gates passed before measurement.

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

Successful exports removed **704,489,580 bytes** of derived latent tensors,
including the redundant no-quantization gate tensors. Videos and compact
measurements remain preserved. No memory-saving claim follows from this cleanup.
