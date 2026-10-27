# P1/P3 development feasibility — 2026-10-28 KST

All engineering checks in the [development protocol](../P1_P3_FEASIBILITY.md) passed. This is not a main-study result or a quality comparison. Implementation: `f39a48f`; source fingerprint and raw artifact manifests: [evidence.json](evidence.json).

- 27 unit/reference regression tests passed before model execution.
- Replay identity: 2 policies × 2 existing development prompts × 21 chunks = **84/84** TF outputs bit-exact with BF16; BF16 also matched archived `g3_base`.
- Every layer replayed all **105 writes** per prompt (5 calls × 21 chunks). TF uses independent pre-chunk snapshots.
- P3 replay and existing anchor segments/pending tensors matched at every tested chunk and layer, including eviction.
- Separate TF and FR diagnostic passes produced tensors bit-exact with the corresponding runs without logging, for both policies. TF history metrics reused aligned BF16 reference slices; FR rolling diagnostics retained insertion-time originals.

## Execution costs

One RTX 4090, 24 GB; 63 latent frames, 21-frame context, RTN4, group 16, seed 0. Quantized measurements use one development prompt. Times exclude model loading, VAE decoding and quality metrics. They include progress I/O, archived-reference checks and, for P3, anchor equivalence checks. They are feasibility costs, not a controlled speed benchmark.

| Pass | P1 seconds | P1 allocated peak GiB | P3 seconds | P3 allocated peak GiB |
|---|---:|---:|---:|---:|
| BF16 + teacher replay + TF + FR | 132.4 | 18.75 | 109.3 | 19.74 |
| Same, with TF history diagnostics | 140.8 | 18.87 | 117.3 | 20.21 |
| FR alone, with insertion-original diagnostics | 74.5 | 12.62 | 36.0 | 6.90 |
| Storage-noise trajectory alone | 78.4 | 9.56 | 34.5 | 8.76 |

Peaks include model, reconstruction buffers and any diagnostics. Codes are byte containers, not packed INT4. These peaks do not establish low-bit inference memory savings. Policies were run separately; simultaneous execution of all four comparison arms was not tested.

## Perturbation control tested

At each actual encode event, compute the norm of `Q(x)-x` on that arm's own input, separately for K and V. Normalize an independent Gaussian random direction to this norm, add it to x, cast to BF16 and retain it according to the policy. This is an isotropic fixed-norm direction before rounding, not an unnormalized Gaussian perturbation. Each layer has a private RNG; unit tests verified global RNG isolation, reproducibility and zero-error identity.

- P1: 6,300 K/V encode events; maximum realized-vs-target norm mismatch after BF16 rounding **0.0487%**.
- P3: 1,200 K/V encode events; maximum realized-vs-target norm mismatch after BF16 rounding **0.0515%**.

This controls incremental storage-error magnitude on the control arm's inputs. It does **not** match the accumulated `read_history_*` error of another trajectory, its per-group structure, or temporal correlation. It is not a small-initial-perturbation chaos baseline. No quality advantage or causal decomposition is inferred here. The final main-study control definition remains to be fixed before measurements.

## Retained artifacts and next steps

Raw status/logs, full historical error records and all final latent tensors remain under `${HOME}/sfq_data/feasibility`; public evidence records relative paths, sizes and SHA256 hashes. Historical pilot outputs were not modified.

The next main-study protocol must fix the control estimand, held-out prompts/seeds, primary quality endpoint, prompt-paired analysis and complete runtime budget. Neither 126-frame end-to-end decoding/quality cost, INT2 feasibility nor packed equal-memory comparisons were measured here. Do not extrapolate these results into an overnight completion guarantee.

Replay matches storage operations; it does not make input-dependent quantization errors identical. FR and TF distances are descriptive, not an additive separation of feedback and storage effects.

Live local status: `bash tools/watch_progress.sh`; one snapshot: `bash tools/watch_progress.sh 0`.
