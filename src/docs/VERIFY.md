# Verification results

Verified 2026-10-27 (Asia/Seoul). Fixed criteria: [RECONSTRUCTION_PROTOCOL.md](RECONSTRUCTION_PROTOCOL.md).
All required gates passed; no acceptance threshold was relaxed.

| Gate | Result | Measured evidence |
|---|---|---|
| G1 | PASS | CPU and CUDA restored tensors exactly match the patched local reference for RTN/KIVI × bits 2,3,4,8 × lengths 32,4680,32760 × random/constant/positive/zero inputs. Independent key/value widths also match. |
| G1b | PASS | Synthetic eviction, sink retention, prompt reset, immutable chunk segments, and live-prefix residual assertions passed. Small actual attention tests match pristine and legacy implementations. |
| G2 | PASS | Release RNG, seed 0, 21 frames, 3 prompts: **3/3 final latents bit-exact** against pristine Self-Forcing. |
| G3 | PASS | Chunk RNG, seed 0, 63 frames, 2 prompts: **42/42 chunks bit-exact** with/without 7777 additional global CUDA draws per chunk. |
| G4 | PASS | BF16 / RTN INT4 / RTN INT2 / KIVI INT4 (residual 0), 2 prompts each, 63 frames: **168/168 chunks bit-exact** against legacy source. |
| G5 | PASS | KIVI INT4 residual 0, seed 0, 10 historical prompts: **210/210 rows match**, including counts and active tokens. Maximum relative difference across K/V means: **0.0** (criterion ≤1e-6). |
| Chunk rollout sanity | PASS | 2 full 63-frame model runs completed; **40/42 chunks differ from rolling**. This verifies a distinct operational path; it is not evidence of quality improvement. |
| G6 MUSIQ (optional) | Not run | Outside the required reconstruction gates; no new perceptual-quality claim. |

The component suite comprises 13 tests: 11 in `all_tests.log` and 2 additional
storage-accounting tests in `memory.log`; all passed. CUDA quantizer comparison
suite: 3 tests, PASS. Component assertions do not replace the full-model gates.

## Historical reproduction details

First G5 row (prompt 0, chunk 0):

- key relative error: `0.037011147709563375`
- value relative error: `0.14003970734775067`
- records: `120`; active tokens: `4680`

G5 deliberately reproduces the historical aggregation boundary: the previous
chunk's refresh records enter the following chunk's group, including across
prompt boundaries. Correct prompt/chunk/phase/layer labels are retained in raw
records; this old aggregation is only for compatibility. The metric measures
re-quantization injection against the current buffer, not propagation against
an independent BF16 trajectory.

The DMD checkpoint matched MD5 `ef41ae42bfb87abe43d5f640fb634d0c` and
5,676,252,553 bytes. Base-model/tokenizer SHA256 hashes, reference revision/diff
hashes, exact executed commands, comparison rows, unit logs and source hashes
are preserved in [verification-evidence.json](verification-evidence.json).
The implementation commit list is [IMPLEMENTATION_COMMITS.txt](IMPLEMENTATION_COMMITS.txt).

Large latent captures, raw records and full runtime logs stay outside Git at
`${DATA_ROOT}/verify`. The compact evidence above is tracked.

## Scope and limits

- G4 establishes compatibility of the **rolling simulation** with reconstructed
  legacy source in this environment and these configurations. Chunk storage has
  unit tests and model-rollout sanity checks, with no historical ground truth.
- Codes reside in byte tensors. Hypothetical packed bytes are distinct from
  actual tensor storage; this is not a low-bit kernel performance benchmark.
- KIVI now defaults to uint8. Reference tests explicitly enable historical
  signed-int8 behavior. Original G1–G5 measurements are preserved as historical evidence.
- Residual windows intentionally fix the old physical-tail bug. Rolling mode
  retains a BF16 override alongside full quantized codes; chunk mode keeps
  whole recent chunks in BF16 and may exceed the requested token count.
- Transformers/tokenizers differ from the archived environment (current
  5.20.2/0.23.3 versus archived 5.17.0/0.23.2). G5 nevertheless reproduced every
  historical numerical row exactly here. Full current versions are in
  [environment-frozen.txt](environment-frozen.txt).
- LongLive, TF/FR experiments, new quality measurements and GitHub publication
  were excluded by the handoff. No upstream remote is attached to this repo.


## Reconstruction review follow-up (2026-10-27)

- All 20 current CPU component tests passed (41.503 s).
- CUDA quantizer/reference suite passed, with signed historical compatibility
  explicitly enabled only inside the reference comparison.
- Full-model residual identity: rolling **42/42**, chunk **42/42**, both bit-exact
  with the fixed previously verified BF16 baseline (two prompts, 63 frames).
- Large-residual attention unit checks cover both policies, sink 0/1, ten frames,
  and five calls per frame. These replace output-difference-only validation as
  the identity check; output difference remains a separate sanity observation.
- `source_k_rel/source_v_rel` share an insertion-time-original definition across
  policies. Historical `k_rel/v_rel` scopes remain labeled and are not pooled.
- `verify.py` now writes generated progress outside curated documentation.
- Published source/evidence contain no machine-specific account home path.
  Internal handoff/session notes are archived locally. Prior Git history is
  preserved; no history rewrite was performed.

New evidence: [review-evidence.json](review-evidence.json). The original G1–G5
measurements above remain archived historical measurements; this review did not
repeat the entire ten-prompt G5 GPU run, whose raw records were independently
confirmed by the reviewer. New defaults do not change RTN or KIVI 2/4-bit math.


## Actual anchoring-path validity check (post-pilot)

The previous no-quantization teacher gate used `plain_snapshot`, which did not
validate the quantized teacher's `anchor_store` path. A separate retrospective
gate now tests the original `caf0b4b` implementation using an identity quantizer
through `compressed_cache(reference=...)`, hence `anchor_store` and `ChunkStore`.
All **42/42 teacher outputs and reference comparisons passed bit-exactly**.
The encoded-segment path was exercised (3,570 encode-pair calls per prompt).
This is a validity check performed after pilot results, not a changed analysis.

New schema-2 KV diagnostics also align the time and historical-token mask.
The legacy post-write injection aliases remain exact against the legacy source
in integration tests; they are separate from the common read-time diagnostics.
