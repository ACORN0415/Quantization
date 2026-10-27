# Implementation continuation

Updated: 2026-10-27 (Asia/Seoul).

## Active scope

Read `docs/HANDOFF_codex.md` first. User authorized continued implementation and
verification without repeated permission requests. Scope ends at reconstruction
and gates G1–G5; no TF/FR experiments, LongLive port or GitHub push.

## Completed

- Pristine import `da5438c` preserved, no upstream remote attached.
- T1 inference imports cleaned; isolated CLI `--help` passes.
- Quantizer, rolling/chunk stores, prompt resets, chunk/release RNG, captures,
  raw historical errors and storage accounting implemented.
- README, NOTICE and source modification notices added.
- G1 CPU/CUDA reference comparisons PASS; G1b synthetic tests PASS.
- Actual attention compared against pristine and patched legacy source on
  small CPU inputs: PASS, including rolling eviction and sinks.
- All 11 tests PASS; full-model G2–G5 still required.

## Local paths

- Implementation: `/home/acorn0415/sf-kvquant`.
- Historical archive: `/home/acorn0415/Quantization`.
- Isolated Python: `/home/acorn0415/sfq_data/py`.
- Reference checkouts outside this repository: `sfq_data/sf-pristine`,
  `sfq_data/sf-legacy` (01+02+16), `sfq_data/kv-reference` (patch 17 alone).
- Large weights: `sfq_data/ckpt`; untracked symlinks in implementation tree.
- Download initially stalled. Old process terminated after SIGINT had no effect;
  retry uses HTTP (`HF_HUB_DISABLE_XET=1`), log `sfq_data/download_resume.log`.
  Keep old partial until the replacement finishes. Check `DL_DONE` and MD5.
- Unit results: `sfq_data/{unit,unit_cuda,all_tests,integration}.log`.

## Next action

1. Check download completion. Do not start full-model gates before `DL_DONE`.
2. Run `~/sfq_data/py scripts/verify.py --gates G2 G3 G4 G5 CHUNK`.
3. Monitor `~/sfq_data/verify/status.json`, per-run logs and `docs/VERIFY.md`.
4. Diagnose any failure before changing criteria; scripts stop on failure and
   fingerprint completed runs for reuse.
5. Update this record, commit measured validation results, report incomplete gates
   explicitly if any remain.

Known compatibility detail: historical KIVI uses signed int8 even for 8-bit
unsigned codes. Default reproduces it; `--unsigned_codes` selects correct uint8.
Residual windows deliberately correct the old physical-tail bug. Rolling
residual storage is an override simulation, not a compact packed layout.
