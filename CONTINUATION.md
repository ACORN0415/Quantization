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
- 11 component tests plus 2 storage-accounting tests PASS.
- Full-model G2 PASS: all 3 release-mode final latents bit-exact.
- G3/G4/G5/CHUNK queue is running; consult live status before restarting.

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

1. Downloads completed (`DL_DONE`); DMD MD5 and byte count matched.
2. Monitor the already running verification queue; do not launch a second GPU queue.
   After failure/exit, resume only necessary gates with `scripts/verify.py`.
3. Monitor `~/sfq_data/verify/status.json`, per-run logs and `docs/VERIFY.md`.
4. Diagnose any failure before changing criteria; scripts stop on failure and
   fingerprint completed runs for reuse.
5. Update this record, commit measured validation results, report incomplete gates
   explicitly if any remain.

Known compatibility detail: historical KIVI uses signed int8 even for 8-bit
unsigned codes. Default reproduces it; `--unsigned_codes` selects correct uint8.
Residual windows deliberately correct the old physical-tail bug. Rolling
residual storage is an override simulation, not a compact packed layout.
