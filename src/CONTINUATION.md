# Active continuation — 2026-10-27

The user explicitly authorized GitHub publication and new experiments after the
original reconstruction handoff was completed. Canonical active code is now
`/home/acorn0415/Quantization/src`, imported with full local Git history.
The old `/home/acorn0415/sf-kvquant` is the preserved reconstruction snapshot.

GitHub destination: `ACORN0415/Quantization`, `main`. Existing VS Code Git
askpass authentication works; `gh` is unnecessary. The shell does not inherit
the IDE bridge automatically. Local helper `~/sfq_data/git_with_ide.py` bridges
the current VS Code socket without storing credentials. Initial publication
through `d64c633` succeeded; subsequent implementation commits are pushed at
work boundaries. No upstream Git remote was attached.

New experiment: fixed pilot in `experiments/P3_TF_FR_PILOT.md` (2 new prompts,
seed 0, 126 frames, RTN INT4/INT2 P3, TF/FR/BF16, no bias/sink). This pilot does
not replace the multi-seed/20-prompt study. Gate/queue logs are
`~/sfq_data/p3_gate_final.log` and `~/sfq_data/p3_queue.log`; durable cases and
progress live in `~/sfq_data/p3_pilot`. Do not launch a duplicate GPU queue.

The initial independent-loop/no-quant TF model gate passed all 42 chunks.
Final source-frozen gate is running before the four measured cases. The serial
queue starts automatically after gate success. Source changes during a case
prevent completion. Keep the model runner and kvq helpers unchanged while the
queue is active. Analyze with `scripts/summarize_p3.py` after all cases complete.

Commit completed work units and push using existing IDE authentication. Remove
only verified derived temporary tensors, retaining videos, metrics, source
hashes and cleanup manifests. Do not delete weights or historical raw evidence.

## Reconstruction reference (completed)


Updated: 2026-10-27 (Asia/Seoul).

## Completed objective

The user authorized implementation and verification per `docs/HANDOFF_codex.md`.
Reconstruction and all required G1–G5 gates are complete. Chunk-policy model
rollout sanity also passed. Read `docs/VERIFY.md` for exact results and limits;
`docs/verification-evidence.json` retains compact evidence and executed commands.

- New implementation is in `/home/acorn0415/sf-kvquant`.
- Pristine import `da5438c` preserved; no git remote attached.
- Quantizers, rolling/chunk storage, residual fix, prompt reset, seeded RNG,
  latent captures, raw error records and separate memory accounting implemented.
- G2: 3/3 original-release final latents bit-exact.
- G3: 42/42 RNG-control chunks bit-exact.
- G4: 168/168 legacy comparison chunks bit-exact (four configurations).
- G5: all 210 historical rows exact; max relative difference 0.0.
- Chunk policy: both 63-frame model runs complete; 40/42 chunks differ from rolling.
- G1/G1b CPU/CUDA/component checks passed. No criterion was relaxed.

## Reproduction locations

- Historical archive: `/home/acorn0415/Quantization` (unchanged).
- Isolated Python: `/home/acorn0415/sfq_data/py` (sets PYTHONNOUSERSITE=1).
- Weights: `sfq_data/ckpt`; downloads complete and DMD hash verified.
- References outside this repo: `sfq_data/sf-pristine`, `sfq_data/sf-legacy`
  (patches 01+02+16), `sfq_data/kv-reference` (patch 17 alone).
- Full logs, raw records, captures and completed-job records: `sfq_data/verify`.
- Local reference setup checker: `scripts/prepare_references.py --check`.
- Resumable validator: `scripts/verify.py`; existing completion fingerprints
  predate later hardening of input hashes, so a future rerun may recompute them.

## Next work boundary

No required work remains in this handoff. Do not start TF/FR, LongLive, fresh
quality sweeps or publication just because the earlier discussion proposed them.
Await the user's next research instruction. G6 MUSIQ was optional and not run.
Do not create or push a GitHub repository until explicitly instructed.

Known limits: 1.3B-specific dimensions; byte-code simulation rather than packed
kernels; default signed-int8 KIVI 8-bit compatibility behavior; historical error
aggregation preserved only for reproduction. See README and VERIFY for details.
