# Implementation continuation

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
