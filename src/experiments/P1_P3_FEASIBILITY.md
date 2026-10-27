# P1/P3 engineering feasibility checks

This is a development protocol, not a preregistration or a quality study.
Use only the two existing development prompts, seed 0, RTN4, 63 latent
frames, 21-frame context, group size 16. Do not select study endpoints or
held-out prompts using these results. Preserve prior pilot outputs.

## Checks before a new study

1. Replay every BF16 reference K/V write (four denoising calls plus context
   refresh per chunk) through a persistent P1 RollingStore. Clone it before
   advancing each reference chunk; TF writes affect only the working clone.
   Verify identity replay against BF16 and archived g3_base outputs, including
   eviction. Verify P3 replay against its existing anchor construction.
2. Measure RTN4 P1 and P3 TF/FR sampling time and peak allocated/reserved CUDA
   memory in separate policy passes. Do not treat these peaks as packed
   inference memory savings. Keep final tensors, not temporary cache copies.
3. In a separate diagnostic pass verify output tensor equality with the pass
   without logging. TF historical-original errors can use aligned reference
   cache slices; FR diagnostics retain their own insertion-time originals.
4. Explore a storage-event perturbation control: for each K/V encode event,
   match the FP32 norm of Q(x)-x with an isotropic random direction from an
   independent RNG, then cast to BF16. Report actual rounded error separately.
   This matches *incremental storage error*, not accumulated read-history
   error; do not label it a read-error-matched or chaos-baseline control.
   Confirm zero-error identity, RNG isolation, and timing/memory feasibility.

Replay matches the storage schedule, not input-dependent realized errors.
FR and TF distances are descriptive and not an additive causal decomposition.
P3 replay/anchor equivalence is only claimed for the tested deterministic,
segment-local quantizer and current residual/sink configuration.

## Visibility

Run from `src`: `~/sfq_data/py scripts/progress.py --watch`.
Raw log: `tail -f ~/sfq_data/feasibility.log`.
Machine-readable phases: `~/sfq_data/feasibility/status.json`.
Stage summaries: `~/sfq_data/WORK_STATUS.md`.

Commit the development protocol and code before model execution. Publish
feasibility evidence separately from historical pilot results. Fix the final
control definition, sample, primary endpoint and analysis before the main study.
