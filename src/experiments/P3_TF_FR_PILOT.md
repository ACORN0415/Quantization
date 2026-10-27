# P3 teacher-forced / free-running pilot

Registered 2026-10-27, before inspecting pilot outputs. User explicitly authorized
new experiments after reconstruction gates G1–G5. This is a feasibility pilot,
not the proposed multi-seed/20-prompt journal study.

## Fixed scope

- Self-Forcing / Wan2.1 1.3B, checkpoint and environment from VERIFY.md.
- Two newly authored prompts in `p3_pilot_prompts.txt`, seed 0.
- 126 latent frames (42 chunks), 21-frame window, no sink, 3 frames/chunk.
- P3 `ChunkStore`, group 16, RTN K4V4 and K2V2, no bias correction, residual 0.
- BF16 reference, teacher-forced quantized recomputation and free-running
  quantized trajectory use identical per-chunk initial/re-noising draws.
- Reference caches advance only through BF16 calls. Before each chunk, TF
  receives an independent chunk-segment reconstruction of the pre-chunk BF16
  cache. Only its current chunk evolves during that chunk's denoising/refresh.
- Both compressed trajectories use the same attention implementation as BF16;
  no packed kernel or timing advantage is claimed.

## Gates before inference measurements

1. Unit assertions: anchoring covers correct pre-eviction chunks, honors sink
   retention, contains no references to mutable BF16 buffers, and never changes
   completed quantized segments during subsequent calls.
2. No-quantization TF is bit-exact with its lockstep BF16 reference.
3. The independent chunk loop's BF16 output is bit-exact against existing
   verified `g3_base` 63-frame captures, for both historical test prompts.
4. Do not start measured pilot cases until these gates pass. Do not relax gates
   after inspecting outputs. Save source/prompt hashes and commands.

## Measurements and interpretation

Per chunk: relative latent L2 TF-vs-BF16, FR-vs-BF16, FR-vs-TF, and scalar
FR-vs-BF16 minus TF-vs-BF16. The last is a descriptive proxy, not an additive
causal error decomposition. Report early/middle/late averages (chunks 1–6,
7–20, 21–41), excluding chunk 0 from summaries.

Decode the three complete trajectories. Compute MUSIQ at each chunk's middle
pixel frame and an adjacent-frame absolute-difference motion proxy. Preserve
prompt-level numbers and videos. MUSIQ and motion alone do not establish
identity preservation or perceptual equivalence. No significance test, success
threshold, equivalence or generalization claim from n=2. No claim that the
latent-distance difference alone proves harmful propagation or separates chaos.

The larger proposed study (multiple seeds, >=20 prompts, subject consistency,
BF16 perturbation controls and another model) remains follow-up work; no
publication claim follows from this pilot.

## Storage and commits

Per-case summaries and source hashes are durable. Temporary TF caches are
released each chunk. Save only full trajectories for decoding, then remove
those tensors after successful video/metric export, retaining their SHA256 and
byte counts in a cleanup manifest. Keep failed-case evidence until diagnosed.
Commit protocol, implementation/gates, and measured results separately.
