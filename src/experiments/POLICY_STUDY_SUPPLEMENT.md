# Post-hoc subject and temporal diagnostics

This is an exploratory supplement decided after the registered MUSIQ results
were available. It does not amend or replace the frozen study or its primary
endpoint. No new generation, prompt selection, seed replacement or parameter
tuning on these outcomes is permitted.

## Inputs and fixed analysis

Evaluate exactly 260 existing MP4 files: 20 shared BF16 references and TF, FR,
noise for all 80 policy cases. Discover inputs through the registered plan and
complete records, never arbitrary globbing (which also finds development videos).
Require 501 frames, 832×480, 16 fps; hash every input. These scores use lossy
encoded video, whereas the original MUSIQ used raw decoded frames. They are
separate measurements. Retain latents; no inference environment packages change.

Subject feature consistency follows the VBench DINO definition: DINO ViT-B/16,
short-side resize 224 without antialiasing, RGB /255, ImageNet normalization;
for every frame after the first average nonnegative cosine similarity with
the previous and the first video frame. No center crop. Batch inference in
FP32. This is a whole-frame appearance proxy, not tracked-object identity,
semantic correctness or a human judgement of subject preservation.

Temporal diagnostics use torchvision RAFT-small C_T_V2 weights at 256×448,
bilinear resize, normalization to [-1,1], 12 updates. Sample the adjacent
pairs (m−1,m) and (m,m+1) at each registered chunk midpoint m=12c+3.
Warp frame t+1 to t with forward flow. Include only in-bounds pixels satisfying
forward/backward consistency ||f+warp(b)||² <= .01 (||f||²+||warp(b)||²)+.5.
Report mean RGB absolute residual /255 on eligible pixels, eligible fraction,
mean flow magnitude and unwarped frame difference. Residual lower is better
only conditional on motion/coverage; static or collapsed videos can score well.
No claims of official VBench motion-smoothness or temporal-flickering scores.
VBench temporal-flickering assumes static video and is not suitable as a
standalone quality measure here. Empty masks, missing files, wrong dimensions,
nonfinite values or source/input hash changes halt rather than dropping rows.

For each chunk c, pixel range is [0,9) for c=0 and [12c−3,12c+9) otherwise.
Assign subject transitions by destination frame, keeping the first-frame anchor
fixed throughout the video. Subject excludes frame zero. Report all 42 chunks,
and early 1–6, middle 7–20, late 21–41 plus whole video. Temporal pairs stay
inside their specified chunk. Subject per-video windows average transitions;
temporal windows average pair scores. Keep pair/chunk values for auditing.

Compare P3−P1 within each prompt/seed/bit/arm, average seeds within prompt,
then report mean over ten prompts and exploratory 95% percentile prompt-cluster
bootstrap intervals (20,000 resamples, RNG seed 872341). No confirmatory
p-values, multiplicity-adjusted discovery or new generalization claim. FR is
the central policy contrast; TF/noise and BF16-relative scores are descriptive.

## Gates and provenance

Before the batch, test warp direction with translated synthetic frames, exact
identity and empty-mask handling; verify identical feature vectors score one.
Smoke-test a complete old development video, finite metrics and GPU cost.
Resume only artifacts matching evaluator, protocol, model and video hashes.
Write progress after each video, atomic JSON and a detached completion pipeline
through final analysis, Korean integrated report, commit and GitHub push.

Model sources: [DINO](https://github.com/facebookresearch/dino), commit
`7c446df5b9f45747937fb0d72314eb9f7b66930a` (Apache-2.0), downloaded outside Git;
[VBench subject formula and transforms](https://github.com/Vchitect/VBench/tree/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench)
(Apache-2.0), independently implemented here with OpenCV video decode;
torchvision 0.21.0 [RAFT](https://github.com/pytorch/vision/tree/v0.21.0/torchvision/models/optical_flow)
(BSD-3-Clause). Record checkpoint SHA256 and installed library versions with
the results. We do not redistribute model code or weights.
