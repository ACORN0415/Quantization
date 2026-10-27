# Upstream sources

This repository is not a git fork. Code was copied (subset) from the projects
below at the listed commits; the first commit of this repository contains those
files **unmodified**, so `git diff <first-commit> -- <file>` shows every change
made here.

| project | url | commit | license | what was taken |
|---|---|---|---|---|
| Self-Forcing | https://github.com/guandeh17/Self-Forcing | `33593df3e81fa3ec10239271dd2c100facac6de1` | Apache-2.0 (`LICENSE`) | CLI inference path only: `inference.py`, `pipeline/causal_inference.py`, `utils/`, `demo_utils/memory.py`, `wan/modules/`, two configs |
| Wan2.1 (inside Self-Forcing's `wan/`) | https://github.com/Wan-Video/Wan2.1 | via Self-Forcing | Apache-2.0 | `wan/modules/` model, T5, VAE |

Not copied, by design:

- Training, demo, TensorRT, I2V, distributed, 14B configs of Self-Forcing.
- **kv-quant-longhorizon** (https://github.com/suraj-ranganath/kv-quant-longhorizon,
  `b4c0936`) ships no license, so none of its code is redistributed here. The
  quantizers in `kvq/` are an independent reimplementation of the published
  RTN / KIVI-style schemes; `tests/` only compares against that repository when
  a local checkout is supplied (`KVQ_REF_REPO`).

Model weights are not part of this repository (see README).
