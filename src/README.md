# Self-Forcing KV quantization reconstruction

A standalone inference subset for reconstructing and validating historical KV
cache quantization experiments on **Wan2.1-T2V-1.3B / Self-Forcing**. This is an
independent Git repository. Source attribution and pinned upstream revisions are
in [vendor/UPSTREAM.md](vendor/UPSTREAM.md); original licenses and [NOTICE](NOTICE)
are preserved. The standalone import history is preserved locally; see [vendor/IMPORT.md](vendor/IMPORT.md) for the `src/` relocation and comparison paths.

This implementation simulates quantization with byte codes and BF16 attention.
It does not provide packed low-bit kernels or claim a speedup. Model dimensions
(30 blocks, 12 heads, head dimension 128, 1560 tokens/frame) follow the upstream
1.3B inference path. Training, distributed inference and image-to-video are
excluded. The historical experimental archive remains in `../Quantization`.

## Setup

The current machine uses Python 3.10 in `~/sfq_data/env`. Invoke it through
`~/sfq_data/py`, which sets `PYTHONNOUSERSITE=1` to exclude unrelated packages in
`~/.local`. Torch 2.6.0+cu124 and flash-attn 2.7.4.post1 match the historical
inference environment. See [docs/RECONSTRUCTION_PROTOCOL.md](docs/RECONSTRUCTION_PROTOCOL.md) for the
reconstruction specification and historical environment differences.

For another machine, create an isolated Python 3.10 environment and install:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Install the matching flash-attn 2.7.4.post1 wheel to reproduce FA2-based results.
The original module falls back to PyTorch SDPA when flash-attn is absent, but
that fallback is not assumed to be bit-exact with FA2.

Download weights separately (about 23 GB, never tracked by Git):

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$HOME/sfq_data/ckpt/wan_models/Wan2.1-T2V-1.3B"
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir "$HOME/sfq_data/ckpt"
ln -s "$HOME/sfq_data/ckpt/wan_models" wan_models
ln -s "$HOME/sfq_data/ckpt/checkpoints" checkpoints
```

DMD checkpoint MD5: `ef41ae42bfb87abe43d5f640fb634d0c`.

## Run

```bash
~/sfq_data/py inference.py \
  --config_path configs/a1.yaml \
  --checkpoint_path checkpoints/self_forcing_dmd.pt --use_ema \
  --data_path prompts.txt --output_folder results/example/videos \
  --num_output_frames 63 --seed 0 --rng chunk \
  --kv_method rtn --kv_bits 4 --storage rolling \
  --latent_dump_root results/example/latents \
  --kv_error_log results/example/raw_errors.json \
  --stats_path results/example/stats.json --save_with_index
```

Omit `--kv_method` for the upstream BF16 cache path. `--rng release` (the default)
uses the original global noise stream. `--rng chunk` uses independent initial
noise and re-noising generators per chunk. `--drain_rng` consumes 7777 global
RNG draws per chunk to test stream isolation. The free-VRAM threshold and
DynamicSwap behavior are retained from the release.

`configs/a1.yaml` uses a 21-frame rolling window without a sink; `a2.yaml` pins
three sink frames. The upstream default config has no rolling eviction and is
only valid up to its 21-frame buffer capacity.

Storage policies:

- `rolling`: restore the full buffer, roll/insert as upstream, attend with the
  current BF16 keys/values, then quantize the full buffer after every call.
- `chunk`: keep the current chunk in BF16, quantize completed chunks once, evict
  complete segments and preserve aligned sink segments. Window and sink lengths
  must be multiples of the fixed chunk length.

RTN and KIVI-style keys/values accept separate widths (`--key_bits`,
`--value_bits`, 2..8). KIVI stores group minima instead of clamped zero points.
KIVI defaults to uint8 codes, including at 8 bits. `--legacy_unsigned_int8`
explicitly enables historical signed-int8 wraparound for reference comparisons.

`--residual_tokens N` preserves the newest live tokens in the rolling buffer,
including before the cache fills. This deliberately fixes the historical
physical-buffer-tail bug. The rolling implementation retains full quantized
codes plus a BF16 residual override; do not interpret this as a compact KIVI
layout. In chunk mode, completed chunks overlapping the residual window remain
BF16 until the entire chunk leaves it, so the BF16 window can exceed N tokens.

Statistics distinguish actual tensor storage from hypothetical packed bytes.
Transient reconstruction buffers and model weights are not included in the
cache-only storage count. CUDA peak memory includes reconstruction and diagnostic
buffers; it is not evidence of packed-cache memory savings. The raw
rolling KV error measures re-quantization injection against the current buffer,
not cumulative error against an unquantized trajectory. It is retained only
for historical validation and must not be used to infer propagation.

With `--kv_error_log`, schema 2 uses `k_rel/v_rel` (also named
`read_history_k_rel/read_history_v_rel`) for **pre-attention history only**.
Both policies exclude the current BF16 chunk and use insertion-time BF16
originals for exactly the historical tokens being read. `read_history_tokens`
and `kv_error_phase` record the mask and timing.

`source_k_rel/source_v_rel` remain explicitly post-storage diagnostics over the
whole active cache, including current tokens; they are not the read-time metric.
Rolling's old injection values live in `legacy_post_k_rel/legacy_post_v_rel`.
The historical export script selects those aliases or accepts old raw logs;
it never silently interprets schema-2 read errors as historical injection.
Rolling source diagnostics retain extra BF16 originals, reported separately as
`diagnostic_bf16_bytes`.

Rolling quantizes unused zeros too. Since 4680 is not divisible by 16, boundary
groups mix live values and zeros. This intentional compatibility behavior differs
from chunk-local grouping.

Live CLI: `~/sfq_data/py scripts/progress.py --watch`. Generated gate progress
is `~/sfq_data/verify/progress.md`; curated VERIFY.md is preserved.

## Verification

[docs/VERIFY.md](docs/VERIFY.md) records measured gates, failures and pending
work. A passed small attention test does not imply a passed full-model gate.

Unit/reference checks (reference source stays outside this repository):

```bash
KVQ_REF_REPO="$HOME/sfq_data/kv-reference" \
SF_REF_REPO="$HOME/sfq_data/sf-pristine" \
LEGACY_REF_REPO="$HOME/sfq_data/sf-legacy" \
  ~/sfq_data/py -m unittest discover -s tests -v

TEST_DEVICE=cuda KVQ_REF_REPO="$HOME/sfq_data/kv-reference" \
  ~/sfq_data/py -m unittest discover -s tests -p test_quant.py -v
```

The optional reference is pinned kv-quant-longhorizon `b4c0936` with historical
patch 17 applied **alone**. Do not copy its unlicensed source into this repo.
The legacy Self-Forcing checkout is pinned `33593df` plus cumulative patches
01, 02 and 16; do not apply 03, 09, 15 and 16 sequentially.

Prepare/check the pinned local reference copies using
`~/sfq_data/py scripts/prepare_references.py` (add `--check` for a read-only check).
This requires the original upstream clones under `~/sfq_data/upstream` and the
historical patches in `~/Quantization`; it does not fetch or publish code.

Once downloads report `DL_DONE` and the DMD hash is verified:

```bash
~/sfq_data/py scripts/verify.py --gates G1 G2 G3 G4 G5 CHUNK
```

Logs, commands, captured latents and resumable completion records live in
`~/sfq_data/verify`. Completed runs are reused only for matching source and
command fingerprints. A failed gate stops the queue for diagnosis; its fixed
criterion is not automatically relaxed. `scripts/reference_run.py` captures
reference outputs without changing sampling operations.


## P3 TF/FR pilot

The completed [126-frame pilot](experiments/p3_pilot_results/RESULTS.md) covers
two newly authored prompts, seed 0, and RTN INT4/INT2. Its protocol was committed
before measurements. Each case retains source hashes, window summaries and
quality diagnostics. This is preliminary evidence, not a multi-seed study.

With verified `g3_base` captures already available under `~/sfq_data/verify`:

```bash
~/sfq_data/py scripts/p3_tf_fr.py --mode gate --out "$HOME/sfq_data/p3_followup"
~/sfq_data/py scripts/run_p3_pilot.py --out "$HOME/sfq_data/p3_followup"
~/sfq_data/py scripts/summarize_p3.py --data "$HOME/sfq_data/p3_followup"
```

Review follow-up validation and identity results are in `docs/VERIFY.md`.


The missing actual-anchor identity validation is recorded in
[the follow-up](experiments/P3_VALIDITY_FOLLOWUP.md). Final `bf16/tf/fr.pt`
files are retained by default. Old cases are protected from a different source
version; use an explicit versioned output directory for new experiments.
Generated summaries are `SUMMARY.md`; curated `RESULTS.md` is preserved.
