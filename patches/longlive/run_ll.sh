#!/usr/bin/env bash
# Step H: the same quantization grid as Self-Forcing, on a model with an
# explicit attention sink (3 latent frames) and a 9-frame rolling window.
# Eviction starts at chunk 4 here, versus chunk 7 in Self-Forcing.
set -u
PY=~/gpu/sf-venv/bin/python
run () { tag="$1"; shift; echo "=== $tag ==="
  mkdir -p "results/ll_${tag}"
  $PY inference.py --config_path configs/ll_quant63.yaml --seed 0 \
    --latent_dump_root "results/latents_ll/${tag}" \
    --kv_err_log "results/ll_${tag}/kv_err.json" \
    --output_folder "results/ll_${tag}/videos" \
    "$@" > "results/ll_${tag}/run.log" 2>&1
  echo "$tag EXIT=$?"; grep -E "\[kv_quant\]|\[fakequant\]" "results/ll_${tag}/run.log" | head -2; }

run ctrl1em3 --noise_perturb 1e-3
run kv_int4  --kv_quant RTN --kv_bits 4
run kv_int2  --kv_quant RTN --kv_bits 2
run w4a4     --fakequant w4a4
# The session-2 pseudo-sink result predicts this removes most of the damage.
run sink_bf16_int2 --kv_quant RTN --kv_bits 2 --kv_sink_bf16
echo "ALL_DONE"
