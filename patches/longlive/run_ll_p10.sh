#!/usr/bin/env bash
# Session 3 section 6: re-run the LongLive configurations on the 10-prompt set so
# the cross-model table comes from one prompt sample rather than two.
set -u
PY=~/gpu/sf-venv/bin/python
run () { tag="$1"; shift; echo "=== $tag ==="
  mkdir -p "results/llp10_${tag}"
  $PY inference.py --config_path configs/ll_quant63_p10.yaml --seed 0 \
    --latent_dump_root "results/latents_llp10/${tag}" \
    --kv_err_log "results/llp10_${tag}/kv_err.json" \
    --output_folder "results/llp10_${tag}/videos" \
    "$@" > "results/llp10_${tag}/run.log" 2>&1
  echo "$tag EXIT=$?"; }
run bf16
run ctrl1em3 --noise_perturb 1e-3
run kv_int4  --kv_quant RTN --kv_bits 4
run kv_int2  --kv_quant RTN --kv_bits 2
echo "ALL_DONE"
