#!/usr/bin/env bash
set -u
PY=~/gpu/sf-venv/bin/python
COMMON="--config_path configs/self_forcing_dmd_long.yaml \
 --checkpoint_path checkpoints/self_forcing_dmd.pt \
 --data_path prompts/quant3/prompts3.txt \
 --num_output_frames 63 --save_with_index --seed 0 --use_ema"

run () {  # tag, extra args...
  tag="$1"; shift
  echo "=== $tag ==="
  mkdir -p "results/${tag}/videos"
  $PY inference.py $COMMON \
    --output_folder "results/${tag}/videos" \
    --latent_dump_root "results/latents63/${tag}" \
    --stats_path "results/${tag}/stats.json" \
    "$@" > "results/${tag}/run.log" 2>&1
  echo "$tag EXIT=$?"
  grep -E "\[kv_quant\]|\[fakequant\]" "results/${tag}/run.log" | head -2
}

# Step C: UCSD KV cache quantization
run kv_int4  --kv_quant RTN  --kv_bits 4
run kv_int2  --kv_quant RTN  --kv_bits 2
run kv_kivi4 --kv_quant KIVI --kv_bits 4

# Control: how fast does BF16 diverge from an infinitesimal nudge?
run ctrl_perturb1em3 --noise_perturb 1e-3
run ctrl_perturb1em5 --noise_perturb 1e-5

# Milder operating point so the early curve stays below the divergence ceiling
run fq_w8a16 --fakequant w8a16

echo "ALL_DONE"
