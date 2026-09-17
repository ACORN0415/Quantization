#!/usr/bin/env bash
# Step D: fake weight/activation quantization sweep on the 63-frame rollout.
set -u
PY=~/gpu/sf-venv/bin/python
CONFIGS="w8a8 w4a16 w4a8 w4a4 kv_only_w4a4 no_kv_w4a4"
for cfg in $CONFIGS; do
  echo "=== $cfg ==="
  mkdir -p "results/fq_${cfg}/videos"
  $PY inference.py \
    --config_path configs/self_forcing_dmd_long.yaml \
    --output_folder "results/fq_${cfg}/videos" \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/quant3/prompts3.txt \
    --latent_dump_root "results/latents63/fq_${cfg}" \
    --stats_path "results/fq_${cfg}/stats.json" \
    --num_output_frames 63 \
    --fakequant "$cfg" \
    --save_with_index --seed 0 --use_ema \
    > "results/fq_${cfg}/run.log" 2>&1
  echo "$cfg EXIT=$?"
  grep -E "\[fakequant\]|\[stats\]" "results/fq_${cfg}/run.log" | head -5
done
echo "ALL_DONE"
