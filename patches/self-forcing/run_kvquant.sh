#!/usr/bin/env bash
# Step C: UCSD KV-cache quantization on the 63-frame rollout.
set -u
PY=~/gpu/sf-venv/bin/python
# tag:method:bits
RUNS="kv_int4:RTN:4 kv_int2:RTN:2 kv_kivi4:KIVI:4"
for spec in $RUNS; do
  tag="${spec%%:*}"; rest="${spec#*:}"; method="${rest%%:*}"; bits="${rest##*:}"
  echo "=== $tag ($method INT$bits) ==="
  mkdir -p "results/${tag}/videos"
  $PY inference.py \
    --config_path configs/self_forcing_dmd_long.yaml \
    --output_folder "results/${tag}/videos" \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/quant3/prompts3.txt \
    --latent_dump_root "results/latents63/${tag}" \
    --stats_path "results/${tag}/stats.json" \
    --num_output_frames 63 \
    --kv_quant "$method" --kv_bits "$bits" \
    --save_with_index --seed 0 --use_ema \
    > "results/${tag}/run.log" 2>&1
  echo "$tag EXIT=$?"
  grep -E "\[kv_quant\]|\[stats\]" "results/${tag}/run.log" | head -5
done
echo "ALL_DONE"
