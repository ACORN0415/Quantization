#!/usr/bin/env bash
set -u
PY=~/gpu/sf-venv/bin/python
# FR (free-running) with the fixed asymmetric quantizer
for spec in kivi4:4 kivi2:2; do
  tag="kivi_fix_${spec%%:*}"; bits="${spec##*:}"
  echo "=== FR $tag (KIVI INT$bits) ==="
  mkdir -p "results/kivi_fix/${tag}/videos"
  $PY inference.py --config_path configs/self_forcing_dmd_long.yaml \
    --output_folder "results/kivi_fix/${tag}/videos" \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/quant3/prompts3.txt \
    --latent_dump_root "results/latents63/${tag}" \
    --stats_path "results/kivi_fix/${tag}/stats.json" \
    --kv_err_log "results/kivi_fix/${tag}/kv_err.json" \
    --num_output_frames 63 --kv_quant KIVI --kv_bits "$bits" \
    --save_with_index --seed 0 --use_ema > "results/kivi_fix/${tag}/run.log" 2>&1
  echo "$tag EXIT=$?"
done
# TF (teacher-forced) for the same two
for spec in kivi4:4 kivi2:2; do
  tag="kivi_fix_${spec%%:*}"; bits="${spec##*:}"
  echo "=== TF $tag ==="
  mkdir -p "results/tf_${tag}"
  $PY teacher_forced.py --tag "$tag" --use_ema --num_output_frames 63 --max_prompts 3 \
    --kv_quant KIVI --kv_bits "$bits" \
    --out_dir "results/tf_${tag}" --latent_dump_root "results/latents63_tf/${tag}" \
    > "results/tf_${tag}/run.log" 2>&1
  echo "TF $tag EXIT=$?"
done
echo "ALL_DONE"
