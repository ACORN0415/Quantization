#!/usr/bin/env bash
set -u
PY=~/gpu/sf-venv/bin/python
COMMON="--config_path configs/self_forcing_dmd_long.yaml \
 --checkpoint_path checkpoints/self_forcing_dmd.pt \
 --data_path prompts/quant3/prompts3.txt \
 --num_output_frames 63 --save_with_index --seed 0 --use_ema"

# Control at a perturbation size bf16 can actually represent (~0.4% resolution).
# If this matches the 1e-3 curve, divergence is perturbation-size independent.
echo "=== ctrl_perturb1em2 ==="
mkdir -p results/ctrl_perturb1em2/videos
$PY inference.py $COMMON \
  --output_folder results/ctrl_perturb1em2/videos \
  --latent_dump_root results/latents63/ctrl_perturb1em2 \
  --stats_path results/ctrl_perturb1em2/stats.json \
  --noise_perturb 1e-2 --no_decode \
  > results/ctrl_perturb1em2/run.log 2>&1
echo "ctrl_perturb1em2 EXIT=$?"

# In-situ KV cache quantization error (session plan section 6, optional item).
for spec in kverr_int4:RTN:4 kverr_int2:RTN:2 kverr_kivi4:KIVI:4; do
  tag="${spec%%:*}"; rest="${spec#*:}"; m="${rest%%:*}"; b="${rest##*:}"
  echo "=== $tag ==="
  mkdir -p "results/${tag}"
  $PY inference.py $COMMON \
    --output_folder "results/${tag}/videos" \
    --stats_path "results/${tag}/stats.json" \
    --kv_quant "$m" --kv_bits "$b" \
    --kv_err_log "results/${tag}/kv_err.json" --no_decode \
    > "results/${tag}/run.log" 2>&1
  echo "$tag EXIT=$?"
done
echo "ALL_DONE"
