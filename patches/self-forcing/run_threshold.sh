#!/usr/bin/env bash
# Step I: where between INT4 (2.8% cache error) and INT2 (13%) does the rollout
# break? Two knobs: bit-width, and block size (larger blocks = coarser scale =
# more error at the same bit-width).
set -u
PY=~/gpu/sf-venv/bin/python
C="--config_path configs/self_forcing_dmd_long.yaml \
 --checkpoint_path checkpoints/self_forcing_dmd.pt \
 --data_path prompts/quant3/prompts3.txt \
 --num_output_frames 63 --save_with_index --seed 0 --use_ema"

run () {  # tag, extra args...
  tag="$1"; shift
  echo "=== $tag ==="
  mkdir -p "results/thr_${tag}/videos"
  $PY inference.py $C \
    --output_folder "results/thr_${tag}/videos" \
    --latent_dump_root "results/latents63/thr_${tag}" \
    --stats_path "results/thr_${tag}/stats.json" \
    --kv_err_log "results/thr_${tag}/kv_err.json" \
    "$@" > "results/thr_${tag}/run.log" 2>&1
  echo "$tag EXIT=$?"
}

# bit-width sweep (block 16, the session-1 setting)
run int3_g16  --kv_quant RTN --kv_bits 3  --kv_block_size 16

# coarser scales at INT4: raise the cache error without changing bit-width
run int4_g64   --kv_quant RTN --kv_bits 4 --kv_block_size 64
run int4_g256  --kv_quant RTN --kv_bits 4 --kv_block_size 256
run int4_g1024 --kv_quant RTN --kv_bits 4 --kv_block_size 1024

# asymmetric K/V allocation: is value error more expensive than key error?
# (V has no RMSNorm in Wan's DiT; K does, via norm_k.)
run k4v2 --kv_quant RTN --kv_bits 4 --kv_key_bits 4 --kv_value_bits 2
run k2v4 --kv_quant RTN --kv_bits 4 --kv_key_bits 2 --kv_value_bits 4
echo "ALL_DONE"
