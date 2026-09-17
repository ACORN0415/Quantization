#!/usr/bin/env bash
# Step F: teacher-forced injection error for the five configurations that have
# a matching free-running run from session 1.
set -u
PY=~/gpu/sf-venv/bin/python
C="--use_ema --num_output_frames 63 --max_prompts 3"

run () { tag="$1"; shift; echo "=== $tag ==="; mkdir -p "results/tf_${tag}"
  $PY teacher_forced.py --tag "$tag" $C "$@" \
    --out_dir "results/tf_${tag}" --latent_dump_root "results/latents63_tf/${tag}" \
    > "results/tf_${tag}/run.log" 2>&1
  echo "$tag EXIT=$?"; grep -E "peak|kv_quant\]|fakequant" "results/tf_${tag}/run.log" | tail -2; }

run kv_int4      --kv_quant RTN  --kv_bits 4
run kv_int2      --kv_quant RTN  --kv_bits 2
run w4a4         --fakequant w4a4
run w8a8         --fakequant w8a8
run kv_only_w4a4 --fakequant kv_only_w4a4
echo "ALL_DONE"
