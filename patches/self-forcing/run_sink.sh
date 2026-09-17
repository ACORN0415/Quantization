#!/usr/bin/env bash
# Pseudo-sink diagnostic: pin chunk 0 (3 latent frames) in the rolling cache and
# see whether the chunk-7 sensitivity step in the teacher-forced injection error
# disappears. sink_size is already implemented upstream; it just defaults to 0.
set -u
PY=~/gpu/sf-venv/bin/python
C="--use_ema --num_output_frames 63 --max_prompts 3 --config_path configs/self_forcing_dmd_long_sink3.yaml"

run () { tag="$1"; shift; echo "=== $tag ==="; mkdir -p "results/tf_${tag}"
  $PY teacher_forced.py --tag "$tag" $C "$@" \
    --out_dir "results/tf_${tag}" > "results/tf_${tag}/run.log" 2>&1
  echo "$tag EXIT=$?"; }

# gate first: with no quantization the second pass must still be bit-exact
run sink3_gate --bitexact_gate --max_prompts 1
grep -E "BIT-EXACT" results/tf_sink3_gate/run.log

run sink3_kv_int4 --kv_quant RTN --kv_bits 4
run sink3_kv_int2 --kv_quant RTN --kv_bits 2
echo "ALL_DONE"
