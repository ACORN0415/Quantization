#!/usr/bin/env bash
# Gate F2: does subtracting the Jensen bias pull the distribution back?
set -u
PY=~/gpu/sf-venv/bin/python
run () { tag="$1"; cfg="$2"; bits="$3"; shift 3
  out="results/gateF2/${tag}"; echo "=== $tag ==="
  mkdir -p "$out/videos"
  $PY inference.py --config_path "configs/${cfg}.yaml" \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/quant3/prompts10.txt --output_folder "$out/videos" \
    --latent_dump_root "results/latents_gateF2/${tag}" --stats_path "$out/stats.json" \
    --num_output_frames 63 --save_with_index --seed 0 --use_ema \
    --kv_quant RTN --kv_bits "$bits" "$@" > "$out/run.log" 2>&1
  echo "$tag EXIT=$?"; }

run a1_int2_tum gateA_A1 2 --tum_correct
run a1_int4_tum gateA_A1 4 --tum_correct
run a4_int2_tum gateA_A4 2 --tum_correct
echo "ALL_DONE"
