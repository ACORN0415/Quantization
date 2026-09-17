#!/usr/bin/env bash
# 4-2: 126 frames (42 chunks) with the cache length unchanged, so the rolling
# window keeps evicting. Does the INT4 damage keep growing past chunk 20, or
# level off? Gate D showed a 126-frame *cache* would need 33.7 GB and does not
# fit on a 4090 — this lengthens the rollout, not the cache.
set -u
PY=~/gpu/sf-venv/bin/python
run () { tag="$1"; cfg="$2"; shift 2
  out="results/long126/${tag}"; echo "=== $tag ==="
  mkdir -p "$out/videos"
  $PY inference.py --config_path "configs/${cfg}.yaml" \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/quant3/prompts5.txt --output_folder "$out/videos" \
    --latent_dump_root "results/latents_long126/${tag}" --stats_path "$out/stats.json" \
    --num_output_frames 126 --save_with_index --seed 0 --use_ema \
    "$@" > "$out/run.log" 2>&1
  echo "$tag EXIT=$?"; }
run a1_bf16 gateA_A1
run a1_int4 gateA_A1 --kv_quant RTN --kv_bits 4
run a4_bf16 gateA_A4
run a4_int4 gateA_A4 --kv_quant RTN --kv_bits 4
echo "ALL_DONE"
touch results/long126/done
