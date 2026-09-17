#!/usr/bin/env bash
set -u
PY=~/gpu/sf-venv/bin/python
for b in 4 2; do
  tag="e2c_int${b}_sink_int2"; out="results/gateE2/${tag}"
  echo "=== $tag ==="; mkdir -p "$out/videos"
  $PY inference.py --config_path configs/gateA_A2.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/quant3/prompts10.txt --output_folder "$out/videos" \
    --latent_dump_root "results/latents_gateE2/${tag}" --stats_path "$out/stats.json" \
    --num_output_frames 63 --save_with_index --seed 0 --use_ema \
    --kv_quant RTN --kv_bits "$b" --sink_mode int2 > "$out/run.log" 2>&1
  echo "$tag EXIT=$?"
done
echo "ALL_DONE"
