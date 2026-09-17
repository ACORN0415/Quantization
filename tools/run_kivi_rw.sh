#!/usr/bin/env bash
# 4-1: KIVI with a residual window — the most recent chunk stays BF16, which is
# what the paper specifies and the upstream implementation omits.
set -u
PY=~/gpu/sf-venv/bin/python
run () { tag="$1"; bits="$2"; resid="$3"
  out="results/kivi_rw/${tag}"; echo "=== $tag ==="
  mkdir -p "$out/videos"
  $PY inference.py --config_path configs/gateA_A1.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/quant3/prompts10.txt --output_folder "$out/videos" \
    --latent_dump_root "results/latents_kivirw/${tag}" --stats_path "$out/stats.json" \
    --kv_err_log "$out/kv_err.json" \
    --num_output_frames 63 --save_with_index --seed 0 --use_ema \
    --kv_quant KIVI --kv_bits "$bits" --kv_residual_tokens "$resid" \
    > "$out/run.log" 2>&1
  echo "$tag EXIT=$?"; }
run kivi4_rw0    4 0       # KIVI-style, no residual window (what sessions 1-3 measured)
run kivi4_rw1ch  4 4680    # true KIVI: newest chunk in BF16
run kivi2_rw0    2 0
run kivi2_rw1ch  2 4680
echo "ALL_DONE"
touch results/kivi_rw/done
