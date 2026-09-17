#!/usr/bin/env bash
# Gate A: isolate what makes LongLive robust — the sink, the shorter window, or
# the fact that it was fine-tuned with that cache structure. Same Self-Forcing
# weights throughout; only the cache configuration changes.
set -u
PY=~/gpu/sf-venv/bin/python
PROMPTS=prompts/quant3/prompts10.txt

run () {  # cfg, tag, extra...
  cfg="$1"; tag="$2"; shift 2
  out="results/gateA/${cfg}_${tag}"
  echo "=== ${cfg}_${tag} ==="
  mkdir -p "$out/videos"
  $PY inference.py \
    --config_path "configs/gateA_${cfg}.yaml" \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path "$PROMPTS" \
    --output_folder "$out/videos" \
    --latent_dump_root "results/latents_gateA/${cfg}_${tag}" \
    --stats_path "$out/stats.json" \
    --kv_err_log "$out/kv_err.json" \
    --num_output_frames 63 --save_with_index --seed 0 --use_ema \
    "$@" > "$out/run.log" 2>&1
  echo "${cfg}_${tag} EXIT=$?"
}

for cfg in "$@"; do
  run "$cfg" bf16
  run "$cfg" ctrl  --noise_perturb 1e-3
  run "$cfg" int4  --kv_quant RTN --kv_bits 4
  run "$cfg" int2  --kv_quant RTN --kv_bits 2
done
echo "ALL_DONE"
