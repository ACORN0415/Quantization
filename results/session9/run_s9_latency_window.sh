#!/usr/bin/env bash
# S9 창 길이별 종단 오버헤드 — 사전 등록 Quantization/문서/지시서/S9_latency_window_prereg.md. 3 프롬프트 · 63f 출력 · sink 0 · 디코드 포함.
set -u
cd /vfsdata/gpu_data/gpu/Self-Forcing
. tools/runlib.sh
PY=/vfsdata/gpu_data/gpu/sf-venv/bin/python
O=results/session9/latency_window; mkdir -p $O
P=prompts/quant3/prompts_eval_c1_3.txt
echo "START $(date '+%F %T') HEAD=$(git rev-parse --short HEAD) diff=$(git diff HEAD | sha256sum | cut -c1-16)" | tee $O/run.status
run () { tag="$1"; cfg="$2"; shift 2
  s6_run "$tag" $PY inference.py --config_path $cfg --checkpoint_path checkpoints/self_forcing_dmd.pt --data_path $P --num_output_frames 63 --save_with_index --seed 0 --use_ema \
    --output_folder $O/$tag/videos --stats_path $O/$tag.stats.json --kv_bytes_report $O/$tag.bytes.json "$@"; }
for W in 12 21 36 63; do
  case $W in 12) C=configs/gateA_A3.yaml;; 21) C=configs/gateA_A1.yaml;; 36) C=configs/gateA_A1_w36.yaml;; 63) C=configs/gateA_A1_w63.yaml;; esac
  run bf16_w$W $C
  run int2_w$W $C --kv_quant RTN --kv_bits 2 --attn_backend fused --fused_bias_lambda 1
  run k4v2_w$W $C --kv_quant RTN --kv_bits 4 --kv_key_bits 4 --kv_value_bits 2 --attn_backend fused --fused_bias_lambda 1
done
echo "END $(date '+%F %T')" | tee -a $O/run.status
s6_verdict LATENCY_WINDOW
