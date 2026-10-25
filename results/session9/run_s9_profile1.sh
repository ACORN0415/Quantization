#!/usr/bin/env bash
# S9 프로파일 1부 — 사전 등록 S9_profile_prereg.md §1. 1 프롬프트 · 36f · 창 21f(A1) · 디코드 포함.
set -u; cd /vfsdata/gpu_data/gpu/Self-Forcing; . tools/runlib.sh
PY=/vfsdata/gpu_data/gpu/sf-venv/bin/python; O=results/session9/profile; P=prompts/quant3/prompts_eval_c1_1.txt
echo "START1 $(date '+%F %T') HEAD=$(git rev-parse --short HEAD) diff=$(git diff HEAD | sha256sum | cut -c1-16)" >> $O/run.status
COMMON="--config_path configs/gateA_A1.yaml --checkpoint_path checkpoints/self_forcing_dmd.pt --data_path $P --num_output_frames 36 --save_with_index --seed 0 --use_ema"
run () { tag="$1"; shift; rm -f $O/prof_$tag.json; TORCH_PROFILE_OUT=$O/prof_$tag.json s6_run "$tag" $PY inference.py $COMMON --output_folder $O/$tag/videos --stats_path $O/$tag.stats.json "$@"; }
run bf16
run int2_l0 --kv_quant RTN --kv_bits 2 --attn_backend fused --fused_bias_lambda 0 --kv_bytes_report $O/int2_l0.bytes.json
run int2_l1 --kv_quant RTN --kv_bits 2 --attn_backend fused --fused_bias_lambda 1 --kv_bytes_report $O/int2_l1.bytes.json
run k4v2_l1 --kv_quant RTN --kv_bits 4 --kv_key_bits 4 --kv_value_bits 2 --attn_backend fused --fused_bias_lambda 1 --kv_bytes_report $O/k4v2_l1.bytes.json
echo "PROFILE1_DONE $(date '+%F %T')" >> $O/run.status; s6_verdict PROFILE1
