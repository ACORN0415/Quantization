#!/usr/bin/env bash
# 63f 창 fused 팔만, BF16 버퍼 선할당 생략(KV_NO_BF16_BUFFERS=1). BF16 63f 는 본질적으로 18 GB 라 불가.
set -u; cd /vfsdata/gpu_data/gpu/Self-Forcing; . tools/runlib.sh
PY=/vfsdata/gpu_data/gpu/sf-venv/bin/python; O=results/session9/latency_window; P=prompts/quant3/prompts_eval_c1_3.txt
export KV_NO_BF16_BUFFERS=1
run () { tag="$1"; shift; s6_run "$tag" $PY inference.py --config_path configs/gateA_A1_w63.yaml --checkpoint_path checkpoints/self_forcing_dmd.pt --data_path $P --num_output_frames 63 --save_with_index --seed 0 --use_ema --output_folder $O/$tag/videos --stats_path $O/$tag.stats.json --kv_bytes_report $O/$tag.bytes.json "$@"; }
run int2_w63_nobuf --kv_quant RTN --kv_bits 2 --attn_backend fused --fused_bias_lambda 1
run k4v2_w63_nobuf --kv_quant RTN --kv_bits 4 --kv_key_bits 4 --kv_value_bits 2 --attn_backend fused --fused_bias_lambda 1
# 대조: 21f 에서 선할당 생략이 시간을 바꾸지 않는지 (같은 결과여야 함 — fused 는 첫 호출에 버퍼를 비움)
s6_run int2_w21_nobuf $PY inference.py --config_path configs/gateA_A1.yaml --checkpoint_path checkpoints/self_forcing_dmd.pt --data_path $P --num_output_frames 63 --save_with_index --seed 0 --use_ema --output_folder $O/int2_w21_nobuf/videos --stats_path $O/int2_w21_nobuf.stats.json --kv_bytes_report $O/int2_w21_nobuf.bytes.json --kv_quant RTN --kv_bits 2 --attn_backend fused --fused_bias_lambda 1
echo "W63_DONE $(date '+%F %T')" >> $O/run.status; s6_verdict LATENCY_W63
