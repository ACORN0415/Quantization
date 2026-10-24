#!/usr/bin/env bash
# S9 PSAC8 참고 백엔드 관문 (사전 등록 §3): 1 프롬프트 · 63f · A1 · seed 0. G1 noquant vs BF16, G2 chunk0, G3 상태, G4 바이트, 속도.
set -u
cd /vfsdata/gpu_data/gpu/Self-Forcing
. tools/runlib.sh
PY=/vfsdata/gpu_data/gpu/sf-venv/bin/python
O=results/session9/psac_gates
COMMON="--config_path configs/gateA_A1.yaml --checkpoint_path checkpoints/self_forcing_dmd.pt --data_path prompts/quant3/prompts_eval_psac1.txt --num_output_frames 63 --save_with_index --seed 0 --use_ema"
LOGP="--psac_log_points 1,0;6,15;10,29;20,29"
s6_run BF16     $PY inference.py $COMMON --output_folder $O/BF16/videos --stats_path $O/BF16.stats.json --latent_dump_root $O/BF16/latents
PSAC_NOQUANT=1 s6_run NOQUANT $PY inference.py $COMMON --output_folder $O/NOQUANT/videos --stats_path $O/NOQUANT.stats.json --latent_dump_root $O/NOQUANT/latents --kv_quant RTN --kv_bits 2 --attn_backend psac --psac_rank 0 --psac_bias none $LOGP --psac_state_log $O/NOQUANT.state.json
s6_run PSAC8    $PY inference.py $COMMON --output_folder $O/PSAC8/videos --stats_path $O/PSAC8.stats.json --latent_dump_root $O/PSAC8/latents --kv_quant RTN --kv_bits 2 --attn_backend psac --psac_rank 8 --psac_bias r4d $LOGP --psac_state_log $O/PSAC8.state.json --kv_bytes_report $O/PSAC8.bytes.json
s6_run INT2L1   $PY inference.py $COMMON --output_folder $O/INT2L1/videos --stats_path $O/INT2L1.stats.json --latent_dump_root $O/INT2L1/latents --kv_quant RTN --kv_bits 2 --attn_backend psac --psac_rank 0 --psac_bias delta --psac_lambda 1 $LOGP --psac_state_log $O/INT2L1.state.json --kv_bytes_report $O/INT2L1.bytes.json
s6_run K4V2L1   $PY inference.py $COMMON --output_folder $O/K4V2L1/videos --stats_path $O/K4V2L1.stats.json --latent_dump_root $O/K4V2L1/latents --kv_quant RTN --kv_bits 4 --kv_key_bits 4 --kv_value_bits 2 --attn_backend psac --psac_rank 0 --psac_bias delta --psac_lambda 1 $LOGP --psac_state_log $O/K4V2L1.state.json --kv_bytes_report $O/K4V2L1.bytes.json
for a in BF16 NOQUANT PSAC8 INT2L1 K4V2L1; do s6_expect_files $a.latent 1 "$O/$a/latents/prompt000/full.pt"; done
s6_run gates $PY /home/devel/Quantization/tools/psac_gates.py $O
s6_verdict PSAC_GATES
