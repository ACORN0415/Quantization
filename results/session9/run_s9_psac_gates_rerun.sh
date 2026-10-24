#!/usr/bin/env bash
# 관문 재실행: NOQUANT 팔만 (inference.py 출력 블록 수정 + G1' per-call FA2 비교 추가) → 관문 판정. 나머지 팔 latent 는 1차 실행 그대로.
set -u
cd /vfsdata/gpu_data/gpu/Self-Forcing
. tools/runlib.sh
PY=/vfsdata/gpu_data/gpu/sf-venv/bin/python
O=results/session9/psac_gates
COMMON="--config_path configs/gateA_A1.yaml --checkpoint_path checkpoints/self_forcing_dmd.pt --data_path prompts/quant3/prompts_eval_psac1.txt --num_output_frames 63 --save_with_index --seed 0 --use_ema"
LOGP="--psac_log_points 1,0;6,15;10,29;20,29"
PSAC_NOQUANT=1 s6_run NOQUANT $PY inference.py $COMMON --output_folder $O/NOQUANT/videos --stats_path $O/NOQUANT.stats.json --latent_dump_root $O/NOQUANT/latents --kv_quant RTN --kv_bits 2 --attn_backend psac --psac_rank 0 --psac_bias none $LOGP --psac_state_log $O/NOQUANT.state.json
s6_run gates $PY /home/devel/Quantization/tools/psac_gates.py $O
s6_verdict PSAC_GATES_RERUN
