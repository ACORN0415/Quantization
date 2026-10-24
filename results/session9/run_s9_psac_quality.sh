#!/usr/bin/env bash
# S9 PSAC8 품질 실험 — 사전 등록 Quantization/문서/지시서/S9_PSAC8_quality_prereg.md 그대로.
# 4 팔 × prompts_eval_psac10 (미사용 10개) · 63f · A1 · seed 0 · EMA · 디코드. 양자화 세 팔은 같은 psac 복원 경로.
set -u
cd /vfsdata/gpu_data/gpu/Self-Forcing
. tools/runlib.sh
PY=/vfsdata/gpu_data/gpu/sf-venv/bin/python
VB=/vfsdata/gpu_data/gpu/vbench-venv/bin/python
Q=/home/devel/Quantization
O=results/session9/psac_quality
mkdir -p $O
echo "START $(date '+%F %T') HEAD=$(git rev-parse --short HEAD) diff=$(git diff HEAD | sha256sum | cut -c1-16)" | tee $O/run.status
P=prompts/quant3/prompts_eval_psac10.txt
COMMON="--config_path configs/gateA_A1.yaml --checkpoint_path checkpoints/self_forcing_dmd.pt --data_path $P --num_output_frames 63 --save_with_index --seed 0 --use_ema"
s6_run PSAC8  $PY inference.py $COMMON --output_folder $O/PSAC8/videos  --stats_path $O/PSAC8.stats.json  --kv_bytes_report $O/PSAC8.bytes.json  --kv_quant RTN --kv_bits 2 --attn_backend psac --psac_rank 8 --psac_bias r4d
s6_run K4V2L1 $PY inference.py $COMMON --output_folder $O/K4V2L1/videos --stats_path $O/K4V2L1.stats.json --kv_bytes_report $O/K4V2L1.bytes.json --kv_quant RTN --kv_bits 4 --kv_key_bits 4 --kv_value_bits 2 --attn_backend psac --psac_rank 0 --psac_bias delta --psac_lambda 1
s6_run INT2L1 $PY inference.py $COMMON --output_folder $O/INT2L1/videos --stats_path $O/INT2L1.stats.json --kv_bytes_report $O/INT2L1.bytes.json --kv_quant RTN --kv_bits 2 --attn_backend psac --psac_rank 0 --psac_bias delta --psac_lambda 1
s6_run BF16   $PY inference.py $COMMON --output_folder $O/BF16/videos   --stats_path $O/BF16.stats.json
for a in PSAC8 K4V2L1 INT2L1 BF16; do s6_expect_files $a.videos 10 "$O/$a/videos/*.mp4"; done
RUNS="PSAC8=$O/PSAC8/videos K4V2L1=$O/K4V2L1/videos INT2L1=$O/INT2L1/videos BF16=$O/BF16/videos"
s6_run musiq  $PY $Q/tools/metrics_all.py --runs $RUNS --prompts $P --skip lpips clip flow --out_csv $O/musiq.csv
s6_run vbench $VB $Q/tools/vbench_dims.py --device cpu --out_csv $O/vbench.csv --runs $RUNS --dims subject_consistency temporal_flickering
s6_run dyn    $VB $Q/tools/vbench_dims.py --device cuda --out_csv $O/vbench_dyn.csv --runs $RUNS --dims dynamic_degree
s6_run motion $PY $Q/tools/motion_proxy.py --out $O/motion_proxy.json --runs $RUNS
s6_run analyse $PY $Q/tools/analyse_psac_quality.py $O
echo "END $(date '+%F %T')" | tee -a $O/run.status
s6_verdict PSAC_QUALITY
