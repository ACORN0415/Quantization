#!/usr/bin/env bash
# S9 후보 2 C1 1단계: P3 fused 경로 λ 응답 — 사전 등록 Quantization/문서/지시서/S9_C1_lambda_prereg.md 그대로.
set -u
cd /vfsdata/gpu_data/gpu/Self-Forcing
. tools/runlib.sh
PY=/vfsdata/gpu_data/gpu/sf-venv/bin/python
VB=/vfsdata/gpu_data/gpu/vbench-venv/bin/python
Q=/home/devel/Quantization
O=results/session9/c1_lambda; mkdir -p $O
P=prompts/quant3/prompts_eval_c1_10.txt
echo "START $(date '+%F %T') HEAD=$(git rev-parse --short HEAD) diff=$(git diff HEAD | sha256sum | cut -c1-16)" | tee $O/run.status
COMMON="--config_path configs/gateA_A1.yaml --checkpoint_path checkpoints/self_forcing_dmd.pt --data_path $P --num_output_frames 63 --save_with_index --seed 0 --use_ema"
arm () { tag="$1"; shift
  s6_run "$tag" $PY inference.py $COMMON --output_folder $O/$tag/videos --stats_path $O/$tag.stats.json --kv_bytes_report $O/$tag.bytes.json --kv_quant RTN --attn_backend fused "$@"
  s6_expect_files "$tag.videos" 10 "$O/$tag/videos/*.mp4"; }
arm int2_l0.5 --kv_bits 2 --fused_bias_lambda 0.5
arm int2_l1   --kv_bits 2 --fused_bias_lambda 1
arm int2_l2   --kv_bits 2 --fused_bias_lambda 2
arm int2_l4   --kv_bits 2 --fused_bias_lambda 4
arm k4v2_l1   --kv_bits 4 --kv_key_bits 4 --kv_value_bits 2 --fused_bias_lambda 1
arm k4v2_l2   --kv_bits 4 --kv_key_bits 4 --kv_value_bits 2 --fused_bias_lambda 2
s6_run BF16 $PY inference.py $COMMON --output_folder $O/BF16/videos --stats_path $O/BF16.stats.json
s6_expect_files BF16.videos 10 "$O/BF16/videos/*.mp4"
RUNS=""; for t in int2_l0.5 int2_l1 int2_l2 int2_l4 k4v2_l1 k4v2_l2 BF16; do RUNS="$RUNS $t=$O/$t/videos"; done
s6_run musiq  $PY $Q/tools/metrics_all.py --runs $RUNS --prompts $P --skip lpips clip flow --out_csv $O/musiq.csv
s6_run vbench $VB $Q/tools/vbench_dims.py --device cpu --out_csv $O/vbench.csv --runs $RUNS --dims subject_consistency temporal_flickering
s6_run dyn    $VB $Q/tools/vbench_dims.py --device cuda --out_csv $O/vbench_dyn.csv --runs $RUNS --dims dynamic_degree
s6_run motion $PY $Q/tools/motion_proxy.py --out $O/motion_proxy.json --runs $RUNS
s6_run analyse $PY $Q/tools/analyse_arms.py $O --arms int2_l0.5 int2_l1 int2_l2 int2_l4 k4v2_l1 k4v2_l2 BF16 \
  --contrasts int2_l2-int2_l1 int2_l0.5-int2_l1 int2_l4-int2_l1 k4v2_l2-k4v2_l1 int2_l1-BF16 int2_l2-BF16 k4v2_l1-BF16
echo "END $(date '+%F %T')" | tee -a $O/run.status
s6_verdict C1_LAMBDA
