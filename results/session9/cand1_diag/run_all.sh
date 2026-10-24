#!/usr/bin/env bash
# 후보 1 진단 (a) — consec(prompts 0~2) → consec2(prompts 3~5), rank 4/8/16, 변형 B/E/D
set -u
PY=/vfsdata/gpu_data/gpu/sf-venv/bin/python
OUTDIR=/home/devel/Quantization/results/session9/cand1_diag
cd /home/devel/Quantization
echo "START v2(팔 보강: R1_lλ, R4d0/R4f0, 변형 N) $(date '+%F %T') HEAD=$(git rev-parse --short HEAD) sf-tools@$(git -C /vfsdata/gpu_data/gpu/Self-Forcing rev-parse --short HEAD)" | tee -a $OUTDIR/run_all.status
PSAC_RANKS=4,8,16 CONSEC_ROOT=/vfsdata/gpu_data/attr_dump/consec/points  CONSEC_PROMPTS=0,1,2 CAND1_OUT=$OUTDIR/consec.json  $PY tools/cand1_diag.py > $OUTDIR/consec.log 2>&1;  echo "consec  EXIT=$? $(date '+%T')" | tee -a $OUTDIR/run_all.status
PSAC_RANKS=4,8,16 CONSEC_ROOT=/vfsdata/gpu_data/attr_dump/consec2/points CONSEC_PROMPTS=0,1,2 CAND1_OUT=$OUTDIR/consec2.json $PY tools/cand1_diag.py > $OUTDIR/consec2.log 2>&1; echo "consec2(파일 p0~2 = prompts 3~5) EXIT=$? $(date '+%T')" | tee -a $OUTDIR/run_all.status
echo "ALL_DONE $(date '+%F %T')" | tee -a $OUTDIR/run_all.status
