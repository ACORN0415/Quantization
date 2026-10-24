#!/usr/bin/env bash
# K4V2 기준선 팔만 (ARMS_FILTER): K INT4 무보정 / TUM Δ=scale λ1 / Δ=2·scale λ1, 참조 R0·R1_l0.25 포함
set -u
PY=/vfsdata/gpu_data/gpu/sf-venv/bin/python
OUTDIR=/home/devel/Quantization/results/session9/cand1_diag
cd /home/devel/Quantization
export ARMS_FILTER=R0,R1_l0.25,K4_ PSAC_RANKS=8 PSAC_VARIANTS=E CONSEC_PROMPTS=0,1,2
CONSEC_ROOT=/vfsdata/gpu_data/attr_dump/consec/points  CAND1_OUT=$OUTDIR/k4_consec.json  $PY tools/cand1_diag.py > $OUTDIR/k4_consec.log 2>&1;  echo "k4 consec  EXIT=$? $(date +%T)" >> $OUTDIR/run_all.status
CONSEC_ROOT=/vfsdata/gpu_data/attr_dump/consec2/points CAND1_OUT=$OUTDIR/k4_consec2.json $PY tools/cand1_diag.py > $OUTDIR/k4_consec2.log 2>&1; echo "k4 consec2 EXIT=$? $(date +%T)" >> $OUTDIR/run_all.status
echo "K4_DONE $(date +%T)" >> $OUTDIR/run_all.status
