#!/usr/bin/env bash
# v3 (2차 리뷰 반영): R4f 평균항 부호 수정 · 저장 dtype(fp16/int8)·chunk 별 scale · R4m 추가 · PSAC_SCOPE=chunk20(온라인 대응) 와 all(오프라인 대응) 둘 다
set -u
PY=/vfsdata/gpu_data/gpu/sf-venv/bin/python
OUTDIR=/home/devel/Quantization/results/session9/cand1_diag
cd /home/devel/Quantization
export TUM_LAMBDAS=1.0,0.25 PSAC_RANKS=8,16 PSAC_VARIANTS=B,E,D,N CONSEC_PROMPTS=0,1,2
echo "START v3 $(date '+%F %T') HEAD=$(git rev-parse --short HEAD)" >> $OUTDIR/run_all.status
for scope in chunk20 all; do
  for cap in consec consec2; do
    PSAC_SCOPE=$scope CONSEC_ROOT=/vfsdata/gpu_data/attr_dump/$cap/points CAND1_OUT=$OUTDIR/v3_${scope}_${cap}.json $PY tools/cand1_diag.py > $OUTDIR/v3_${scope}_${cap}.log 2>&1
    echo "v3 $scope $cap EXIT=$? $(date +%T)" >> $OUTDIR/run_all.status
  done
done
echo "V3_DONE $(date '+%F %T')" >> $OUTDIR/run_all.status
