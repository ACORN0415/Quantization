#!/usr/bin/env bash
set -u
PY=~/gpu/sf-venv/bin/python
for c in A1 A2 A3 A4 A5; do
  R=""
  for t in bf16 ctrl int4 int2; do R="$R ${c}_${t}=results/gateA/${c}_${t}/videos"; done
  echo "=== $c (ref ${c}_bf16) ==="
  $PY metrics_all.py --runs $R --ref "${c}_bf16" \
    --prompts prompts/quant3/prompts10.txt \
    --out_csv "results/gateA_metrics_${c}.csv" --skip flow clip
  echo "$c EXIT=$?"
done
echo "ALL_DONE"
