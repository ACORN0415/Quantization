#!/usr/bin/env bash
set -u
PY=~/gpu/sf-venv/bin/python
for spec in A1:gateA_A1 A2:gateA_A2; do
  tag="${spec%%:*}"; cfg="${spec##*:}"
  echo "=== E1 $tag ==="
  $PY slot_contamination.py --tag "e1_${tag}_int4" --config_path "configs/${cfg}.yaml" \
    --max_prompts 10 --at_chunks 8 14 20 --kv_quant RTN --kv_bits 4 \
    > "results/gateE_e1_${tag}.log" 2>&1
  echo "$tag EXIT=$?"
done
echo "ALL_DONE"
