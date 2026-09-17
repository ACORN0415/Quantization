#!/usr/bin/env bash
# Gate G: how self/recent/oldest mass evolves across the whole rollout, by
# bit-width, to locate where the distribution starts collapsing.
set -u
PY=~/gpu/sf-venv/bin/python
CH="0 1 2 3 4 5 6 7 8 10 12 14 16 18 20"
run () { tag="$1"; shift; echo "=== $tag ==="
  $PY attn_mass.py --tag "g_${tag}" --config_path configs/gateA_A1.yaml \
    --data_path prompts/quant3/prompts10.txt --max_prompts 3 \
    --chunks $CH "$@" > "results/attn_mass_g_${tag}.log" 2>&1
  echo "$tag EXIT=$?"; }
run bf16
run int4 --kv_quant RTN --kv_bits 4
run int3 --kv_quant RTN --kv_bits 3
run int2 --kv_quant RTN --kv_bits 2
run k4v2 --kv_quant RTN --kv_bits 4 --kv_key_bits 4 --kv_value_bits 2
run k2v4 --kv_quant RTN --kv_bits 4 --kv_key_bits 2 --kv_value_bits 4
echo "ALL_DONE"
