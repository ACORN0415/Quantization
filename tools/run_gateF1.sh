#!/usr/bin/env bash
# Gate F1: slot-wise key noise variance against the attention mass it gains.
# Jensen predicts E[exp(s+eps)] = exp(s)exp(sigma^2/2), so a slot whose keys
# carry more quantization noise should attract mass that BF16 did not give it.
set -u
PY=~/gpu/sf-venv/bin/python
run () { tag="$1"; shift; echo "=== $tag ==="
  $PY attn_mass.py --tag "$tag" --config_path configs/gateA_A1.yaml \
    --data_path prompts/quant3/prompts10.txt --max_prompts 3 \
    --chunks 6 14 20 "$@" > "results/attn_mass_${tag}.log" 2>&1
  echo "$tag EXIT=$?"; }
run f1_bf16
run f1_int4 --kv_quant RTN --kv_bits 4
run f1_int3 --kv_quant RTN --kv_bits 3
run f1_int2 --kv_quant RTN --kv_bits 2
echo "ALL_DONE"
