#!/usr/bin/env bash
# Gate B: attention mass on the first chunk, before and after it is evicted.
set -u
PY=~/gpu/sf-venv/bin/python
run () { tag="$1"; shift; echo "=== $tag ==="
  $PY attn_mass.py --tag "$tag" --data_path prompts/quant3/prompts10.txt \
    --max_prompts 3 "$@" > "results/attn_mass_${tag}.log" 2>&1
  echo "$tag EXIT=$?"
  grep -E "verify|chunk .* step|slot|BUCKET" "results/attn_mass_${tag}.log" | head -30; }

# A1: no sink — does chunk 0 take more mass than its token share, and where does
# that mass go once it is evicted at chunk 7?
run A1_bf16  --config_path configs/gateA_A1.yaml --chunks 6 7 14 20
# A2: sink pinned — is the mass on the sink sustained?
run A2_bf16  --config_path configs/gateA_A2.yaml --chunks 6 7 14 20
# INT4 / INT2: does quantizing the cache shift mass toward the quantized keys?
run A1_int4  --config_path configs/gateA_A1.yaml --chunks 6 7 14 20 --kv_quant RTN --kv_bits 4
run A1_int2  --config_path configs/gateA_A1.yaml --chunks 6 7 14 20 --kv_quant RTN --kv_bits 2
echo "ALL_DONE"
