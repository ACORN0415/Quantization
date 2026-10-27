#!/usr/bin/env bash
set -eu
quant_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
quant_data="${SFQ_DATA_ROOT:-$HOME/sfq_data}"
exec "$quant_data/py" "$quant_repo/tools/pause_policy_study.py" resume --data "$quant_data/policy_study"
