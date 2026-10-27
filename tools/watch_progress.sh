#!/usr/bin/env bash
# Current reconstruction/feasibility progress. 0 prints once; default refresh 5s.
set -eu
quant_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
quant_data="${SFQ_DATA_ROOT:-$HOME/sfq_data}"
quant_python="$quant_data/py"
if [ ! -x "$quant_python" ]; then quant_python=python3; fi
if [ "${1:-5}" = "0" ]; then
  exec "$quant_python" "$quant_repo/src/scripts/progress.py" --data "$quant_data"
fi
exec "$quant_python" "$quant_repo/src/scripts/progress.py" --data "$quant_data" --watch --interval "${1:-5}"
