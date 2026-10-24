#!/usr/bin/env bash
# 후보 1 진단(cand1_diag.py) 감시: 프로세스 생존·GPU 사용·진행률·최근 로그·오류.
#   tools/watch_cand1.sh        5초마다 갱신
#   tools/watch_cand1.sh 30     30초마다
#   tools/watch_cand1.sh 0      한 번 출력하고 종료
INTERVAL="${1:-5}"
LOGDIR=/home/devel/Quantization/results/session9/cand1_diag

render() {
  echo "================ $(date '+%Y-%m-%d %H:%M:%S') ================"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
             --format=csv,noheader,nounits 2>/dev/null |
    awk -F', ' '{printf "GPU   util %3s%%   mem %5s/%5s MiB   %s C   %s W\n",$1,$2,$3,$4,$5}'
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null |
    sed 's/^/      proc  /'
  echo
  local line pid et
  line=$(pgrep -af "^[^ ]*python[0-9.]* [^ ]*cand1_diag\.py" | head -1)   # 명령행이 python 으로 시작하는 것만 (래퍼 셸 제외)
  if [ -n "$line" ]; then
    pid=${line%% *}; et=$(ps -o etime= -p "$pid" | tr -d ' ')
    printf "RUN   pid %s   elapsed %s   cpu %s%%   rss %s MB\n" "$pid" "$et" \
      "$(ps -o %cpu= -p "$pid" | tr -d ' ')" "$(( $(ps -o rss= -p "$pid" | tr -d ' ') / 1024 ))"
  else
    printf "RUN   (cand1_diag.py 실행 중 아님)\n"
  fi
  echo
  for log in "$LOGDIR"/*.log; do
    [ -e "$log" ] || continue
    local done total
    done=$(grep -cE '^\[[0-9]+/[0-9]+\]' "$log"); total=$(grep -oE '^\[[0-9]+/([0-9]+)\]' "$log" | tail -1 | sed -E 's/.*\/([0-9]+)\]/\1/')
    printf "LOG   %s   진행 %s/%s%s\n" "$(basename "$log")" "${done:-0}" "${total:-?}" \
      "$(grep -q 'DONE' "$log" && echo '   DONE' || true)"
    grep -iE 'error|traceback|exception|nan|assert' "$log" | tail -3 | sed 's/^/      !! /'
    tail -4 "$log" | cut -c1-200 | sed 's/^/      /'
    echo
  done
}

if [ "$INTERVAL" = "0" ]; then render; exit 0; fi
while true; do clear 2>/dev/null; render; sleep "$INTERVAL"; done
