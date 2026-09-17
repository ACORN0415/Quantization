#!/usr/bin/env bash
# Live view of the quantization experiments.
#   ~/gpu/watch_progress.sh          refresh every 5 min
#   ~/gpu/watch_progress.sh 30       refresh every 30 s
#   ~/gpu/watch_progress.sh 0        print once and exit
INTERVAL="${1:-300}"
SF=~/gpu/Self-Forcing
LL=~/gpu/LongLive
REPO=~/gpu/Quantization

bar() {  # bar <done> <total> <width>
  local d=$1 t=$2 w=${3:-20} i filled
  [ "$t" -le 0 ] && t=1
  filled=$(( d * w / t ))
  printf "["
  for ((i=0;i<w;i++)); do [ $i -lt $filled ] && printf "#" || printf "."; done
  printf "] %d/%d" "$d" "$t"
}

gate_state() {  # gate_state <label> <path-that-exists-when-done>
  if [ -e "$2" ]; then printf "  [x] %s\n" "$1"; else printf "  [ ] %s\n" "$1"; fi
}

render() {
  echo "================ $(date '+%Y-%m-%d %H:%M:%S') ================"

  # --- hardware ---
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
             --format=csv,noheader,nounits 2>/dev/null |
    awk -F', ' '{printf "GPU   %3s%%   %5s/%5s MiB   %s C   %s W\n",$1,$2,$3,$4,$5}'
  df -h /home 2>/dev/null | awk 'NR==2{printf "DISK  %s used of %s  (%s free)\n",$3,$2,$4}'

  # --- what is running ---
  echo
  local line pid script tag et
  line=$(pgrep -af "inference\.py|attn_mass\.py|slot_contamination\.py|jensen_corr\.py|metrics_all\.py|teacher_forced\.py|decode_latents\.py|profile_bottleneck\.py" \
         | grep -v pgrep | head -1)
  if [ -n "$line" ]; then
    pid=${line%% *}
    script=$(echo "$line" | grep -oE '[a-z_]+\.py' | head -1)
    tag=$(echo "$line" | grep -oE -- '--tag [A-Za-z0-9_]+' | head -1 | cut -d' ' -f2)
    [ -z "$tag" ] && tag=$(echo "$line" | grep -oE 'latents[A-Za-z0-9_]*/[A-Za-z0-9_]+' | head -1)
    et=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
    printf "RUN   %-24s %-28s  elapsed %s\n" "$script" "${tag:-?}" "${et:-?}"
  else
    printf "RUN   (idle — nothing running)\n"
  fi

  # --- current sweep ---
  local log latest
  log=$(ls -t $SF/results/*_sweep.log $LL/results/*_sweep.log 2>/dev/null | head -1)
  if [ -n "$log" ]; then
    echo
    echo "SWEEP $(basename "$log")"
    local done_n fail_n
    done_n=$(grep -cE "EXIT=0" "$log" 2>/dev/null)
    fail_n=$(grep -cE "EXIT=[1-9]" "$log" 2>/dev/null)
    printf "      done %s   failed %s%s\n" "$done_n" "$fail_n" \
           "$(grep -q ALL_DONE "$log" && echo '   ALL_DONE' || echo '')"
    [ "${fail_n:-0}" -gt 0 ] && grep -E "EXIT=[1-9]" "$log" | tail -3 | sed 's/^/      FAIL  /'
    tail -3 "$log" | sed 's/^/      /'
  fi

  # --- prompt progress of the newest latent dirs ---
  echo
  echo "PROMPTS"
  local d n
  for d in $(ls -dt $SF/results/latents_gate*/*/ $SF/results/latents63*/*/ $LL/results/latents_ll*/*/ 2>/dev/null | head -4); do
    n=$(ls -d "$d"prompt* 2>/dev/null | wc -l)
    printf "  %-34s %s\n" "$(basename "$(dirname "$d")")/$(basename "$d")" "$(bar "$n" 10 16)"
  done

  # --- session 4 gates ---
  echo
  echo "SESSION 4"
  gate_state "F0  self quantized?          " "$REPO/docs/gateF_self_quant.md"
  gate_state "E1  slot contamination       " "$SF/results/gateE/e1_A2_int4/slot_contamination.json"
  gate_state "E2  sink content override    " "$SF/results/gateE2c_metrics.csv"
  gate_state "F1  sigma^2 vs delta-mass    " "$SF/results/gateF/jensen_sigma.json"
  gate_state "F2  TUM correction           " "$SF/results/gateF2_metrics.csv"
  gate_state "G   collapse trajectory      " "$SF/results/gateG_collapse.csv"
  gate_state "4-1 KIVI residual window     " "$SF/results/kivi_rw/done"
  gate_state "4-2 126-frame rollout        " "$SF/results/long126/done"
  printf "  commits: %s\n" "$(git -C $REPO rev-list --count HEAD 2>/dev/null)"
}

if [ "$INTERVAL" = "0" ]; then render; exit 0; fi
while true; do clear 2>/dev/null; render; sleep "$INTERVAL"; done
