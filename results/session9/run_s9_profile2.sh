#!/usr/bin/env bash
# 2부: 1부가 끝난 뒤 커널 스윕 (GPU 단독 사용).
set -u; cd /vfsdata/gpu_data/gpu/Self-Forcing; O=results/session9/profile
while ! grep -q PROFILE1_DONE $O/run.status 2>/dev/null; do sleep 30; done
echo "START2 $(date '+%F %T')" >> $O/run.status
B9_OUT=$O/b9_kernel_sweep.json /vfsdata/gpu_data/gpu/sf-venv/bin/python kernels/bench/b9_kernel_sweep.py > $O/run2.log 2>&1; echo "b9 EXIT=$? $(date +%T)" >> $O/run.status
echo "PROFILE2_DONE $(date '+%F %T')" >> $O/run.status
