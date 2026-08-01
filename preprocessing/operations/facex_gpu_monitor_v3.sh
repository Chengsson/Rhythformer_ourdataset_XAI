#!/usr/bin/env bash

set -u

pid_path=$1
log_path=$2

while [[ ! -f "${pid_path}" ]]; do
  sleep 1
done

launcher_pid=$(<"${pid_path}")
while kill -0 "${launcher_pid}" 2>/dev/null; do
  {
    date -Is
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,utilization.gpu,temperature.gpu \
      --format=csv,noheader
  } >> "${log_path}"
  sleep 30
done
