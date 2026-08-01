#!/usr/bin/env bash

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
preprocess_dir=$(cd -- "${script_dir}/.." && pwd)
input_root=${INPUT_ROOT:-/mnt/adata/Non-contact Video Archive/Video}
output_root=${OUTPUT_ROOT:-/mnt/adata/OurDataset_RhythmFormer_FaceX_128}
facexformer_root=${FACEXFORMER_ROOT:-/home/nordlinglab/Louis/facex_preprocess/facexformer}
checkpoint=${FACEXFORMER_CHECKPOINT:-${facexformer_root}/ckpts/model.pt}
python_bin=${PYTHON_BIN:-/home/nordlinglab/miniconda3/envs/facexformer-preprocess/bin/python}
wait_log="${output_root}/gpu_wait_v3.log"

date -Is > "${output_root}/control_v3_start_time.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

consecutive_idle=0
while [[ "${consecutive_idle}" -lt 3 ]]; do
  mapfile -t used < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  idle=1
  if [[ "${#used[@]}" -ne 4 ]]; then
    idle=0
  else
    for value in "${used[@]}"; do
      value=${value//[[:space:]]/}
      if [[ ! "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -gt 1000 ]]; then
        idle=0
      fi
    done
  fi
  if [[ "${idle}" -eq 1 ]]; then
    consecutive_idle=$((consecutive_idle + 1))
  else
    consecutive_idle=0
  fi
  printf '%s memory_used_mib=%s consecutive_idle=%s\n' \
    "$(date -Is)" "${used[*]:-unavailable}" "${consecutive_idle}" \
    >> "${wait_log}"
  if [[ "${consecutive_idle}" -lt 3 ]]; then
    sleep 30
  fi
done

date -Is > "${output_root}/launcher_v3_start_time.txt"
"${python_bin}" \
  "${preprocess_dir}/launch_facexformer_4gpu.py" \
  --input-root "${input_root}" \
  --output-dir "${output_root}" \
  --facexformer-root "${facexformer_root}" \
  --checkpoint "${checkpoint}" \
  --gpus 0,1,2,3 \
  --batch-size 16 \
  --codec FFV1 \
  --min-rgb-correlation 0.99 \
  --min-detection-rate 0.80 \
  --skip-session \
    "subject381=operator exclusion: detection 0.5995 below 0.80" \
  --skip-session \
    "subject402=operator exclusion: corrupt H.264; 7552/7560 decodable" \
  >> "${output_root}/launcher_v3.log" 2>&1 &
launcher_pid=$!
printf '%s\n' "${launcher_pid}" > "${output_root}/launcher_v3.pid"

wait "${launcher_pid}"
launcher_status=$?
printf '%s\n' "${launcher_status}" > "${output_root}/launcher_v3.exit"
date -Is > "${output_root}/launcher_v3_end_time.txt"
exit "${launcher_status}"
