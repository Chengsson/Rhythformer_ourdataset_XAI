#!/usr/bin/env bash

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
preprocess_dir=$(cd -- "${script_dir}/.." && pwd)
input_root=${INPUT_ROOT:-/mnt/adata/Non-contact Video Archive/Video}
output_root=${OUTPUT_ROOT:-/mnt/adata/OurDataset_RhythmFormer_FaceX_128}
facexformer_root=${FACEXFORMER_ROOT:-/home/nordlinglab/Louis/facex_preprocess/facexformer}
checkpoint=${FACEXFORMER_CHECKPOINT:-${facexformer_root}/ckpts/model.pt}
recovery="${output_root}/attempt_03_recovery"
python_bin=${PYTHON_BIN:-/home/nordlinglab/miniconda3/envs/facexformer-preprocess/bin/python}
script="${preprocess_dir}/facexformer_preprocess.py"
launcher="${preprocess_dir}/launch_facexformer_4gpu.py"
log="${output_root}/postflight_v3.log"

while [[ ! -f "${output_root}/launcher_v3.exit" ]]; do
  sleep 30
done

launcher_status=$(<"${output_root}/launcher_v3.exit")
if [[ "${launcher_status}" -ne 0 ]]; then
  printf 'launcher_v3 exit=%s; postflight skipped\n' "${launcher_status}" \
    >> "${log}"
  printf '%s\n' "${launcher_status}" > "${output_root}/postflight_v3.exit"
  exit "${launcher_status}"
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

"${python_bin}" "${script}" validate \
  --output-dir "${output_root}" \
  --min-detection-rate 0.80 \
  --min-rgb-correlation 0.99 \
  >> "${log}" 2>&1
status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${output_root}/postflight_v3.exit"
  exit "${status}"
fi

formal_dirs=$(find "${output_root}" -maxdepth 1 -type d -name 'subject*' | wc -l)
temp_dirs=$(find "${output_root}" -maxdepth 1 -type d -name '.subject*.tmp.*' | wc -l)
if [[ "${formal_dirs}" -ne 607 || "${temp_dirs}" -ne 0 ]]; then
  printf 'unexpected directories: formal=%s temp=%s\n' \
    "${formal_dirs}" "${temp_dirs}" >> "${log}"
  printf '1\n' > "${output_root}/postflight_v3.exit"
  exit 1
fi

"${python_bin}" - "${output_root}" <<'PY' >> "${log}" 2>&1
import csv
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
recovered = {
    "subject033", "subject097", "subject121", "subject122", "subject123",
    "subject124", "subject125", "subject126", "subject127", "subject146",
}
with (root / "quality_report.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
counts = Counter(row["status"] for row in rows)
assert len(rows) == 618, (len(rows), counts)
assert counts == {"resumed": 597, "skipped": 11, "success": 10}, counts
by_id = {row["session_id"]: row for row in rows}
for session_id in recovered:
    row = by_id[session_id]
    assert row["status"] == "success", row
    assert float(row["detection_rate"]) >= 0.80, row
    assert float(row["temporal_rgb_correlation"]) >= 0.99, row
print(f"initial run passed: rows={len(rows)} statuses={dict(counts)}")
PY
status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${output_root}/postflight_v3.exit"
  exit "${status}"
fi

find "${output_root}" -maxdepth 2 -type f \
  \( -path '*/subject*/vid.avi' \
     -o -path '*/subject*/ground_truth.txt' \
     -o -path '*/subject*/.facex_quality.json' \) \
  -printf '%P\t%s\t%T@\n' \
  | grep -Ev '^subject(033|097|121|122|123|124|125|126|127|146)/' \
  | LC_ALL=C sort > "${recovery}/immutable_597_after_processing.tsv"
if ! cmp -s \
  "${recovery}/immutable_597_before.tsv" \
  "${recovery}/immutable_597_after_processing.tsv"; then
  printf 'existing 597 formal outputs changed during processing\n' >> "${log}"
  printf '1\n' > "${output_root}/postflight_v3.exit"
  exit 1
fi

find "${output_root}" -maxdepth 2 -type f \
  \( -path '*/subject*/vid.avi' \
     -o -path '*/subject*/ground_truth.txt' \
     -o -path '*/subject*/.facex_quality.json' \) \
  -printf '%P\t%s\t%T@\n' | LC_ALL=C sort \
  > "${recovery}/formal_607_before_resume_drill.tsv"

"${python_bin}" "${launcher}" \
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
  > "${output_root}/resume_drill_v3.log" 2>&1
status=$?
if [[ "${status}" -ne 0 ]]; then
  printf 'resume drill exit=%s\n' "${status}" >> "${log}"
  printf '%s\n' "${status}" > "${output_root}/postflight_v3.exit"
  exit "${status}"
fi

find "${output_root}" -maxdepth 2 -type f \
  \( -path '*/subject*/vid.avi' \
     -o -path '*/subject*/ground_truth.txt' \
     -o -path '*/subject*/.facex_quality.json' \) \
  -printf '%P\t%s\t%T@\n' | LC_ALL=C sort \
  > "${recovery}/formal_607_after_resume_drill.tsv"
if ! cmp -s \
  "${recovery}/formal_607_before_resume_drill.tsv" \
  "${recovery}/formal_607_after_resume_drill.tsv"; then
  printf 'resume drill changed formal output files\n' >> "${log}"
  printf '1\n' > "${output_root}/postflight_v3.exit"
  exit 1
fi

"${python_bin}" - "${output_root}" <<'PY' >> "${log}" 2>&1
import csv
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
with (root / "quality_report.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
counts = Counter(row["status"] for row in rows)
assert len(rows) == 618, (len(rows), counts)
assert counts == {"resumed": 607, "skipped": 11}, counts
print(f"resume drill passed: rows={len(rows)} statuses={dict(counts)}")
PY
status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${output_root}/postflight_v3.exit"
  exit "${status}"
fi

"${python_bin}" "${output_root}/generate_recovered_contact_sheets_v3.py" \
  "${output_root}" >> "${log}" 2>&1
status=$?
printf '%s\n' "${status}" > "${output_root}/postflight_v3.exit"
exit "${status}"
