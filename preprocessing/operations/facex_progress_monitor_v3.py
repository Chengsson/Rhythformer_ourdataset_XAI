#!/usr/bin/env python3
"""Record attempt-3 shard, error, GPU, and disk status every 15 minutes."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


output = Path(sys.argv[1])
pid_path = Path(sys.argv[2])
log_path = Path(sys.argv[3])
data_mount = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("/mnt/adata")


def launcher_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def snapshot() -> None:
    counts: Counter[str] = Counter()
    rows = 0
    for report_path in sorted(output.glob("quality_report_shard_*.csv")):
        with report_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                counts[row["status"]] += 1

    alerts = []
    for shard_log in sorted(output.glob("shard_*.log")):
        text = shard_log.read_text(errors="replace")
        if "FAILED" in text:
            alerts.append(shard_log.name)

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip().replace("\n", " | ")
    free_bytes = shutil.disk_usage(data_mount).free
    timestamp = datetime.now().astimezone().isoformat()
    with log_path.open("a") as handle:
        handle.write(
            f"{timestamp} rows={rows} statuses={dict(counts)} "
            f"alerts={alerts} free_bytes={free_bytes} gpu=[{gpu}]\n"
        )


while not pid_path.is_file():
    time.sleep(1)

launcher_pid = int(pid_path.read_text().strip())
while launcher_running(launcher_pid):
    snapshot()
    time.sleep(900)
snapshot()
