#!/usr/bin/env python3
"""Log output-disk capacity and stop attempt 3 at 50 GB free."""

from __future__ import annotations

import os
import signal
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


mount = Path(sys.argv[1])
pid_path = Path(sys.argv[2])
log_path = Path(sys.argv[3])
warning_bytes = 80_000_000_000
stop_bytes = 50_000_000_000


def log(message: str) -> None:
    with log_path.open("a") as handle:
        handle.write(f"{datetime.now().astimezone().isoformat()} {message}\n")


while not pid_path.is_file():
    time.sleep(1)

launcher_pid = int(pid_path.read_text().strip())
while True:
    try:
        os.kill(launcher_pid, 0)
    except ProcessLookupError:
        log("launcher exited")
        raise SystemExit(0)

    free_bytes = shutil.disk_usage(mount).free
    log(f"free_bytes={free_bytes}")
    if free_bytes <= stop_bytes:
        log(f"STOP free_bytes<={stop_bytes}; sending SIGTERM pid={launcher_pid}")
        os.kill(launcher_pid, signal.SIGTERM)
        raise SystemExit(50)
    if free_bytes <= warning_bytes:
        log(f"WARNING free_bytes<={warning_bytes}")
    time.sleep(30)
