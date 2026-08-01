#!/usr/bin/env python3
"""Send exactly one attempt-3 ntfy notification after postflight."""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path


OUTPUT = Path(
    os.environ.get(
        "OUTPUT_ROOT", "/mnt/adata/OurDataset_RhythmFormer_FaceX_128"
    )
)
NTFY_TOPIC = os.environ.get("NTFY_TOPIC")
EXIT_PATH = OUTPUT / "postflight_v3.exit"
MARKER_PATH = OUTPUT / ".ntfy_attempt_03_sent"


def quality_summary() -> str:
    rows = []
    report = OUTPUT / "quality_report.csv"
    if report.is_file():
        with report.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    counts = Counter(row["status"] for row in rows)
    return (
        f"rows={len(rows)}, success={counts['success']}, "
        f"resumed={counts['resumed']}, skipped={counts['skipped']}, "
        f"failed={counts['failed']}"
    )


while not EXIT_PATH.is_file():
    time.sleep(30)
if MARKER_PATH.is_file():
    raise SystemExit(0)
if not NTFY_TOPIC:
    raise SystemExit("NTFY_TOPIC is required")

exit_code = int(EXIT_PATH.read_text().strip())
completed_at = datetime.now().astimezone().isoformat()
result = "completed successfully" if exit_code == 0 else "failed"
summary = quality_summary()
message = (
    f"FaceXFormer alignment recovery attempt 3 {result}. "
    f"exit_code={exit_code}; {summary}; completed_at={completed_at}"
)
payload = json.dumps(
    {
        "topic": NTFY_TOPIC,
        "title": "FaceXFormer alignment recovery attempt 3 finished",
        "message": message,
        "priority": 4 if exit_code == 0 else 5,
        "tags": ["white_check_mark" if exit_code == 0 else "x"],
    }
).encode()

while True:
    try:
        request = urllib.request.Request(
            "https://ntfy.sh",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
        MARKER_PATH.write_text(
            f"sent_at={completed_at}\nexit_code={exit_code}\n{summary}\n"
        )
        raise SystemExit(0)
    except (OSError, urllib.error.URLError):
        time.sleep(60)
