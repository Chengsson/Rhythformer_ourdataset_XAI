#!/usr/bin/env python3
"""Render contact sheets for the ten alignment-recovered sessions."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np


output = Path(sys.argv[1])
review_dir = output / "visual_review_v3_recovered"
review_dir.mkdir(exist_ok=False)
session_ids = [
    "subject033",
    "subject097",
    "subject121",
    "subject122",
    "subject123",
    "subject124",
    "subject125",
    "subject126",
    "subject127",
    "subject146",
]

with (output / "session_mapping.csv").open(newline="") as handle:
    mapping = {row["session_id"]: row for row in csv.DictReader(handle)}
with (output / "quality_report.csv").open(newline="") as handle:
    quality = {row["session_id"]: row for row in csv.DictReader(handle)}

records = []
for session_id in session_ids:
    video_path = output / session_id / "vid.avi"
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for fraction in (0.1, 0.5, 0.9):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((frame_count - 1) * fraction))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode review frame for {session_id}")
        frames.append(frame)
    cap.release()
    sheet = np.concatenate(frames, axis=1)
    label = (
        f"{session_id} original={mapping[session_id]['original_subject']} "
        f"activity={mapping[session_id]['activity']}"
    )
    cv2.putText(
        sheet,
        label,
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    image_path = review_dir / f"{session_id}.png"
    if not cv2.imwrite(str(image_path), sheet):
        raise RuntimeError(f"Cannot write {image_path}")
    records.append(
        {
            "session_id": session_id,
            "original_subject": mapping[session_id]["original_subject"],
            "activity": mapping[session_id]["activity"],
            "detection_rate": quality[session_id]["detection_rate"],
            "interpolation_rate": quality[session_id]["interpolation_rate"],
            "temporal_rgb_correlation": quality[session_id][
                "temporal_rgb_correlation"
            ],
            "contact_sheet": str(image_path),
            "visual_result": "pending_manual_review",
        }
    )

with (review_dir / "candidates.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
print(f"Rendered {len(records)} recovered-session contact sheets")
