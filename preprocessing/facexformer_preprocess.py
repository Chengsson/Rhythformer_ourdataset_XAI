#!/usr/bin/env python3
"""Convert camera1 recordings to UBFC-compatible FaceXFormer face crops."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


CCW_SUBJECTS = {"984", "660", "301", "717", "764"}
DEFAULT_FPS = 50.0
OUTPUT_SIZE = 128
MAPPING_FIELDS = (
    "session_id",
    "original_subject",
    "activity",
    "video_path",
    "align_path",
    "rotation",
    "start_frame",
    "ppg_row_offset",
    "num_frames",
    "fps",
)
QUALITY_FIELDS = (
    "session_id",
    "status",
    "reason",
    "total_frames",
    "detected_frames",
    "interpolated_frames",
    "detection_rate",
    "interpolation_rate",
    "temporal_rgb_correlation",
    "elapsed_seconds",
    "processing_fps",
)
SESSION_ID_PATTERN = re.compile(r"subject\d{3,}")


@dataclass(frozen=True)
class Session:
    session_id: str
    original_subject: str
    activity: str
    video_path: str
    align_path: str
    rotation: str
    start_frame: int
    ppg_row_offset: int
    num_frames: int
    fps: float


@dataclass
class Quality:
    session_id: str
    status: str
    reason: str
    total_frames: int
    detected_frames: int
    interpolated_frames: int
    detection_rate: float
    interpolation_rate: float
    temporal_rgb_correlation: float
    elapsed_seconds: float
    processing_fps: float


class DetectionRateError(RuntimeError):
    def __init__(self, total_frames: int, detected_frames: int, minimum: float):
        self.total_frames = total_frames
        self.detected_frames = detected_frames
        rate = detected_frames / total_frames if total_frames else float("nan")
        super().__init__(
            f"Detection rate {rate:.6f} is below {minimum:.6f}"
        )


def subject_sort_key(path: Path) -> tuple[int, int | str]:
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


def video_stem(video_path: Path, camera: int) -> str:
    suffix = f"_camera{camera}"
    stem = video_path.stem
    if not stem.lower().endswith(suffix.lower()):
        raise ValueError(f"Unexpected camera filename: {video_path.name}")
    return stem[: -len(suffix)]


def activity_from_stem(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].isdigit():
        activity_parts = []
        for part in parts[1:]:
            if part.isdigit() and len(part) == 8:
                break
            activity_parts.append(part)
        if activity_parts:
            return "_".join(activity_parts)
    return stem


def find_align_file(video_path: Path, camera: int) -> Path:
    stem = video_stem(video_path, camera)
    matches = sorted(
        path
        for path in video_path.parent.glob(f"{stem}*_align.csv")
        if not path.name.startswith("._")
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one align.csv for {video_path.name}, found {len(matches)}"
        )
    return matches[0]


def read_alignment(align_path: Path) -> tuple[int, np.ndarray]:
    frame = pd.read_csv(align_path, header=None)
    if frame.empty or frame.shape[1] <= 2:
        raise ValueError(f"Invalid align.csv: {align_path}")
    start_frame = int(float(frame.iloc[0, 0]))
    ppg = pd.to_numeric(frame.iloc[:, 2], errors="coerce").to_numpy(dtype=float)
    finite = np.flatnonzero(np.isfinite(ppg))
    if not finite.size:
        raise ValueError(f"No numeric PPG values: {align_path}")
    # Keep the complete synchronized prefix only; an internal NaN ends the interval.
    first_invalid = np.flatnonzero(~np.isfinite(ppg))
    end = int(first_invalid[0]) if first_invalid.size else len(ppg)
    return start_frame, ppg[:end]


def discover_sessions(
    input_root: Path,
    camera: int = 1,
    max_frames: int | None = None,
    subjects: set[str] | None = None,
) -> tuple[list[Session], list[dict[str, str]]]:
    sessions: list[Session] = []
    discovery_errors: list[dict[str, str]] = []
    subject_dirs = sorted(
        (path for path in input_root.iterdir() if path.is_dir()),
        key=subject_sort_key,
    )
    session_index = 0
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        if subjects and subject not in subjects:
            continue
        videos = sorted(
            path
            for path in subject_dir.iterdir()
            if path.is_file()
            and not path.name.startswith("._")
            and path.suffix.lower() == ".mp4"
            and path.stem.lower().endswith(f"_camera{camera}")
        )
        for video_path in videos:
            session_index += 1
            session_id = f"subject{session_index:03d}"
            stem = video_stem(video_path, camera)
            activity = activity_from_stem(stem)
            rotation = "ccw" if subject in CCW_SUBJECTS else "cw"
            try:
                align_path = find_align_file(video_path, camera)
                start_frame, ppg = read_alignment(align_path)
                cap = cv2.VideoCapture(str(video_path))
                video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                if video_frames <= 0 or start_frame < 0:
                    raise ValueError(
                        f"Invalid video length/start: {video_frames}/{start_frame}"
                    )
                length = min(video_frames - start_frame, len(ppg))
                if max_frames is not None:
                    length = min(length, max_frames)
                if length <= 0:
                    raise ValueError("No overlapping video/PPG frames")
                sessions.append(
                    Session(
                        session_id=session_id,
                        original_subject=subject,
                        activity=activity,
                        video_path=str(video_path),
                        align_path=str(align_path),
                        rotation=rotation,
                        start_frame=start_frame,
                        ppg_row_offset=0,
                        num_frames=length,
                        fps=fps if fps > 0 else DEFAULT_FPS,
                    )
                )
            except Exception as exc:
                discovery_errors.append(
                    {
                        "session_id": session_id,
                        "original_subject": subject,
                        "activity": activity,
                        "video_path": str(video_path),
                        "align_path": "",
                        "rotation": rotation,
                        "start_frame": "",
                        "ppg_row_offset": "",
                        "num_frames": "0",
                        "fps": "",
                        "reason": str(exc),
                    }
                )
    return sessions, discovery_errors


def shard_for_session_id(session_id: str, num_shards: int) -> int:
    return (int(session_id.removeprefix("subject")) - 1) % num_shards


def parse_skip_sessions(values: Sequence[str] | None) -> dict[str, str]:
    skips = {}
    for value in values or []:
        session_id, separator, reason = value.partition("=")
        if (
            not separator
            or not SESSION_ID_PATTERN.fullmatch(session_id)
            or int(session_id.removeprefix("subject")) == 0
            or not reason.strip()
        ):
            raise ValueError(
                "--skip-session must use SESSION=REASON with a canonical "
                "session ID such as subject001"
            )
        if session_id in skips:
            raise ValueError(f"Duplicate --skip-session: {session_id}")
        skips[session_id] = reason.strip()
    return skips


def rotate_frame(frame: np.ndarray, rotation: str) -> np.ndarray:
    code = (
        cv2.ROTATE_90_COUNTERCLOCKWISE
        if rotation == "ccw"
        else cv2.ROTATE_90_CLOCKWISE
    )
    return cv2.rotate(frame, code)


def landmarks_to_box(
    landmarks: np.ndarray,
    scale: float = 1.6,
    center_y_offset: float = 0.08,
) -> np.ndarray:
    points = np.asarray(landmarks, dtype=float).reshape(-1, 2)
    if points.shape[0] != 68 or not np.isfinite(points).all():
        raise ValueError("Expected 68 finite landmarks")
    low = points.min(axis=0)
    high = points.max(axis=0)
    side = scale * max(high[0] - low[0], high[1] - low[1])
    if side <= 0:
        raise ValueError("Degenerate landmarks")
    center = (low + high) / 2.0
    center[1] -= center_y_offset * side
    return np.array([center[0], center[1], side], dtype=float)


def interpolate_and_smooth_boxes(
    boxes: np.ndarray,
    median_window: int = 5,
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
) -> np.ndarray | None:
    values = np.asarray(boxes, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("boxes must have shape (frames, 3)")
    valid = np.isfinite(values).all(axis=1)
    if not valid.any():
        return None
    indices = np.arange(len(values))
    result = values.copy()
    for column in range(3):
        result[:, column] = np.interp(
            indices, indices[valid], values[valid, column]
        )

    median_size = _odd_window(median_window, len(result))
    if median_size >= 3:
        result = median_filter(result, size=(median_size, 1), mode="nearest")
    sg_window = _odd_window(savgol_window, len(result))
    if sg_window > savgol_polyorder:
        result = savgol_filter(
            result,
            window_length=sg_window,
            polyorder=min(savgol_polyorder, sg_window - 1),
            axis=0,
            mode="interp",
        )
    result[:, 2] = np.maximum(result[:, 2], 2.0)
    return result


def _odd_window(requested: int, length: int) -> int:
    value = min(max(1, requested), length)
    if value % 2 == 0:
        value -= 1
    return max(1, value)


def square_crop_bounds(
    box: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int]:
    cx, cy, side = map(float, box)
    side_i = max(2, min(int(round(side)), width, height))
    x0 = int(round(cx - side_i / 2))
    y0 = int(round(cy - side_i / 2))
    x0 = min(max(0, x0), width - side_i)
    y0 = min(max(0, y0), height - side_i)
    return x0, y0, x0 + side_i, y0 + side_i


def temporal_rgb_correlation(
    source_means: np.ndarray, decoded_means: np.ndarray
) -> float:
    source = np.asarray(source_means, dtype=float)
    decoded = np.asarray(decoded_means, dtype=float)
    if source.shape != decoded.shape or source.size == 0:
        return float("nan")
    correlations = []
    for channel in range(source.shape[1]):
        a, b = source[:, channel], decoded[:, channel]
        if np.std(a) == 0 or np.std(b) == 0:
            correlations.append(1.0 if np.allclose(a, b, atol=2.0) else 0.0)
        else:
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return float(min(correlations))


def select_best_face_box(
    boxes: np.ndarray | None,
    probabilities: np.ndarray | None,
) -> np.ndarray | None:
    if boxes is None or probabilities is None:
        return None
    try:
        numeric_boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        numeric_probabilities = np.asarray(
            probabilities, dtype=float
        ).reshape(-1)
    except (TypeError, ValueError):
        return None
    count = min(len(numeric_boxes), len(numeric_probabilities))
    if count == 0:
        return None
    valid = np.isfinite(numeric_probabilities[:count])
    valid &= np.isfinite(numeric_boxes[:count]).all(axis=1)
    valid_indices = np.flatnonzero(valid)
    if not valid_indices.size:
        return None
    selected = valid_indices[
        np.argmax(numeric_probabilities[valid_indices])
    ]
    return numeric_boxes[selected]


def validate_session_dir(
    session_dir: Path,
    expected_frames: int | None = None,
    expected_fps: float | None = None,
    size: int = OUTPUT_SIZE,
) -> tuple[bool, str, dict[str, float | int]]:
    video_path = session_dir / "vid.avi"
    gt_path = session_dir / "ground_truth.txt"
    if not video_path.is_file() or not gt_path.is_file():
        return False, "missing vid.avi or ground_truth.txt", {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "vid.avi cannot be opened", {}
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    decoded_frames = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded_frames += 1
    cap.release()
    try:
        gt_count = len(np.loadtxt(gt_path, ndmin=1))
    except Exception as exc:
        return False, f"ground truth cannot be read: {exc}", {}
    metrics = {
        "reported_frames": reported_frames,
        "decoded_frames": decoded_frames,
        "gt_count": gt_count,
        "width": width,
        "height": height,
        "fps": fps,
    }
    if width != size or height != size:
        return False, f"unexpected dimensions {width}x{height}", metrics
    if decoded_frames != gt_count or reported_frames != decoded_frames:
        return False, "video and GT frame counts differ", metrics
    if expected_frames is not None and decoded_frames != expected_frames:
        return False, f"expected {expected_frames} frames", metrics
    if expected_fps is not None and not math.isclose(
        fps, expected_fps, rel_tol=0.0, abs_tol=0.1
    ):
        return False, f"expected FPS {expected_fps}, got {fps}", metrics
    return True, "", metrics


class FaceXLandmarkDetector:
    def __init__(
        self,
        facexformer_root: Path,
        checkpoint: Path,
        device: str,
        batch_size: int,
    ) -> None:
        import torch
        import torchvision
        from facenet_pytorch import MTCNN
        from PIL import Image
        from torchvision.transforms import InterpolationMode

        self.torch = torch
        self.Image = Image
        self.device = torch.device(device)
        self.batch_size = batch_size
        sys.path.insert(0, str(facexformer_root))
        # The official constructor requests ImageNet weights before the FaceXFormer
        # checkpoint is loaded. Build the identical architecture without that
        # redundant download; the official checkpoint supplies all parameters.
        original_swin_b = torchvision.models.swin_b
        torchvision.models.swin_b = (
            lambda *args, **kwargs: original_swin_b(weights=None)
        )
        try:
            network = importlib.import_module("network")
            self.model = network.FaceXFormer().to(self.device)
        finally:
            torchvision.models.swin_b = original_swin_b
        payload = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(payload["state_dict_backbone"])
        self.model.eval()
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    size=(224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def detect(self, frames_bgr: Sequence[np.ndarray]) -> list[np.ndarray | None]:
        pending_images = []
        pending_meta: list[tuple[int, float, float, float, float]] = []
        result: list[np.ndarray | None] = [None] * len(frames_bgr)
        for index, frame in enumerate(frames_bgr):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self.Image.fromarray(rgb)
            boxes, probs = self.mtcnn.detect(image)
            selected_box = select_best_face_box(boxes, probs)
            if selected_box is None:
                continue
            x0, y0, x1, y1 = selected_box
            box_w, box_h = x1 - x0, y1 - y0
            x0 = max(0.0, x0 - 0.25 * box_w)
            y0 = max(0.0, y0 - 0.25 * box_h)
            x1 = min(float(image.width), x1 + 0.25 * box_w)
            y1 = min(float(image.height), y1 + 0.25 * box_h)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            face = image.crop((int(x0), int(y0), int(x1), int(y1)))
            pending_images.append(self.transform(face))
            pending_meta.append((index, x0, y0, x1, y1))

        for offset in range(0, len(pending_images), self.batch_size):
            tensors = self.torch.stack(
                pending_images[offset : offset + self.batch_size]
            ).to(self.device)
            count = tensors.shape[0]
            labels = {
                "segmentation": self.torch.zeros(
                    (count, 224, 224), device=self.device
                ),
                "lnm_seg": self.torch.zeros((count, 5, 2), device=self.device),
                "landmark": self.torch.zeros(
                    (count, 68, 2), device=self.device
                ),
                "headpose": self.torch.zeros((count, 3), device=self.device),
                "attribute": self.torch.zeros((count, 40), device=self.device),
                "a_g_e": self.torch.zeros((count, 3), device=self.device),
                "visibility": self.torch.zeros((count, 29), device=self.device),
            }
            tasks = self.torch.ones(
                count, dtype=self.torch.long, device=self.device
            )
            with self.torch.inference_mode():
                landmark_output = self.model(tensors, labels, tasks)[0]
            normalized = landmark_output.reshape(count, 68, 2)
            normalized = normalized.detach().cpu().numpy()
            for local_index, points in enumerate(normalized):
                frame_index, x0, y0, x1, y1 = pending_meta[
                    offset + local_index
                ]
                # Match the official align_corners=False denormalization.
                points = ((points + 1.0) * 224.0 - 1.0) / 2.0
                points[:, 0] = x0 + points[:, 0] * (x1 - x0) / 224.0
                points[:, 1] = y0 + points[:, 1] * (y1 - y0) / 224.0
                result[frame_index] = points
        return result


def detect_box_trajectory(
    session: Session,
    detector: FaceXLandmarkDetector,
    read_batch_size: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(session.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, session.start_frame)
    boxes = np.full((session.num_frames, 3), np.nan, dtype=float)
    position = 0
    while position < session.num_frames:
        frames = []
        for _ in range(min(read_batch_size, session.num_frames - position)):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(rotate_frame(frame, session.rotation))
        if not frames:
            break
        landmarks = detector.detect(frames)
        for local_index, points in enumerate(landmarks):
            if points is not None:
                boxes[position + local_index] = landmarks_to_box(points)
        position += len(frames)
    cap.release()
    return boxes[:position]


def write_session(
    session: Session,
    boxes: np.ndarray,
    output_dir: Path,
    codec: str,
) -> tuple[np.ndarray, int]:
    temp_dir = output_dir / f".{session.session_id}.tmp.{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    video_path = temp_dir / "vid.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*codec),
        session.fps,
        (OUTPUT_SIZE, OUTPUT_SIZE),
    )
    if not writer.isOpened():
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"Codec {codec} could not open {video_path}")

    cap = cv2.VideoCapture(session.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, session.start_frame)
    source_means = []
    written = 0
    for box in boxes:
        ok, frame = cap.read()
        if not ok:
            break
        frame = rotate_frame(frame, session.rotation)
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = square_crop_bounds(box, width, height)
        crop = cv2.resize(
            frame[y0:y1, x0:x1],
            (OUTPUT_SIZE, OUTPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        source_means.append(crop.mean(axis=(0, 1)))
        writer.write(crop)
        written += 1
    cap.release()
    writer.release()
    if written != len(boxes):
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"Decoded only {written}/{len(boxes)} source frames")

    _, ppg = read_alignment(Path(session.align_path))
    ppg = ppg[session.ppg_row_offset : session.ppg_row_offset + written]
    np.savetxt(temp_dir / "ground_truth.txt", ppg[None, :], fmt="%.10g")
    target = output_dir / session.session_id
    if target.exists():
        shutil.rmtree(target)
    os.replace(temp_dir, target)
    return np.asarray(source_means), written


def decoded_rgb_means(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    means = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        means.append(frame.mean(axis=(0, 1)))
    cap.release()
    return np.asarray(means)


def process_session(
    session: Session,
    detector: FaceXLandmarkDetector,
    output_dir: Path,
    codec: str,
    read_batch_size: int,
    median_window: int,
    savgol_window: int,
    savgol_polyorder: int,
    min_rgb_correlation: float,
    min_detection_rate: float,
    resume: bool,
) -> Quality:
    started = time.monotonic()
    target = output_dir / session.session_id
    if resume:
        valid, _, metrics = validate_session_dir(
            target, session.num_frames, session.fps
        )
        if valid:
            elapsed = time.monotonic() - started
            frames = int(metrics["decoded_frames"])
            saved_quality = {}
            quality_path = target / ".facex_quality.json"
            if quality_path.is_file():
                saved_quality = json.loads(quality_path.read_text())
            detection_rate = float(saved_quality.get("detection_rate", "nan"))
            if (
                not np.isfinite(detection_rate)
                or detection_rate < min_detection_rate
            ):
                raise DetectionRateError(
                    frames,
                    int(saved_quality.get("detected_frames", 0)),
                    min_detection_rate,
                )
            return Quality(
                session.session_id,
                "resumed",
                "",
                frames,
                int(saved_quality.get("detected_frames", 0)),
                int(saved_quality.get("interpolated_frames", 0)),
                detection_rate,
                float(saved_quality.get("interpolation_rate", "nan")),
                float(saved_quality.get("temporal_rgb_correlation", "nan")),
                elapsed,
                frames / max(elapsed, 1e-9),
            )

    raw_boxes = detect_box_trajectory(session, detector, read_batch_size)
    if len(raw_boxes) != session.num_frames:
        raise RuntimeError(
            f"Source decoded {len(raw_boxes)}/{session.num_frames} aligned frames"
        )
    valid_mask = np.isfinite(raw_boxes).all(axis=1)
    detected = int(valid_mask.sum())
    detection_rate = detected / session.num_frames
    if detection_rate < min_detection_rate:
        raise DetectionRateError(
            session.num_frames,
            detected,
            min_detection_rate,
        )
    boxes = interpolate_and_smooth_boxes(
        raw_boxes, median_window, savgol_window, savgol_polyorder
    )
    assert boxes is not None
    source_means, written = write_session(session, boxes, output_dir, codec)
    decoded_means = decoded_rgb_means(target / "vid.avi")
    correlation = temporal_rgb_correlation(source_means, decoded_means)
    valid, reason, _ = validate_session_dir(target, written, session.fps)
    if not valid:
        shutil.rmtree(target)
        raise RuntimeError(reason)
    if not np.isfinite(correlation) or correlation < min_rgb_correlation:
        shutil.rmtree(target)
        raise RuntimeError(
            f"Temporal RGB correlation {correlation:.6f} is below "
            f"{min_rgb_correlation:.6f}; change codec before the full run"
        )
    elapsed = time.monotonic() - started
    missing = session.num_frames - detected
    quality = Quality(
        session.session_id,
        "success",
        "",
        session.num_frames,
        detected,
        missing,
        detection_rate,
        missing / session.num_frames,
        correlation,
        elapsed,
        session.num_frames / max(elapsed, 1e-9),
    )
    (target / ".facex_quality.json").write_text(
        json.dumps(asdict(quality), indent=2, allow_nan=True)
    )
    return quality


def write_csv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def run_shard(args: argparse.Namespace) -> int:
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = set(args.subject) if args.subject else None
    sessions, discovery_errors = discover_sessions(
        input_root, args.camera, args.max_frames, subjects
    )
    activities = set(args.activity) if args.activity else None
    if activities:
        sessions = [
            session for session in sessions if session.activity in activities
        ]
        discovery_errors = [
            error for error in discovery_errors if error["activity"] in activities
        ]
    skip_reasons = parse_skip_sessions(args.skip_session)
    known_session_ids = {
        session.session_id for session in sessions
    } | {error["session_id"] for error in discovery_errors}
    unknown_skips = sorted(set(skip_reasons) - known_session_ids)
    if unknown_skips:
        raise ValueError(
            "Unknown --skip-session ID(s): " + ", ".join(unknown_skips)
        )
    selected = [
        session
        for session in sessions
        if shard_for_session_id(session.session_id, args.num_shards)
        == args.shard_index
    ]
    selected_skips = {
        session_id: reason
        for session_id, reason in skip_reasons.items()
        if shard_for_session_id(session_id, args.num_shards) == args.shard_index
    }
    runnable = [
        session for session in selected if session.session_id not in skip_reasons
    ]
    selected_errors = [
        error
        for error in discovery_errors
        if shard_for_session_id(error["session_id"], args.num_shards)
        == args.shard_index
        and error["session_id"] not in skip_reasons
    ]
    detector = None
    if runnable:
        detector = FaceXLandmarkDetector(
            Path(args.facexformer_root).resolve(),
            Path(args.checkpoint).resolve(),
            args.device,
            args.batch_size,
        )
    quality_rows = [
        {
            "session_id": error["session_id"],
            "status": "skipped",
            "reason": f"discovery/alignment: {error['reason']}",
            "total_frames": 0,
            "detected_frames": 0,
            "interpolated_frames": 0,
            "detection_rate": float("nan"),
            "interpolation_rate": float("nan"),
            "temporal_rgb_correlation": float("nan"),
            "elapsed_seconds": 0.0,
            "processing_fps": 0.0,
        }
        for error in selected_errors
    ]
    mapping_rows = [
        {field: error.get(field, "") for field in MAPPING_FIELDS}
        for error in selected_errors
    ]
    session_by_id = {session.session_id: session for session in sessions}
    error_by_id = {error["session_id"]: error for error in discovery_errors}
    for session_id, reason in selected_skips.items():
        total_frames = (
            session_by_id[session_id].num_frames
            if session_id in session_by_id
            else 0
        )
        quality_rows.append(
            asdict(
                Quality(
                    session_id,
                    "skipped",
                    reason,
                    total_frames,
                    0,
                    0,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    0.0,
                    0.0,
                )
            )
        )
        if session_id in session_by_id:
            mapping_rows.append(asdict(session_by_id[session_id]))
        else:
            mapping_rows.append(
                {
                    field: error_by_id[session_id].get(field, "")
                    for field in MAPPING_FIELDS
                }
            )
    for error in selected_errors:
        stale_target = output_dir / error["session_id"]
        if stale_target.exists():
            shutil.rmtree(stale_target)
    for session_id in selected_skips:
        stale_target = output_dir / session_id
        if stale_target.exists():
            shutil.rmtree(stale_target)
    for session in runnable:
        print(
            f"[{session.session_id}] {session.original_subject}/"
            f"{session.activity}: {session.num_frames} frames",
            flush=True,
        )
        try:
            assert detector is not None
            quality = process_session(
                session,
                detector,
                output_dir,
                args.codec,
                args.read_batch_size,
                args.median_window,
                args.savgol_window,
                args.savgol_polyorder,
                args.min_rgb_correlation,
                args.min_detection_rate,
                args.resume,
            )
        except Exception as exc:
            target = output_dir / session.session_id
            if target.exists():
                shutil.rmtree(target)
            total_frames = (
                exc.total_frames
                if isinstance(exc, DetectionRateError)
                else session.num_frames
            )
            detected_frames = (
                exc.detected_frames
                if isinstance(exc, DetectionRateError)
                else 0
            )
            quality = Quality(
                session.session_id,
                "failed",
                str(exc),
                total_frames,
                detected_frames,
                total_frames - detected_frames,
                detected_frames / total_frames if total_frames else float("nan"),
                (
                    (total_frames - detected_frames) / total_frames
                    if total_frames
                    else float("nan")
                ),
                float("nan"),
                0.0,
                0.0,
            )
        quality_rows.append(asdict(quality))
        mapping_rows.append(asdict(session))
        write_csv(
            output_dir / f"quality_report_shard_{args.shard_index:02d}.csv",
            quality_rows,
            QUALITY_FIELDS,
        )
        write_csv(
            output_dir / f"session_mapping_shard_{args.shard_index:02d}.csv",
            mapping_rows,
            MAPPING_FIELDS,
        )
        if quality.status == "failed":
            print(f"[{session.session_id}] FAILED: {quality.reason}", flush=True)
            if "correlation" in quality.reason:
                return 2
            return 1
        else:
            print(
                f"[{session.session_id}] {quality.status}: "
                f"detection={quality.detection_rate:.4f}, "
                f"interpolation={quality.interpolation_rate:.4f}",
                flush=True,
            )

    if selected_errors:
        error_path = output_dir / (
            f"discovery_errors_shard_{args.shard_index:02d}.json"
        )
        error_path.write_text(json.dumps(selected_errors, indent=2))
    write_csv(
        output_dir / f"quality_report_shard_{args.shard_index:02d}.csv",
        quality_rows,
        QUALITY_FIELDS,
    )
    write_csv(
        output_dir / f"session_mapping_shard_{args.shard_index:02d}.csv",
        mapping_rows,
        MAPPING_FIELDS,
    )
    return 1 if any(row["status"] == "failed" for row in quality_rows) else 0


def merge_shards(output_dir: Path, num_shards: int) -> int:
    mapping_frames = []
    quality_frames = []
    for shard in range(num_shards):
        mapping_path = output_dir / f"session_mapping_shard_{shard:02d}.csv"
        quality_path = output_dir / f"quality_report_shard_{shard:02d}.csv"
        if not mapping_path.is_file() or not quality_path.is_file():
            raise FileNotFoundError(f"Missing manifest for shard {shard}")
        mapping_frames.append(pd.read_csv(mapping_path))
        quality_frames.append(pd.read_csv(quality_path))
    mapping = pd.concat(mapping_frames, ignore_index=True).sort_values("session_id")
    quality = pd.concat(quality_frames, ignore_index=True).sort_values("session_id")
    if mapping["session_id"].duplicated().any():
        raise RuntimeError("Duplicate session IDs in shard manifests")
    write_csv(
        output_dir / "session_mapping.csv",
        mapping.to_dict(orient="records"),
        MAPPING_FIELDS,
    )
    write_csv(
        output_dir / "quality_report.csv",
        quality.to_dict(orient="records"),
        QUALITY_FIELDS,
    )
    print(
        f"Merged {len(mapping)} sessions: "
        f"{(quality.status == 'success').sum()} success, "
        f"{(quality.status == 'resumed').sum()} resumed, "
        f"{(quality.status == 'skipped').sum()} skipped, "
        f"{(quality.status == 'failed').sum()} failed"
    )
    return 1 if (quality.status == "failed").any() else 0


def validate_output(
    output_dir: Path,
    min_detection_rate: float = 0.80,
    min_rgb_correlation: float = 0.99,
) -> int:
    mapping = pd.read_csv(output_dir / "session_mapping.csv")
    quality = pd.read_csv(output_dir / "quality_report.csv")
    failures = []
    failed_rows = quality.loc[quality.status == "failed", ["session_id", "reason"]]
    failures.extend(
        f"{row.session_id}: preprocessing failed: {row.reason}"
        for row in failed_rows.itertuples(index=False)
    )
    successful = set(
        quality.loc[quality.status.isin(["success", "resumed"]), "session_id"]
    )
    successful_quality = quality.loc[
        quality.status.isin(["success", "resumed"])
    ]
    for row in successful_quality.itertuples(index=False):
        if (
            not np.isfinite(row.detection_rate)
            or row.detection_rate < min_detection_rate
        ):
            failures.append(
                f"{row.session_id}: detection rate {row.detection_rate} is below "
                f"{min_detection_rate}"
            )
        if (
            not np.isfinite(row.temporal_rgb_correlation)
            or row.temporal_rgb_correlation < min_rgb_correlation
        ):
            failures.append(
                f"{row.session_id}: temporal RGB correlation "
                f"{row.temporal_rgb_correlation} is below {min_rgb_correlation}"
            )
    total_frames = successful_quality.total_frames.sum()
    if total_frames:
        aggregate_detection = (
            successful_quality.detected_frames.sum() / total_frames
        )
        aggregate_interpolation = (
            successful_quality.interpolated_frames.sum() / total_frames
        )
        if aggregate_detection < 0.90:
            failures.append(
                f"aggregate detection rate {aggregate_detection} is below 0.9"
            )
        if aggregate_interpolation > 0.10:
            failures.append(
                "aggregate interpolation rate "
                f"{aggregate_interpolation} is above 0.1"
            )
    actual_dirs = {
        path.name
        for path in output_dir.glob("subject*")
        if path.is_dir()
    }
    temp_dirs = sorted(
        path.name for path in output_dir.glob(".subject*.tmp.*") if path.is_dir()
    )
    failures.extend(f"{name}: temporary directory remains" for name in temp_dirs)
    unexpected_dirs = actual_dirs - successful
    failures.extend(
        f"{session_id}: output directory exists for a non-successful session"
        for session_id in sorted(unexpected_dirs)
    )
    missing_dirs = successful - actual_dirs
    failures.extend(
        f"{session_id}: successful session directory is missing"
        for session_id in sorted(missing_dirs)
    )
    for row in mapping.itertuples(index=False):
        if row.session_id not in successful:
            continue
        valid, reason, _ = validate_session_dir(
            output_dir / row.session_id, int(row.num_frames), float(row.fps)
        )
        if not valid:
            failures.append(f"{row.session_id}: {reason}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Validated {len(successful)} successful session directories")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one preprocessing shard")
    run.add_argument("--input-root", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--facexformer-root", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--camera", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--read-batch-size", type=int, default=16)
    run.add_argument("--codec", default="XVID")
    run.add_argument("--median-window", type=int, default=5)
    run.add_argument("--savgol-window", type=int, default=11)
    run.add_argument("--savgol-polyorder", type=int, default=2)
    run.add_argument("--min-rgb-correlation", type=float, default=0.99)
    run.add_argument("--min-detection-rate", type=float, default=0.80)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--max-frames", type=int)
    run.add_argument("--subject", action="append")
    run.add_argument("--activity", action="append")
    run.add_argument("--skip-session", action="append")
    run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    merge = subparsers.add_parser("merge", help="merge shard manifests")
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--num-shards", type=int, required=True)

    validate = subparsers.add_parser("validate", help="validate merged output")
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--min-detection-rate", type=float, default=0.80)
    validate.add_argument("--min-rgb-correlation", type=float, default=0.99)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard-index must be in [0, num-shards)")
        if not 0.0 <= args.min_detection_rate <= 1.0:
            raise ValueError("min-detection-rate must be in [0, 1]")
        return run_shard(args)
    if args.command == "merge":
        return merge_shards(Path(args.output_dir).resolve(), args.num_shards)
    return validate_output(
        Path(args.output_dir).resolve(),
        args.min_detection_rate,
        args.min_rgb_correlation,
    )


if __name__ == "__main__":
    raise SystemExit(main())
