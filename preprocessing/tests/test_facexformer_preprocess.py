import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import facexformer_preprocess as fp


class FaceXFormerPreprocessTests(unittest.TestCase):
    @staticmethod
    def session(session_id="subject001", num_frames=10):
        return fp.Session(
            session_id=session_id,
            original_subject="11",
            activity="rotate",
            video_path="/source.mp4",
            align_path="/source_align.csv",
            rotation="cw",
            start_frame=0,
            ppg_row_offset=0,
            num_frames=num_frames,
            fps=50,
        )

    @staticmethod
    def shard_args(output_dir, *extra):
        return fp.build_parser().parse_args(
            [
                "run",
                "--input-root",
                "/input",
                "--output-dir",
                output_dir,
                "--facexformer-root",
                "/facexformer",
                "--checkpoint",
                "/checkpoint",
                *extra,
            ]
        )

    def test_activity_omits_subject_and_timestamp(self):
        self.assertEqual(
            fp.activity_from_stem("11_static_level3_20230921_015418"),
            "static_level3",
        )

    def test_sharding_uses_stable_numeric_session_id(self):
        self.assertEqual(fp.shard_for_session_id("subject001", 4), 0)
        self.assertEqual(fp.shard_for_session_id("subject004", 4), 3)
        self.assertEqual(fp.shard_for_session_id("subject005", 4), 0)

    def test_skip_parser_preserves_reason_and_rejects_malformed_values(self):
        self.assertEqual(
            fp.parse_skip_sessions(
                ["subject381=operator exclusion: detection=0.5995"]
            ),
            {"subject381": "operator exclusion: detection=0.5995"},
        )
        for value in ("subject381", "381=reason", "subject1=reason", "subject002="):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fp.parse_skip_sessions([value])

    def test_explicit_skip_preserves_mapping_and_does_not_initialize_detector(self):
        session = self.session("subject381")
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(
            fp, "discover_sessions", return_value=([session], [])
        ), mock.patch.object(fp, "FaceXLandmarkDetector") as detector:
            result = fp.run_shard(
                self.shard_args(
                    output_dir,
                    "--skip-session",
                    "subject381=operator exclusion",
                )
            )

            detector.assert_not_called()
            self.assertEqual(result, 0)
            self.assertFalse((Path(output_dir) / "subject381").exists())
            with (
                Path(output_dir) / "quality_report_shard_00.csv"
            ).open() as handle:
                quality = list(csv.DictReader(handle))
            with (
                Path(output_dir) / "session_mapping_shard_00.csv"
            ).open() as handle:
                mapping = list(csv.DictReader(handle))
            self.assertEqual(
                (quality[0]["session_id"], quality[0]["status"], quality[0]["reason"]),
                ("subject381", "skipped", "operator exclusion"),
            )
            self.assertEqual(
                (mapping[0]["session_id"], mapping[0]["original_subject"]),
                ("subject381", "11"),
            )

    def test_unknown_explicit_skip_fails_before_detector_initialization(self):
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(
            fp, "discover_sessions", return_value=([self.session()], [])
        ), mock.patch.object(fp, "FaceXLandmarkDetector") as detector:
            with self.assertRaisesRegex(ValueError, "subject402"):
                fp.run_shard(
                    self.shard_args(
                        output_dir,
                        "--skip-session",
                        "subject402=operator exclusion",
                    )
                )
            detector.assert_not_called()

    def test_preprocessing_exception_writes_manifest_and_stops_shard(self):
        sessions = [self.session("subject001"), self.session("subject002")]
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(
            fp, "discover_sessions", return_value=(sessions, [])
        ), mock.patch.object(
            fp, "FaceXLandmarkDetector", return_value=mock.Mock()
        ), mock.patch.object(
            fp, "process_session", side_effect=RuntimeError("decode failed")
        ) as process:
            result = fp.run_shard(self.shard_args(output_dir))

            self.assertEqual(result, 1)
            self.assertEqual(process.call_count, 1)
            with (
                Path(output_dir) / "quality_report_shard_00.csv"
            ).open() as handle:
                quality = list(csv.DictReader(handle))
            self.assertEqual(len(quality), 1)
            self.assertEqual(quality[0]["status"], "failed")
            self.assertEqual(quality[0]["reason"], "decode failed")

    def test_detection_gate_stops_new_session_before_output(self):
        raw_boxes = np.ones((10, 3), dtype=float)
        raw_boxes[7:] = np.nan
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(
            fp, "detect_box_trajectory", return_value=raw_boxes
        ), mock.patch.object(fp, "write_session") as write:
            with self.assertRaisesRegex(RuntimeError, "0.700000"):
                fp.process_session(
                    self.session(),
                    mock.Mock(),
                    Path(output_dir),
                    codec="FFV1",
                    read_batch_size=4,
                    median_window=5,
                    savgol_window=11,
                    savgol_polyorder=2,
                    min_rgb_correlation=0.99,
                    min_detection_rate=0.80,
                    resume=False,
                )
            write.assert_not_called()

    def test_detection_gate_applies_to_resumed_session(self):
        with tempfile.TemporaryDirectory() as output_dir:
            target = Path(output_dir) / "subject001"
            target.mkdir()
            (target / ".facex_quality.json").write_text(
                json.dumps({"detection_rate": 0.79, "detected_frames": 79})
            )
            with mock.patch.object(
                fp,
                "validate_session_dir",
                return_value=(True, "", {"decoded_frames": 100}),
            ), self.assertRaisesRegex(RuntimeError, "0.790000"):
                fp.process_session(
                    self.session(num_frames=100),
                    mock.Mock(),
                    Path(output_dir),
                    codec="FFV1",
                    read_batch_size=4,
                    median_window=5,
                    savgol_window=11,
                    savgol_polyorder=2,
                    min_rgb_correlation=0.99,
                    min_detection_rate=0.80,
                    resume=True,
                )

    def test_landmarks_to_box_formula(self):
        points = np.zeros((68, 2), dtype=float)
        points[:, 0] = np.linspace(10, 110, 68)
        points[:, 1] = np.linspace(20, 70, 68)
        cx, cy, side = fp.landmarks_to_box(points)
        self.assertAlmostEqual(side, 160.0)
        self.assertAlmostEqual(cx, 60.0)
        self.assertAlmostEqual(cy, 32.2)

    def test_missing_boxes_are_interpolated_then_smoothed(self):
        boxes = np.array(
            [
                [10.0, 20.0, 30.0],
                [np.nan, np.nan, np.nan],
                [14.0, 24.0, 34.0],
            ]
        )
        result = fp.interpolate_and_smooth_boxes(
            boxes, median_window=1, savgol_window=1
        )
        np.testing.assert_allclose(result[1], [12.0, 22.0, 32.0])

    def test_all_missing_boxes_return_none(self):
        boxes = np.full((4, 3), np.nan)
        self.assertIsNone(fp.interpolate_and_smooth_boxes(boxes))

    def test_square_bounds_shift_inside_image_without_losing_shape(self):
        x0, y0, x1, y1 = fp.square_crop_bounds([2, 3, 20], 100, 80)
        self.assertEqual((x0, y0), (0, 0))
        self.assertEqual(x1 - x0, y1 - y0)
        self.assertEqual(x1 - x0, 20)

    def test_temporal_rgb_correlation_uses_worst_channel(self):
        source = np.arange(30, dtype=float).reshape(10, 3)
        decoded = source + 1
        self.assertAlmostEqual(
            fp.temporal_rgb_correlation(source, decoded), 1.0
        )

    def test_best_face_box_accepts_object_dtype_from_mtcnn(self):
        boxes = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            dtype=object,
        )
        probabilities = np.array([0.8, 0.9], dtype=object)
        np.testing.assert_allclose(
            fp.select_best_face_box(boxes, probabilities),
            [5.0, 6.0, 7.0, 8.0],
        )

    def test_validate_session_checks_decoded_frames_and_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory)
            writer = cv2.VideoWriter(
                str(session_dir / "vid.avi"),
                cv2.VideoWriter_fourcc(*"MJPG"),
                50,
                (128, 128),
            )
            for value in range(3):
                writer.write(np.full((128, 128, 3), value, dtype=np.uint8))
            writer.release()
            (session_dir / "ground_truth.txt").write_text("1 2 3\n")
            valid, reason, metrics = fp.validate_session_dir(
                session_dir, expected_frames=3, expected_fps=50
            )
            self.assertTrue(valid, reason)
            self.assertEqual(metrics["decoded_frames"], 3)

    def test_end_to_end_writer_with_stub_landmarks(self):
        class StubDetector:
            @staticmethod
            def detect(frames):
                result = []
                for _ in frames:
                    points = np.empty((68, 2), dtype=float)
                    points[:, 0] = np.linspace(30, 90, 68)
                    points[:, 1] = np.linspace(35, 95, 68)
                    result.append(points)
                return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.avi"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"MJPG"),
                50,
                (160, 120),
            )
            for index in range(14):
                frame = np.empty((120, 160, 3), dtype=np.uint8)
                frame[:, :, 0] = 20 + index * 3
                frame[:, :, 1] = 40 + index * 4
                frame[:, :, 2] = 60 + index * 5
                writer.write(frame)
            writer.release()
            align = root / "source_align.csv"
            rows = np.column_stack(
                [np.r_[2, np.zeros(19)], np.zeros(20), np.arange(20)]
            )
            np.savetxt(align, rows, delimiter=",")
            session = fp.Session(
                session_id="subject001",
                original_subject="11",
                activity="rotate",
                video_path=str(source),
                align_path=str(align),
                rotation="cw",
                start_frame=2,
                ppg_row_offset=0,
                num_frames=12,
                fps=50,
            )
            output = root / "output"
            output.mkdir()
            quality = fp.process_session(
                session,
                StubDetector(),
                output,
                codec="MJPG",
                read_batch_size=4,
                median_window=5,
                savgol_window=11,
                savgol_polyorder=2,
                min_rgb_correlation=0.99,
                min_detection_rate=0.80,
                resume=False,
            )
            self.assertEqual(quality.status, "success")
            self.assertGreaterEqual(quality.temporal_rgb_correlation, 0.99)
            valid, reason, metrics = fp.validate_session_dir(
                output / "subject001", 12, 50
            )
            self.assertTrue(valid, reason)
            self.assertEqual(metrics["gt_count"], 12)
            fp.write_csv(
                output / "session_mapping_shard_00.csv",
                [fp.asdict(session)],
                fp.MAPPING_FIELDS,
            )
            fp.write_csv(
                output / "quality_report_shard_00.csv",
                [fp.asdict(quality)],
                fp.QUALITY_FIELDS,
            )
            self.assertEqual(fp.merge_shards(output, 1), 0)
            self.assertEqual(fp.validate_output(output), 0)


if __name__ == "__main__":
    unittest.main()
