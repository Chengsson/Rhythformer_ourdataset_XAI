import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch_facexformer_4gpu as launcher


class FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.stdout = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


class LauncherTests(unittest.TestCase):
    def launcher_args(self, output_dir):
        return [
            "--input-root",
            "/input",
            "--output-dir",
            output_dir,
            "--facexformer-root",
            "/facexformer",
            "--checkpoint",
            "/checkpoint",
        ]

    def test_all_four_shards_must_succeed_before_merge_and_validate(self):
        processes = [FakeProcess(0) for _ in range(4)]
        completed = mock.Mock(returncode=0)
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(
            launcher.subprocess, "Popen", side_effect=processes
        ) as popen, mock.patch.object(
            launcher.subprocess, "run", return_value=completed
        ) as run:
            result = launcher.main(self.launcher_args(output_dir))

        self.assertEqual(result, 0)
        self.assertEqual(popen.call_count, 4)
        self.assertEqual(run.call_count, 2)
        self.assertIn("merge", run.call_args_list[0].args[0])
        self.assertIn("validate", run.call_args_list[1].args[0])

    def test_detection_and_skip_arguments_are_forwarded_to_every_shard(self):
        processes = [FakeProcess(0) for _ in range(4)]
        completed = mock.Mock(returncode=0)
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(
            launcher.subprocess, "Popen", side_effect=processes
        ) as popen, mock.patch.object(
            launcher.subprocess, "run", return_value=completed
        ):
            result = launcher.main(
                [
                    *self.launcher_args(output_dir),
                    "--min-detection-rate",
                    "0.8",
                    "--skip-session",
                    "subject381=low detection",
                    "--skip-session",
                    "subject402=corrupt H.264",
                ]
            )

        self.assertEqual(result, 0)
        for call in popen.call_args_list:
            command = call.args[0]
            self.assertEqual(command.count("--skip-session"), 2)
            self.assertEqual(
                command[command.index("--min-detection-rate") + 1], "0.8"
            )

    def test_failed_shard_terminates_other_shards(self):
        processes = [
            FakeProcess(1),
            FakeProcess(),
            FakeProcess(),
            FakeProcess(),
        ]
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(
            launcher.subprocess, "Popen", side_effect=processes
        ), mock.patch.object(launcher.subprocess, "run") as run:
            result = launcher.main(self.launcher_args(output_dir))

        self.assertEqual(result, 1)
        self.assertTrue(all(process.terminated for process in processes[1:]))
        run.assert_not_called()

    def test_interrupt_cleans_children_closes_logs_and_returns_nonzero(self):
        processes = [FakeProcess() for _ in range(4)]
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(
            launcher.subprocess, "Popen", side_effect=processes
        ) as popen, mock.patch.object(
            launcher, "wait_for_shards", side_effect=KeyboardInterrupt
        ), mock.patch.object(launcher.subprocess, "run") as run:
            result = launcher.main(self.launcher_args(output_dir))
            log_handles = [
                call.kwargs["stdout"] for call in popen.call_args_list
            ]

        self.assertNotEqual(result, 0)
        self.assertTrue(all(process.terminated for process in processes))
        self.assertTrue(all(handle.closed for handle in log_handles))
        run.assert_not_called()

    def test_failure_never_runs_merge_or_validate(self):
        processes = [FakeProcess(0), FakeProcess(2), FakeProcess(), FakeProcess()]
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(
            launcher.subprocess, "Popen", side_effect=processes
        ), mock.patch.object(launcher.subprocess, "run") as run:
            result = launcher.main(self.launcher_args(output_dir))

        self.assertEqual(result, 1)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
