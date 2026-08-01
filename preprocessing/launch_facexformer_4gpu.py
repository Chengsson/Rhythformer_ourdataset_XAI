#!/usr/bin/env python3
"""Launch four independent FaceXFormer preprocessing processes."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


class LauncherInterrupted(Exception):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"received signal {signum}")


def terminate_processes(
    processes: list[subprocess.Popen],
    timeout: float = 10.0,
) -> None:
    running = [process for process in processes if process.poll() is None]
    for process in running:
        process.terminate()

    deadline = time.monotonic() + timeout
    for process in running:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in running:
        if process.poll() is None:
            process.wait()


def wait_for_shards(
    processes: list[subprocess.Popen],
    poll_interval: float = 0.1,
) -> list[int]:
    while True:
        return_codes = [process.poll() for process in processes]
        failed = [
            (index, code)
            for index, code in enumerate(return_codes)
            if code is not None and code != 0
        ]
        if failed:
            shard, code = failed[0]
            raise RuntimeError(f"shard {shard} exited with status {code}")
        if all(code == 0 for code in return_codes):
            return [int(code) for code in return_codes]
        time.sleep(poll_interval)


def _raise_for_signal(signum: int, _frame: object) -> None:
    raise LauncherInterrupted(signum)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--facexformer-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--codec", default="XVID")
    parser.add_argument("--min-rgb-correlation", type=float, default=0.99)
    parser.add_argument("--min-detection-rate", type=float, default=0.80)
    parser.add_argument("--subject", action="append")
    parser.add_argument("--activity", action="append")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--skip-session", action="append")
    args = parser.parse_args(argv)
    if not 0.0 <= args.min_detection_rate <= 1.0:
        parser.error("--min-detection-rate must be in [0, 1]")

    script = Path(__file__).with_name("facexformer_preprocess.py")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    processes = []
    logs = []
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, _raise_for_signal)
    try:
        for shard, gpu in enumerate(gpus):
            log_handle = (output / f"shard_{shard:02d}.log").open("a")
            logs.append(log_handle)
            command = [
                sys.executable,
                str(script),
                "run",
                "--input-root",
                args.input_root,
                "--output-dir",
                str(output),
                "--facexformer-root",
                args.facexformer_root,
                "--checkpoint",
                args.checkpoint,
                "--device",
                f"cuda:{gpu}",
                "--batch-size",
                str(args.batch_size),
                "--codec",
                args.codec,
                "--min-rgb-correlation",
                str(args.min_rgb_correlation),
                "--min-detection-rate",
                str(args.min_detection_rate),
                "--num-shards",
                str(len(gpus)),
                "--shard-index",
                str(shard),
            ]
            if args.max_frames is not None:
                command.extend(["--max-frames", str(args.max_frames)])
            for subject in args.subject or []:
                command.extend(["--subject", subject])
            for activity in args.activity or []:
                command.extend(["--activity", activity])
            for skip_session in args.skip_session or []:
                command.extend(["--skip-session", skip_session])
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            )

        wait_for_shards(processes)
        merge = subprocess.run(
            [
                sys.executable,
                str(script),
                "merge",
                "--output-dir",
                str(output),
                "--num-shards",
                str(len(gpus)),
            ],
            check=False,
        )
        if merge.returncode:
            return merge.returncode
        return subprocess.run(
            [
                sys.executable,
                str(script),
                "validate",
                "--output-dir",
                str(output),
                "--min-detection-rate",
                str(args.min_detection_rate),
                "--min-rgb-correlation",
                str(args.min_rgb_correlation),
            ],
            check=False,
        ).returncode
    except LauncherInterrupted as exc:
        print(f"Launcher interrupted by signal {exc.signum}", file=sys.stderr)
        return 128 + exc.signum
    except KeyboardInterrupt:
        print("Launcher interrupted", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"Shard failure: {exc}", file=sys.stderr)
        return 1
    finally:
        terminate_processes(processes)
        for handle in logs:
            handle.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
