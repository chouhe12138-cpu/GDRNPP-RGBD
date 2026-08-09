#!/usr/bin/env python3
"""Run one prepared training attempt and update its immutable metadata."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import traceback
from pathlib import Path

import torch

from research.experiment_system.artifacts import atomic_write_json, utc_now
from research.experiment_system.checkpoints import record_checkpoint
from research.experiment_system.logs import compact_and_write_warning_summary
from research.experiment_system.manifest import (
    manifest_environment_image_id,
    manifest_source_commit,
    read_json,
    sha256_file,
)
from research.experiment_system.steps import register_step, transition_step


OOM_RE = re.compile(
    r"CUDA out of memory|CUDNN_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "audit", "formal"), required=True)
    parser.add_argument("--isolation-role", choices=("B", "CPM"), required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--cuda-device", default="0")
    return parser.parse_args()


def checkpoint_for_mode(run_dir: Path, mode: str) -> tuple[Path, str, int, str]:
    if mode == "formal":
        return (
            run_dir / "checkpoints/model_epoch_040.pth",
            "epoch_040",
            40,
            "fixed_final",
        )
    kind = "smoke" if mode == "smoke" else "recent"
    return run_dir / "checkpoints/model_epoch_001.pth", "epoch_001", 1, kind


def checkpoint_iteration(path: Path) -> int:
    payload = torch.load(path, map_location="cpu")
    return int(payload.get("iteration", -1)) if isinstance(payload, dict) else -1


def detect_failure_kind(console_log: Path) -> str:
    tail_size = min(console_log.stat().st_size, 2 * 1024 * 1024)
    with console_log.open("rb") as handle:
        handle.seek(-tail_size, os.SEEK_END)
        tail = handle.read().decode("utf-8", errors="replace")
    return "CUDA_OOM" if OOM_RE.search(tail) else "RUNTIME_ERROR"


def write_status(run_dir: Path, **values: object) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": utc_now(),
        **values,
    }
    atomic_write_json(run_dir / "meta/launcher_status.json", payload)


def append_console(console_log: Path, message: str) -> None:
    with console_log.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def complete_supporting_step(run_dir: Path, step_id: str, kind: str, message: str) -> None:
    register_step(run_dir, step_id, kind, message)
    transition_step(run_dir, step_id, "RUNNING", message)
    transition_step(run_dir, step_id, "COMPLETE", message)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = args.config.resolve()
    official = args.official.resolve()
    console_log = run_dir / "train/console.log"
    manifest = read_json(run_dir / "meta/run_manifest.json")
    if int(manifest["seed"]) != 42:
        raise RuntimeError(f"managed run seed must be 42, got {manifest['seed']}")
    command = [
        str(Path(__file__).resolve().parents[2] / "core/gdrn_modeling/train_gdrn.sh"),
        str(config),
        str(args.cuda_device),
    ]
    command_text = " ".join(command)
    write_status(
        run_dir,
        state="RUNNING",
        mode=args.mode,
        exit_code=None,
        failure_kind=None,
    )
    register_step(
        run_dir,
        "train",
        "train",
        command_text,
        inputs=[str(config), str(official)],
        outputs=["train/metrics.jsonl", "checkpoints/"],
    )
    transition_step(run_dir, "train", "RUNNING", "managed training started")

    env = os.environ.copy()
    env["CUDA_DEVICE"] = str(args.cuda_device)
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/gdrnpp-pycache")
    with console_log.open("w", encoding="utf-8") as handle:
        handle.write(
            "MANAGED_RUN_START "
            f"experiment={manifest['experiment_id']} run={manifest['run_id']} "
            f"mode={args.mode} seed={manifest['seed']} "
            f"commit={manifest_source_commit(manifest)} "
            f"image={manifest_environment_image_id(manifest)} "
            f"official_sha256={sha256_file(official)} "
            f"at={utc_now()} command={command_text}\n"
        )
        handle.flush()
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode != 0:
        failure_kind = detect_failure_kind(console_log)
        transition_step(
            run_dir,
            "train",
            "FAILED",
            f"training exited {result.returncode}: {failure_kind}",
        )
        compact_and_write_warning_summary(run_dir)
        write_status(
            run_dir,
            state="FINISHED",
            mode=args.mode,
            exit_code=result.returncode,
            failure_kind=failure_kind,
        )
        return result.returncode

    transition_step(run_dir, "train", "COMPLETE", "managed training completed")
    try:
        checkpoint, checkpoint_id, epoch, selection_kind = checkpoint_for_mode(
            run_dir, args.mode
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        record_checkpoint(
            run_dir,
            checkpoint,
            checkpoint_id,
            epoch,
            checkpoint_iteration(checkpoint),
            selection_kind,
            parent_checkpoint_sha256=sha256_file(official),
        )

        verify_command = [
            "python",
            "-m",
            "research.stage3c_runtime.verify_checkpoint_isolation",
            args.isolation_role,
            "--official",
            str(official),
            "--trained",
            str(checkpoint),
        ]
        register_step(
            run_dir,
            "checkpoint-isolation",
            "verify",
            " ".join(verify_command),
            inputs=[str(official), str(checkpoint)],
            outputs=["checkpoints/checkpoint_index.json"],
        )
        transition_step(
            run_dir,
            "checkpoint-isolation",
            "RUNNING",
            "checkpoint isolation verification started",
        )
        with console_log.open("a", encoding="utf-8") as handle:
            verify = subprocess.run(
                verify_command,
                cwd=Path(__file__).resolve().parents[2],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if verify.returncode:
            transition_step(
                run_dir,
                "checkpoint-isolation",
                "FAILED",
                f"checkpoint isolation exited {verify.returncode}",
            )
            raise RuntimeError("checkpoint isolation verification failed")
        transition_step(
            run_dir,
            "checkpoint-isolation",
            "COMPLETE",
            "checkpoint isolation verification passed",
        )
        compact_and_write_warning_summary(run_dir)

        if args.mode != "formal":
            register_step(run_dir, "summarize", "summarize", "summarize smoke/audit")
            transition_step(
                run_dir, "summarize", "RUNNING", "run summary generation started"
            )
            transition_step(
                run_dir,
                "summarize",
                "COMPLETE",
                "smoke/audit run completed",
                complete_run=True,
            )
        append_console(console_log, f"MANAGED_RUN_FINISH status=PASS at={utc_now()}")
        write_status(
            run_dir,
            state="FINISHED",
            mode=args.mode,
            exit_code=0,
            failure_kind=None,
            checkpoint=str(checkpoint.relative_to(run_dir)),
        )
        return 0
    except Exception:
        append_console(console_log, traceback.format_exc())
        compact_and_write_warning_summary(run_dir)
        # If a verify step did not already fail, mark the run through a new step.
        try:
            complete_supporting_step(
                run_dir,
                "postprocess-context",
                "verify",
                "post-training context captured before failure",
            )
            register_step(
                run_dir,
                "postprocess-failure",
                "verify",
                "post-training verification failed",
            )
            transition_step(
                run_dir,
                "postprocess-failure",
                "RUNNING",
                "post-training verification failed",
            )
            transition_step(
                run_dir,
                "postprocess-failure",
                "FAILED",
                "post-training verification failed",
            )
        except (KeyError, ValueError):
            pass
        write_status(
            run_dir,
            state="FINISHED",
            mode=args.mode,
            exit_code=1,
            failure_kind="POSTPROCESS_ERROR",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
