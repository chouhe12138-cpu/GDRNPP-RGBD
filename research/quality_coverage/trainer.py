#!/usr/bin/env python3
"""Host-side Python API for the Docker-managed Stage 3C-1 experiment."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = PROJECT_ROOT / "docker/l40/stage3c1.sh"
FORMAL_CONFIG = (
    "/workspace/gdrnpp/configs/gdrn/lmo_pbr/"
    "convnext_stage3c1_quality_coverage_lmo.py"
)


@dataclass(frozen=True)
class TrainingOptions:
    """User-facing settings for one Stage 3C-1 run."""

    epochs: int = 40
    batch_size: int = 48
    effective_batch_size: int = 48
    learning_rate: float = 8e-4
    weight_decay: float = 0.01
    workers: int = 8
    evaluate_every: int = 5
    save_every: int = 5
    keep_checkpoints: int = 3
    seed: int = 20260731
    name: str = "quality_coverage_full"
    protocol: Literal["official", "exploratory"] = "official"
    resume: bool = False
    run_baseline_if_missing: bool = True
    draw_plots: bool = True

    def validate(self) -> None:
        if self.protocol not in {"official", "exploratory"}:
            raise ValueError("protocol must be 'official' or 'exploratory'")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("name may contain only letters, numbers, '.', '_' and '-'")
        positive_ints = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "effective_batch_size": self.effective_batch_size,
            "evaluate_every": self.evaluate_every,
            "save_every": self.save_every,
            "keep_checkpoints": self.keep_checkpoints,
        }
        invalid = {key: value for key, value in positive_ints.items() if int(value) <= 0}
        if invalid:
            raise ValueError(f"these settings must be positive: {invalid}")
        if self.workers < 0:
            raise ValueError("workers must be nonnegative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if (
            self.effective_batch_size < self.batch_size
            or self.effective_batch_size % self.batch_size
        ):
            raise ValueError(
                "effective_batch_size must be a positive multiple of batch_size"
            )
        if self.evaluate_every > self.epochs or self.save_every > self.epochs:
            raise ValueError("evaluation/save periods cannot exceed total epochs")
        if self.keep_checkpoints < 3:
            raise ValueError(
                "keep_checkpoints must leave room for best one plus two recent weights"
            )
        if self.protocol == "official":
            expected = {
                "epochs": 40,
                "batch_size": 48,
                "effective_batch_size": 48,
                "learning_rate": 8e-4,
                "weight_decay": 0.01,
                "evaluate_every": 5,
                "save_every": 5,
                "keep_checkpoints": 3,
                "seed": 20260731,
                "name": "quality_coverage_full",
            }
            actual = {key: getattr(self, key) for key in expected}
            changed = {
                key: {"expected": expected[key], "actual": actual[key]}
                for key in expected
                if actual[key] != expected[key]
            }
            if changed:
                raise ValueError(
                    "official protocol is locked; use protocol='exploratory' "
                    f"for changed settings: {changed}"
                )


@dataclass(frozen=True)
class TrainingJob:
    name: str
    output_dir: Path
    log_path: Path
    status_path: Path
    launched: bool


class Stage3C1:
    """A small ``model.train(...)``-style interface for the L40 container."""

    def __init__(
        self,
        *,
        docker: str = "/usr/bin/docker",
        container: str = "lab1_chx_stage3c1",
        workspace: Union[Path, str] = "/data/labs/lab1/docker_data/chx",
    ) -> None:
        self.docker = docker
        self.container = container
        self.workspace = Path(workspace)
        self.outputs = self.workspace / "outputs/EXP-20260731-006"
        self.logs = self.workspace / "logs"

    def train(
        self,
        *,
        epochs: int = 40,
        batch: int = 48,
        effective_batch: int = 48,
        lr: float = 8e-4,
        weight_decay: float = 0.01,
        workers: int = 8,
        evaluate_every: int = 5,
        save_every: int = 5,
        keep_checkpoints: int = 3,
        seed: int = 20260731,
        name: str = "quality_coverage_full",
        protocol: Literal["official", "exploratory"] = "official",
        resume: bool = False,
        baseline: bool = True,
        plots: bool = True,
        launch: bool = True,
    ) -> TrainingJob:
        """Validate, record, and optionally launch a background L40 run."""

        options = TrainingOptions(
            epochs=epochs,
            batch_size=batch,
            effective_batch_size=effective_batch,
            learning_rate=lr,
            weight_decay=weight_decay,
            workers=workers,
            evaluate_every=evaluate_every,
            save_every=save_every,
            keep_checkpoints=keep_checkpoints,
            seed=seed,
            name=name,
            protocol=protocol,
            resume=resume,
            run_baseline_if_missing=baseline,
            draw_plots=plots,
        )
        options.validate()
        output_dir = self.outputs / options.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.logs / f"stage3c1_train_{options.name}_{timestamp}.log"
        status_path = log_path.with_suffix(".status")
        job = TrainingJob(
            name=options.name,
            output_dir=output_dir,
            log_path=log_path,
            status_path=status_path,
            launched=launch,
        )
        self._print_plan(options, job)
        if not launch:
            print("PREVIEW_ONLY: set launch=True when you want to start this run.")
            return job

        self._require_smoke_gate()
        self._prepare_output(options, output_dir)
        config_path = output_dir / "stage3c1_runtime_config.py"
        runner_path = output_dir / (
            f"resume_inside_container_{timestamp}.sh"
            if options.resume
            else "run_inside_container.sh"
        )
        manifest_path = output_dir / "run_manifest.json"
        plot_script_path = output_dir / "plot_curves_runtime.py"
        if not options.resume:
            config_path.write_text(self._render_config(options), encoding="utf-8")
            plot_script_path.write_text(
                (PROJECT_ROOT / "research/quality_coverage/plot_curves.py").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
        runner_path.write_text(
            self._render_runner(
                options,
                config_path,
                output_dir,
                plot_script_path,
            ),
            encoding="utf-8",
        )
        if options.resume:
            with (output_dir / "resume_events.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(
                        {
                            "created_at": datetime.now().astimezone().isoformat(),
                            "runner": runner_path.name,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        else:
            manifest_path.write_text(
                json.dumps(
                    {
                        "created_at": datetime.now().astimezone().isoformat(),
                        "status": (
                            "FORMAL_PROTOCOL"
                            if options.protocol == "official"
                            else "EXPLORATORY_NOT_FOR_PAPER_COMPARISON"
                        ),
                        "options": asdict(options),
                        "container": self.container,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        container_runner = self._container_path(runner_path)
        result = subprocess.run(
            [
                self.docker,
                "exec",
                "-d",
                "-e",
                f"TRAIN_LOG={self._container_path(log_path)}",
                "-e",
                f"TRAIN_STATUS={self._container_path(status_path)}",
                self.container,
                "bash",
                container_runner,
            ],
            check=False,
        )
        if result.returncode:
            status_path.write_text(
                f"state=LAUNCH_FAILED\nexit={result.returncode}\n",
                encoding="utf-8",
            )
            raise RuntimeError(f"docker exec failed with exit code {result.returncode}")
        print("TRAINING_LAUNCH=STARTED")
        print(f"Use model.status(name={options.name!r}) to inspect progress.")
        return job

    def status(self, name: str = "quality_coverage_full") -> dict[str, object]:
        """Print and return the latest status for an experiment name."""

        self._validate_name(name)
        candidates = sorted(
            self.logs.glob(f"stage3c1_train_{name}_*.status"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            result: dict[str, object] = {"state": "NOT_STARTED", "name": name}
            print(json.dumps(result, indent=2))
            return result
        status_path = candidates[0]
        values: dict[str, object] = {"name": name, "status_path": str(status_path)}
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        log_path = status_path.with_suffix(".log")
        if log_path.is_file():
            progress = [
                line for line in log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if " iter: " in line
            ]
            if progress:
                values["latest_progress"] = progress[-1]
        print(json.dumps(values, indent=2, ensure_ascii=False))
        return values

    def watch(self, name: str = "quality_coverage_full") -> int:
        """Follow the latest log. Ctrl-C stops viewing, not Docker training."""

        self._validate_name(name)
        logs = sorted(
            self.logs.glob(f"stage3c1_train_{name}_*.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not logs:
            raise FileNotFoundError(f"no training log found for {name}")
        print("Ctrl-C stops viewing only; the detached training keeps running.")
        return subprocess.run(["tail", "-f", str(logs[0])], check=False).returncode

    def _require_smoke_gate(self) -> None:
        result = subprocess.run([str(CONTROLLER), "validate"], check=False)
        if result.returncode:
            raise RuntimeError("the one-epoch smoke gate has not passed")

    def _prepare_output(self, options: TrainingOptions, output_dir: Path) -> None:
        if options.resume:
            if not (output_dir / "last_checkpoint").is_file():
                raise FileNotFoundError(
                    f"resume=True requires {output_dir / 'last_checkpoint'}"
                )
            manifest = output_dir / "run_manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(f"resume manifest is missing: {manifest}")
            config = output_dir / "stage3c1_runtime_config.py"
            plot_script = output_dir / "plot_curves_runtime.py"
            if not config.is_file() or not plot_script.is_file():
                raise FileNotFoundError(
                    "resume config or plotting script is missing from the run"
                )
            saved_options = json.loads(
                manifest.read_text(encoding="utf-8")
            )["options"]
            current_options = asdict(options)
            saved_options.pop("resume", None)
            current_options.pop("resume", None)
            if saved_options != current_options:
                raise ValueError(
                    "resume settings differ from the original run; "
                    "start a new exploratory experiment instead"
                )
        else:
            if output_dir.exists():
                raise FileExistsError(
                    f"output already exists and will not be overwritten: {output_dir}"
                )
            output_dir.mkdir(parents=True)
            # The directory is private to lab1, but the host user and the
            # fixed non-root container UID both need to create run artifacts.
            output_dir.chmod(0o777)

    def _render_config(self, options: TrainingOptions) -> str:
        container_output = (
            f"/workspace/gdrnpp/output/EXP-20260731-006/{options.name}"
        )
        return f'''_base_ = ["{FORMAL_CONFIG}"]

OUTPUT_DIR = "{container_output}"
SEED = {options.seed}

DATALOADER = dict(NUM_WORKERS={options.workers})

SOLVER = dict(
    IMS_PER_BATCH={options.batch_size},
    REFERENCE_BS={options.effective_batch_size},
    TOTAL_EPOCHS={options.epochs},
    OPTIMIZER_CFG=dict(
        _delete_=True,
        type="Ranger",
        lr={options.learning_rate!r},
        weight_decay={options.weight_decay!r},
    ),
    CHECKPOINT_PERIOD={options.save_every},
    MAX_TO_KEEP={options.keep_checkpoints},
)

TEST = dict(EVAL_PERIOD={options.evaluate_every})
'''

    def _render_runner(
        self,
        options: TrainingOptions,
        config_path: Path,
        output_dir: Path,
        plot_script_path: Path,
    ) -> str:
        container_config = self._container_path(config_path)
        container_output = self._container_path(output_dir)
        container_plot_script = self._container_path(plot_script_path)
        preflight_config = (
            container_config
            if options.protocol == "official"
            else FORMAL_CONFIG
        )
        resume_flag = " --resume" if options.resume else ""
        baseline_block = ""
        if options.run_baseline_if_missing:
            baseline_block = '''
if ! find "${BASELINE_OUTPUT}" -name scores_bop19.json -print -quit 2>/dev/null | grep -q .; then
    if [[ -d "${BASELINE_OUTPUT}" ]] && find "${BASELINE_OUTPUT}" -mindepth 1 -print -quit | grep -q .; then
        echo "Incomplete baseline output exists; refusing to overwrite it."
        exit 1
    fi
    research/quality_coverage/run_baseline.sh
fi
'''
        plot_block = ""
        if options.draw_plots:
            plot_block = f'''
python {shlex.quote(container_plot_script)} \
    {shlex.quote(container_output)} \
    --baseline-output "${{BASELINE_OUTPUT}}"
python -m research.quality_coverage.summarize \
    {shlex.quote(container_output)} \
    "${{BASELINE_OUTPUT}}"
'''
        return f'''#!/usr/bin/env bash
set -Eeuo pipefail

finish() {{
    rc=$?
    printf "state=FINISHED\\nexit=%s\\nfinished=%s\\n" \
        "${{rc}}" "$(date --iso-8601=seconds)" > "${{TRAIN_STATUS}}"
    exit "${{rc}}"
}}
trap finish EXIT

printf "state=RUNNING\\nstarted=%s\\n" \
    "$(date --iso-8601=seconds)" > "${{TRAIN_STATUS}}"
exec > >(tee -a "${{TRAIN_LOG}}") 2>&1

cd /workspace/gdrnpp
export CUDA_DEVICE=0
export MPLCONFIGDIR=/tmp/gdrnpp-matplotlib
export PYTHONPYCACHEPREFIX=/tmp/gdrnpp-pycache
BASELINE_OUTPUT=/workspace/gdrnpp/output/EXP-20260731-006/official_gt

python -m research.quality_coverage.preflight \
    --config {shlex.quote(preflight_config)}
{baseline_block}
./core/gdrn_modeling/train_gdrn.sh \
    {shlex.quote(container_config)} 0{resume_flag}
{plot_block}
'''

    def _container_path(self, host_path: Path) -> str:
        resolved = host_path.resolve()
        mappings = (
            (self.workspace / "outputs", Path("/workspace/gdrnpp/output")),
            (self.workspace / "logs", Path("/workspace/logs")),
        )
        for host_root, container_root in mappings:
            try:
                relative = resolved.relative_to(host_root.resolve())
            except ValueError:
                continue
            return str(container_root / relative)
        raise ValueError(f"path is not mounted in the container: {host_path}")

    def _print_plan(self, options: TrainingOptions, job: TrainingJob) -> None:
        print(
            json.dumps(
                {
                    "mode": (
                        "FORMAL_PROTOCOL"
                        if options.protocol == "official"
                        else "EXPLORATORY_NOT_FOR_PAPER_COMPARISON"
                    ),
                    "options": asdict(options),
                    "output_dir": str(job.output_dir),
                    "launch": job.launched,
                },
                indent=2,
                sort_keys=True,
            )
        )

    @staticmethod
    def _validate_name(name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("invalid experiment name")
