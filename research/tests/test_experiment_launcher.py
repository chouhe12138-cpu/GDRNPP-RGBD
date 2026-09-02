from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "docker/l40/experiment.sh"


def _source_and_run(body: str) -> subprocess.CompletedProcess[str]:
    command = f"source {shlex.quote(str(LAUNCHER))}\n{body}"
    return subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )


def test_train_command_places_output_override_after_opts():
    result = _source_and_run(
        "build_train_command configs/research/smoke.py /workspace/gdrnpp/output/run"
    )
    assert shlex.split(result.stdout) == [
        "core/gdrn_modeling/train_gdrn.sh",
        "configs/research/smoke.py",
        "0",
        "--opts",
        "OUTPUT_DIR",
        "/workspace/gdrnpp/output/run",
    ]


def test_eval_command_places_all_overrides_after_opts():
    result = _source_and_run(
        "build_eval_command configs/research/eval.py "
        "/workspace/gdrnpp/output/model.pth /workspace/gdrnpp/output/eval"
    )
    assert shlex.split(result.stdout) == [
        "python",
        "core/gdrn_modeling/main_gdrn.py",
        "--config-file",
        "configs/research/eval.py",
        "--num-gpus",
        "1",
        "--eval-only",
        "--opts",
        "MODEL.WEIGHTS",
        "/workspace/gdrnpp/output/model.pth",
        "OUTPUT_DIR",
        "/workspace/gdrnpp/output/eval",
    ]


def test_runtime_gate_calls_every_lightweight_check():
    result = _source_and_run(
        "container=test-container\n"
        "require_owned_container() { echo ownership; }\n"
        "verify_required_mounts() { echo mounts; }\n"
        "require_writable_output() { echo output; }\n"
        "require_cuda() { echo cuda; }\n"
        "verify_environment() { echo environment; }\n"
        "verify_native() { echo native; }\n"
        "load_runtime_config() { echo config:$1; }\n"
        "runtime_gate configs/research/train.py"
    )
    assert result.stdout.splitlines() == [
        "ownership",
        "mounts",
        "output",
        "cuda",
        "environment",
        "native",
        "config:configs/research/train.py",
        "RUNTIME_GATE PASS container=test-container config=configs/research/train.py",
    ]


def test_gate_precedes_run_directory_creation_and_nested_targets_are_created():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert source.index('runtime_gate "${config}"') < source.index(
        'run_id="$(next_run_id "${mode}")"'
    )
    for target in (
        '${repo_root}/datasets/BOP_DATASETS',
        '${repo_root}/datasets/VOCdevkit',
        '${repo_root}/pretrained_models',
        '${repo_root}/output',
        '${root}/home/.cache',
    ):
        assert f'"{target}"' in source
