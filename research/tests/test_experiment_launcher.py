from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from core.utils.default_args_setup import my_default_argument_parser


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "docker/l40/experiment.sh"


def _source_and_run(body: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = f"source {shlex.quote(str(LAUNCHER))}\n{body}"
    return subprocess.run(
        ["bash", "-c", command],
        check=check,
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
        "OUTPUT_DIR=/workspace/gdrnpp/output/run",
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
        "MODEL.WEIGHTS=/workspace/gdrnpp/output/model.pth",
        "OUTPUT_DIR=/workspace/gdrnpp/output/eval",
    ]


def test_gpu_capacity_allows_other_processes_when_memory_is_sufficient():
    result = _source_and_run(
        "gpu_id=0\n"
        "GDRN_MIN_FREE_GPU_MB=8000\n"
        "nvidia-smi() {\n"
        "  case \"$*\" in\n"
        "    *--query-gpu=memory.free*) echo 20000 ;;\n"
        "    *--query-compute-apps=*) echo '4321, python, 5000 MiB' ;;\n"
        "  esac\n"
        "}\n"
        "require_gpu_capacity"
    )
    assert result.stdout.splitlines() == [
        "GPU_CAPACITY WARNING gpu=0 active_compute_processes:",
        "4321, python, 5000 MiB",
        "GPU_CAPACITY PASS gpu=0 free_mb=20000 required_mb=8000",
    ]


def test_gpu_capacity_rejects_insufficient_free_memory():
    result = _source_and_run(
        "gpu_id=0\n"
        "nvidia-smi() {\n"
        "  case \"$*\" in\n"
        "    *--query-gpu=memory.free*) echo 7000 ;;\n"
        "    *--query-compute-apps=*) return 0 ;;\n"
        "  esac\n"
        "}\n"
        "require_gpu_capacity",
        check=False,
    )
    assert result.returncode != 0
    assert result.stderr.strip() == (
        "GPU_CAPACITY FAIL gpu=0 free_mb=7000 required_mb=12000"
    )


def test_idle_container_still_rejects_duplicate_gdrn_process():
    result = _source_and_run(
        "container=test-container\n"
        "docker_bin=fake_docker\n"
        "fake_docker() { return 0; }\n"
        "require_idle_container",
        check=False,
    )
    assert result.returncode != 0
    assert "already active in test-container" in result.stderr


def test_launcher_overrides_satisfy_real_dict_action_parser_contract():
    parser = my_default_argument_parser()

    train_args = parser.parse_args(
        [
            "--config-file",
            "configs/research/train.py",
            "--opts",
            "OUTPUT_DIR=/workspace/gdrnpp/output/train",
        ]
    )
    assert train_args.opts == {"OUTPUT_DIR": "/workspace/gdrnpp/output/train"}

    eval_args = parser.parse_args(
        [
            "--config-file",
            "configs/research/eval.py",
            "--eval-only",
            "--opts",
            "MODEL.WEIGHTS=/workspace/gdrnpp/output/model.pth",
            "OUTPUT_DIR=/workspace/gdrnpp/output/eval",
        ]
    )
    assert eval_args.opts == {
        "MODEL.WEIGHTS": "/workspace/gdrnpp/output/model.pth",
        "OUTPUT_DIR": "/workspace/gdrnpp/output/eval",
    }


def test_runtime_gate_calls_every_lightweight_check():
    result = _source_and_run(
        "container=test-container\n"
        "require_clean_worktree() { echo clean; }\n"
        "require_owned_container() { echo ownership; }\n"
        "verify_required_mounts() { echo mounts; }\n"
        "require_writable_output() { echo output; }\n"
        "require_dataset_cache() { echo dataset-cache; }\n"
        "require_bop_renderer_path() { echo bop-renderer; }\n"
        "require_cuda() { echo cuda; }\n"
        "verify_environment() { echo environment; }\n"
        "verify_native() { echo native; }\n"
        "load_runtime_config() { echo config:$1; }\n"
        "validate_run_config() { echo contract:$1:$2; }\n"
        "runtime_gate formal configs/research/train.py"
    )
    assert result.stdout.splitlines() == [
        "clean",
        "ownership",
        "mounts",
        "output",
        "dataset-cache",
        "bop-renderer",
        "cuda",
        "environment",
        "native",
        "config:configs/research/train.py",
        "contract:formal:configs/research/train.py",
        "RUNTIME_GATE PASS container=test-container mode=formal config=configs/research/train.py",
    ]


def test_gate_precedes_run_directory_creation_and_nested_targets_are_created():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert source.index('runtime_gate "${mode}" "${config}"') < source.index(
        'run_id="$(next_run_id "${mode}")"'
    )
    for target in (
        '${repo_root}/datasets/BOP_DATASETS',
        '${repo_root}/datasets/VOCdevkit',
        '${repo_root}/pretrained_models',
        '${repo_root}/output',
        '${root}/cache/gdrnpp_datasets',
        '${root}/home/.cache',
    ):
        assert f'"{target}"' in source


def test_dataset_cache_env_and_runtime_gate_contract_are_explicit():
    source = LAUNCHER.read_text(encoding="utf-8")
    expected = "/home/gdrn/.cache/gdrnpp_datasets"

    assert f"--env GDRN_DATASET_CACHE_DIR={expected}" in source
    assert 'require_mount "${root}/cache" /home/gdrn/.cache true' in source
    assert 'printenv GDRN_DATASET_CACHE_DIR' in source
    assert 'test -w "${expected}"' in source
    assert source.index("require_dataset_cache()") < source.index(
        "runtime_gate()"
    )


def test_bop_renderer_path_env_and_runtime_gate_contract_are_explicit():
    source = LAUNCHER.read_text(encoding="utf-8")
    expected = "/opt/bop_renderer/build"

    assert f"--env BOP_RENDERER_PATH={expected}" in source
    assert "printenv BOP_RENDERER_PATH" in source
    assert 'test -d "${expected}"' in source
    assert source.index("require_bop_renderer_path()") < source.index(
        "runtime_gate()"
    )


def test_bop_renderer_path_gate_accepts_expected_container_directory():
    result = _source_and_run(
        "container=test-container\n"
        "docker_bin=fake_docker\n"
        "fake_docker() {\n"
        "  case \"$*\" in\n"
        "    *'printenv BOP_RENDERER_PATH') echo /opt/bop_renderer/build ;;\n"
        "    *'test -d /opt/bop_renderer/build') return 0 ;;\n"
        "    *) return 1 ;;\n"
        "  esac\n"
        "}\n"
        "require_bop_renderer_path"
    )
    assert result.returncode == 0


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_clean_worktree_gate_rejects_untracked_source(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Launcher Test")
    _git(tmp_path, "config", "user.email", "launcher@example.invalid")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-qm", "baseline")

    clean = _source_and_run(
        f"repo_root={shlex.quote(str(tmp_path))}\nrequire_clean_worktree"
    )
    assert clean.returncode == 0

    (tmp_path / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    dirty = _source_and_run(
        f"repo_root={shlex.quote(str(tmp_path))}\nrequire_clean_worktree",
        check=False,
    )
    assert dirty.returncode != 0
    assert "Git working tree must be clean" in dirty.stderr


def test_run_metadata_records_full_source_and_image_revisions(tmp_path):
    metadata = tmp_path / "run_metadata.json"
    full_commit = "a" * 40
    image_id = "sha256:" + "b" * 64
    image_revision = "c" * 40
    _source_and_run(
        "experiment_id=EXP-test\n"
        f"write_run_metadata {shlex.quote(str(metadata))} RUN-test formal "
        "configs/test.py "
        f"{full_commit} image:test {image_id} {image_revision}"
    )
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["source_commit"] == full_commit
    assert payload["source_tree_clean"] is True
    assert payload["image_id"] == image_id
    assert payload["image_build_revision"] == image_revision
    assert payload["config"] == "configs/test.py"


def test_image_compatibility_rejects_changed_native_inputs(tmp_path):
    source = LAUNCHER.read_text(encoding="utf-8")
    input_block = source.split("native_input_paths=(", 1)[1].split(")", 1)[0]
    for path in (
        "docker/l40/Dockerfile",
        "docker/l40/requirements.lock",
        "docker/l40/build_native.sh",
        "docker/l40/vendor",
        "core/csrc",
        "lib/egl_renderer",
    ):
        assert path in input_block

    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Launcher Test")
    _git(tmp_path, "config", "user.email", "launcher@example.invalid")
    native_source = tmp_path / "core/csrc/example.cpp"
    native_source.parent.mkdir(parents=True)
    native_source.write_text("baseline\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("baseline\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "baseline")
    image_commit = _git(tmp_path, "rev-parse", "HEAD")

    (tmp_path / "notes.txt").write_text("non-native change\n", encoding="utf-8")
    _git(tmp_path, "commit", "-qam", "non-native change")
    compatible = _source_and_run(
        f"repo_root={shlex.quote(str(tmp_path))}\n"
        f"image_revision() {{ echo {image_commit}; }}\n"
        "require_image_source_compatibility test-image"
    )
    assert compatible.stdout.startswith("IMAGE_COMPATIBILITY PASS")

    native_source.write_text("native change\n", encoding="utf-8")
    _git(tmp_path, "commit", "-qam", "native change")
    incompatible = _source_and_run(
        f"repo_root={shlex.quote(str(tmp_path))}\n"
        f"image_revision() {{ echo {image_commit}; }}\n"
        "require_image_source_compatibility test-image",
        check=False,
    )
    assert incompatible.returncode != 0
    assert "rebuild image" in incompatible.stderr


def test_native_hydration_contract_covers_required_artifacts(tmp_path):
    artifacts = (
        "core/csrc/fps/_ext.test.so",
        "core/csrc/flow/flow_cuda.test.so",
        "core/csrc/ransac_voting/ransac_voting.test.so",
        "core/csrc/torch_nndistance/torch_nndistance_aten.test.so",
        "core/csrc/uncertainty_pnp/_ext.test.so",
        "core/csrc/uncertainty_pnp/lib/libceres.so",
        "lib/egl_renderer/CppEGLRenderer.test.so",
    )
    for relative in artifacts:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    _source_and_run(
        f"repo_root={shlex.quote(str(tmp_path))}\nrequire_native_artifacts"
    )
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '"${docker_bin}" run --rm --entrypoint bash' in source
    assert 'git -C "${repo_root}" diff --check' in source
    assert "status --porcelain --untracked-files=no" in source
    assert source.index('require_image_source_compatibility "${image_ref}"') < source.index(
        'hydrate_native_artifacts "${image_ref}"'
    ) < source.index('"${docker_bin}" run -d')
