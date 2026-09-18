from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from core.utils.default_args_setup import my_default_argument_parser


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "docker/l40/experiment.sh"

LM13_OBJECTS = (
    "ape", "benchvise", "camera", "can", "cat", "driller", "duck",
    "eggbox", "glue", "holepuncher", "iron", "lamp", "phone",
)
LM13_HIERARCHY = "/home/gdrn/.cache/gdrnpp_datasets/exp022/lm13/independent_v2.npz"
LMO_FULL_HIERARCHY = "/home/gdrn/.cache/gdrnpp_datasets/exp022/reused_v1.npz"
LM13_IMGN_SPLIT = "lm_imgn_13_train_1k_per_obj_online"
CONVNEXT_CHECKPOINT = (
    "/workspace/gdrnpp/pretrained_models/convnext/convnext_base_1k_224_ema.pth"
)
LM13_CONTAINER_PATHS = (
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/test",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/image_set",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/models",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/train_pbr",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/train_pbr/000000/scene_gt.json",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/image_set/ape_train.txt",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/image_set/ape_test.txt",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/test/000001/scene_gt.json",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/test/000001/scene_gt_info.json",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/test/000001/scene_camera.json",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/models/obj_000001.ply",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/models/models_info.json",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lm/test_targets_bop19.json",
    "/workspace/gdrnpp/datasets/BOP_DATASETS/lmo/test",
    "/workspace/gdrnpp/datasets/lm_imgn/image_set",
    "/workspace/gdrnpp/datasets/lm_imgn/imgn",
    "/workspace/gdrnpp/datasets/lm_imgn/imgn/ape",
    "/workspace/gdrnpp/datasets/lm_imgn/imgn/ape/000000_0-color.png",
    "/workspace/gdrnpp/datasets/lm_imgn/imgn/ape/000000_0-depth.png",
    "/workspace/gdrnpp/datasets/lm_imgn/imgn/ape/000000_0-pose.txt",
    "/workspace/gdrnpp/datasets/VOCdevkit/VOC2012/JPEGImages",
    "/workspace/gdrnpp/datasets/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg",
    CONVNEXT_CHECKPOINT,
    "/workspace/gdrnpp/pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    LM13_HIERARCHY,
    LMO_FULL_HIERARCHY,
) + tuple(
    f"/workspace/gdrnpp/datasets/lm_imgn/image_set/train_{obj}.txt" for obj in LM13_OBJECTS
)

# A stand-in for the container: `present` lists the paths that "exist", and the
# three python entry points the launcher calls are answered from variables.
FAKE_DOCKER = r'''
container=test-container
docker_bin=fake_docker
present=(
{paths}
)
fake_has() {{
  local candidate
  for candidate in "${{present[@]}}"; do
    [[ "${{candidate}}" == "$1" ]] && return 0
  done
  return 1
}}
fake_has_glob() {{
  local prefix="${{1%%\**}}" suffix="${{1##*\*}}" candidate
  for candidate in "${{present[@]}}"; do
    [[ "${{candidate}}" == "${{prefix}}"*"${{suffix}}" ]] && return 0
  done
  return 1
}}
fake_docker() {{
  local verb="$1"; shift
  [[ "${{verb}}" == "exec" ]] || return 1
  while [[ $# -gt 0 && "$1" == -* ]]; do
    case "$1" in
      -w|-e) shift 2 ;;
      *) shift ;;
    esac
  done
  shift
  case "$1" in
    test) fake_has "$3" ;;
    bash) fake_has_glob "$5" ;;
    printenv)
      [[ -n "${{fake_convnext:-}}" ]] || return 1
      printf '%s\n' "${{fake_convnext}}"
      ;;
    python)
      if [[ "$2" == "-c" ]]; then
        case "$5" in
          TRAIN_PROTOCOL.NAME) printf '%s\n' "${{fake_train_protocol:-}}" ;;
          DATASETS.TRAIN) printf '%s\n' "${{fake_train_splits:-}}" ;;
          MODEL.POSE_NET.PCC_HEAD.HIERARCHY_PATH) printf '%s\n' "${{fake_hierarchy:-}}" ;;
          *) return 1 ;;
        esac
        return 0
      fi
      return "${{fake_server_preflight:-0}}"
      ;;
    *) return 1 ;;
  esac
}}
'''


def _resource_gate(
    body: str,
    *,
    missing: tuple[str, ...] = (),
    train_protocol: str = "lm13_gdrn",
    train_splits: tuple[str, ...] = ("lm_13_train_online", LM13_IMGN_SPLIT),
    convnext: str = CONVNEXT_CHECKPOINT,
    hierarchy: str = LM13_HIERARCHY,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    present = [path for path in LM13_CONTAINER_PATHS if path not in missing]
    preamble = FAKE_DOCKER.format(
        paths="\n".join(f"    {shlex.quote(path)}" for path in present)
    )
    preamble += (
        f"fake_train_protocol={shlex.quote(train_protocol)}\n"
        f"fake_train_splits={shlex.quote(chr(10).join(train_splits))}\n"
        f"fake_convnext={shlex.quote(convnext)}\n"
        f"fake_hierarchy={shlex.quote(hierarchy)}\n"
    )
    return _source_and_run(preamble + body, check=check)



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


def test_idle_container_accepts_only_sleep_infinity():
    result = _source_and_run(
        "container=test-container\n"
        "docker_bin=fake_docker\n"
        "fake_docker() {\n"
        "  printf 'PID COMMAND\\n123 sleep infinity\\n'\n"
        "}\n"
        "require_idle_container"
    )
    assert result.returncode == 0


def test_idle_container_rejects_setproctitle_process():
    result = _source_and_run(
        "container=test-container\n"
        "docker_bin=fake_docker\n"
        "fake_docker() {\n"
        "  printf 'PID COMMAND\\n123 sleep infinity\\n456 control.20260909_171616\\n'\n"
        "}\n"
        "require_idle_container",
        check=False,
    )
    assert result.returncode != 0
    assert "control.20260909_171616" in result.stderr
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
        "resolve_resource_profile() { echo lm13; }\n"
        "require_profile_resources() { echo resources:$1:$2; }\n"
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
        "resources:lm13:configs/research/train.py",
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
        '${repo_root}/datasets/lm_imgn',
        '${repo_root}/pretrained_models',
        '${repo_root}/output',
        '${root}/datasets/lm_imgn',
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


def test_launcher_mounts_lm_imgn_read_only_and_injects_the_convnext_path():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert (
        '--mount "type=bind,src=${root}/datasets/lm_imgn,'
        'dst=/workspace/gdrnpp/datasets/lm_imgn,readonly"' in source
    )
    assert (
        'require_mount "${root}/datasets/lm_imgn" '
        '/workspace/gdrnpp/datasets/lm_imgn false' in source
    )
    assert (
        f"--env GDRN_CONVNEXT_BASE_WEIGHTS={CONVNEXT_CHECKPOINT}" in source
    )
    # the host path stays a launcher concern; no config may pin it
    for config in sorted((ROOT / "configs/gdrn/research").rglob("*.py")):
        assert "/data/labs/" not in config.read_text(encoding="utf-8"), config


def test_lm_imgn_mount_is_required_read_only(tmp_path):
    source = tmp_path / "datasets/lm_imgn"
    source.mkdir(parents=True)
    row = f"{source}\t/workspace/gdrnpp/datasets/lm_imgn\t{{}}\n"

    def body(rw: str) -> str:
        return (
            "container=test-container\n"
            "docker_bin=fake_docker\n"
            f"root={shlex.quote(str(tmp_path))}\n"
            "fake_docker() {\n"
            f"  printf '%s\\n' {shlex.quote(row.format(rw))}\n"
            "}\n"
            'require_mount "${root}/datasets/lm_imgn" /workspace/gdrnpp/datasets/lm_imgn false'
        )

    assert _source_and_run(body("false")).returncode == 0
    rejected = _source_and_run(body("true"), check=False)
    assert rejected.returncode != 0
    assert "expected false" in rejected.stderr


@pytest.mark.parametrize("name,expected", [
    ("lm13_gdrn", "lm13"),
    ("lm13_real_only", "lm13"),
    ("lm13_pbr", "lm13_pbr"),
    ("lmo_full_imagenet", "lmo_full_imagenet"),
    ("", "legacy_lmo"),
])
def test_resource_profile_mapping(name, expected):
    result = _source_and_run(
        "container=test-container\n"
        "docker_bin=fake_docker\n"
        f"train_protocol={shlex.quote(name)}\n"
        "container_config_value() { printf '%s\\n' \"${train_protocol}\"; }\n"
        f"resolve_resource_profile configs/research/{name or 'legacy_lmo'}.py"
    )
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("name", ["lm13_typo", "some_future_protocol", "lm13_gdrn_v2"])
def test_unknown_train_protocol_is_rejected(name):
    """An unrecognised name must not silently inherit the legacy LM-O contract."""
    result = _source_and_run(
        "container=test-container\n"
        "docker_bin=fake_docker\n"
        f"train_protocol={shlex.quote(name)}\n"
        "container_config_value() { printf '%s\\n' \"${train_protocol}\"; }\n"
        "resolve_resource_profile configs/research/train.py",
        check=False,
    )
    assert result.returncode != 0
    assert "unknown TRAIN_PROTOCOL.NAME" in result.stderr
    assert name in result.stderr


def test_unknown_train_protocol_stops_the_runtime_gate_before_the_contract():
    """The gate must reject the config outright, not fall through to legacy resources."""
    result = _source_and_run(
        "container=test-container\n"
        "docker_bin=fake_docker\n"
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
        "require_profile_resources() { echo resources:$1; }\n"
        "validate_run_config() { echo contract:$1; }\n"
        "container_config_value() { printf '%s\\n' 'lm13_typo'; }\n"
        "runtime_gate formal configs/research/train.py",
        check=False,
    )
    assert result.returncode != 0
    assert "unknown TRAIN_PROTOCOL.NAME" in result.stderr
    assert "resources:" not in result.stdout
    assert "contract:" not in result.stdout


def test_every_declared_train_protocol_resolves_to_a_profile():
    """The fail-closed branch must never reject a name the repository declares."""
    declared = set()
    for config in sorted((ROOT / "configs").rglob("*.py")):
        declared.update(
            re.findall(
                r'TRAIN_PROTOCOL\s*=\s*dict\(NAME="([^"]+)"',
                config.read_text(encoding="utf-8"),
            )
        )
    assert declared == {"lm13_gdrn", "lm13_real_only", "lm13_pbr", "lmo_full_imagenet"}
    for name in sorted(declared):
        result = _source_and_run(
            "container=test-container\n"
            "docker_bin=fake_docker\n"
            f"train_protocol={shlex.quote(name)}\n"
            "container_config_value() { printf '%s\\n' \"${train_protocol}\"; }\n"
            "resolve_resource_profile configs/research/train.py",
            check=False,
        )
        assert result.returncode == 0, (name, result.stderr)
        assert result.stdout.strip() in {"lm13", "lm13_pbr", "lmo_full_imagenet"}


def test_resource_profile_ignores_noise_before_the_config_value():
    """An import warning on stdout must not demote an LM13 config to legacy."""
    result = _source_and_run(
        "container=test-container\n"
        "docker_bin=fake_docker\n"
        "container_config_value() { printf 'some import warning\\nlm13_gdrn\\n'; }\n"
        "resolve_resource_profile configs/research/train_lm13_gdrn.py"
    )
    assert result.stdout.strip() == "lm13"


def test_resource_profile_comes_from_the_loaded_config():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'container_config_value "${config}" TRAIN_PROTOCOL.NAME' in source
    assert 'container_config_value "${config}" MODEL.POSE_NET.PCC_HEAD.HIERARCHY_PATH' in source


def test_resolve_resource_profile_reads_the_train_protocol_in_the_container():
    result = _resource_gate(
        "resolve_resource_profile configs/research/train_lm13_gdrn.py",
        train_protocol="lm13_real_only",
    )
    assert result.stdout.strip() == "lm13"


def test_require_profile_resources_dispatches_on_the_profile():
    result = _source_and_run(
        "require_lm13_resources() { echo lm13:$1; }\n"
        "require_lm13_pbr_resources() { echo lm13-pbr:$1; }\n"
        "require_lmo_full_imagenet_resources() { echo lmo-full:$1; }\n"
        "require_legacy_lmo_resources() { echo legacy; }\n"
        "require_profile_resources lm13 configs/a.py\n"
        "require_profile_resources lm13_pbr configs/a.py\n"
        "require_profile_resources lmo_full_imagenet configs/a.py\n"
        "require_profile_resources legacy_lmo configs/a.py"
    )
    assert result.stdout.splitlines() == [
        "lm13:configs/a.py",
        "RESOURCE_PROFILE lm13",
        "lm13-pbr:configs/a.py",
        "RESOURCE_PROFILE lm13_pbr",
        "lmo-full:configs/a.py",
        "RESOURCE_PROFILE lmo_full_imagenet",
        "legacy",
        "RESOURCE_PROFILE legacy_lmo",
    ]


def test_lm13_resource_gate_accepts_a_complete_container():
    result = _resource_gate("require_lm13_resources configs/research/train_lm13_gdrn.py")
    assert result.returncode == 0
    assert result.stdout.splitlines() == [f"LM13_HIERARCHY {LM13_HIERARCHY}"]


@pytest.mark.parametrize("missing,expected", [
    (("/workspace/gdrnpp/datasets/lm_imgn/imgn",), "lm_imgn/imgn"),
    (("/workspace/gdrnpp/datasets/lm_imgn/image_set/train_phone.txt",), "train_phone.txt"),
    (("/workspace/gdrnpp/datasets/lm_imgn/imgn/ape/000000_0-pose.txt",), "*-pose.txt"),
    (("/workspace/gdrnpp/datasets/VOCdevkit/VOC2012/JPEGImages",), "JPEGImages"),
    (("/workspace/gdrnpp/datasets/BOP_DATASETS/lm/test",), "BOP_DATASETS/lm/test"),
    (("/workspace/gdrnpp/datasets/BOP_DATASETS/lm/models/models_info.json",), "models_info.json"),
    ((LM13_HIERARCHY,), "independent_v2.npz"),
])
def test_lm13_resource_gate_rejects_missing_resources(missing, expected):
    result = _resource_gate(
        "require_lm13_resources configs/research/train_lm13_gdrn.py",
        missing=missing,
        check=False,
    )
    assert result.returncode != 0
    assert expected in result.stderr


def test_lm13_resource_gate_rejects_a_missing_convnext_checkpoint():
    result = _resource_gate(
        "require_lm13_resources configs/research/train_lm13_gdrn.py",
        missing=(CONVNEXT_CHECKPOINT,),
        check=False,
    )
    assert result.returncode != 0
    assert CONVNEXT_CHECKPOINT in result.stderr


def test_lm13_resource_gate_rejects_an_unset_convnext_variable():
    result = _resource_gate(
        "require_lm13_resources configs/research/train_lm13_gdrn.py",
        convnext="",
        check=False,
    )
    assert result.returncode != 0
    assert "GDRN_CONVNEXT_BASE_WEIGHTS is not set" in result.stderr


def test_lm13_gate_surfaces_a_server_preflight_failure():
    result = _resource_gate(
        "fake_server_preflight=1\n"
        "require_lm13_resources configs/research/train_lm13_gdrn.py",
        check=False,
    )
    assert result.returncode != 0
    assert "LM13 server preflight failed" in result.stderr


def test_lm13_real_only_gate_does_not_require_the_deepim_renders():
    """The real-only ablation trains without lm_imgn, so the gate must not ask for it."""
    imgn = tuple(path for path in LM13_CONTAINER_PATHS if "/lm_imgn/" in path)
    assert imgn
    result = _resource_gate(
        "require_lm13_resources configs/research/train_lm13_real_only.py",
        train_protocol="lm13_real_only",
        train_splits=("lm_13_train_online",),
        missing=imgn,
    )
    assert result.returncode == 0, result.stderr
    # ... while the render-training arm is still rejected without them
    rejected = _resource_gate(
        "require_lm13_resources configs/research/train_lm13_gdrn.py",
        missing=imgn,
        check=False,
    )
    assert rejected.returncode != 0
    assert "lm_imgn" in rejected.stderr


def test_lm13_imgn_requirement_comes_from_the_configured_train_splits():
    result = _source_and_run(
        "container=test-container\n"
        "docker_bin=fake_docker\n"
        "require_lm13_real_data() { echo real; }\n"
        "require_lm_imgn_data() { echo imgn; }\n"
        "require_voc_data() { echo voc; }\n"
        "require_convnext_weights() { echo convnext; }\n"
        "require_lm13_hierarchy() { echo hierarchy; }\n"
        "require_lm13_server_preflight() { echo preflight; }\n"
        "container_config_list() { printf '%s\\n' \"${fake_train_splits}\"; }\n"
        "fake_train_splits=$'lm_13_train_online\\nlm_imgn_13_train_1k_per_obj_online'\n"
        "require_lm13_resources configs/a.py\n"
        "fake_train_splits='lm_13_train_online'\n"
        "require_lm13_resources configs/a.py"
    )
    assert result.stdout.splitlines() == [
        "real", "imgn", "voc", "convnext", "hierarchy", "preflight",
        "real", "voc", "convnext", "hierarchy", "preflight",
    ]


def test_lm13_pbr_gate_does_not_require_the_deepim_renders():
    imgn = tuple(path for path in LM13_CONTAINER_PATHS if "/lm_imgn/" in path)
    assert imgn
    result = _resource_gate(
        "require_lm13_pbr_resources configs/research/train_lm13_pbr.py",
        missing=imgn,
    )
    assert result.returncode == 0
    # the PBR arm still needs its own renders
    rejected = _resource_gate(
        "require_lm13_pbr_resources configs/research/train_lm13_pbr.py",
        missing=("/workspace/gdrnpp/datasets/BOP_DATASETS/lm/train_pbr",),
        check=False,
    )
    assert rejected.returncode != 0
    assert "train_pbr" in rejected.stderr


def test_legacy_lmo_gate_keeps_the_original_requirements():
    result = _resource_gate("require_legacy_lmo_resources")
    assert result.returncode == 0

    # the LM-O contract predates lm_imgn, so its content is not its business
    imgn = tuple(path for path in LM13_CONTAINER_PATHS if "/lm_imgn/" in path)
    result = _resource_gate("require_legacy_lmo_resources", missing=imgn)
    assert result.returncode == 0

    result = _resource_gate(
        "require_legacy_lmo_resources",
        missing=("/workspace/gdrnpp/pretrained_models/lmo_pbr/model_final_wo_optim.pth",),
        check=False,
    )
    assert result.returncode != 0
    assert "model_final_wo_optim.pth" in result.stderr


def test_lmo_full_gate_uses_imagenet_and_hierarchy_without_official_weights():
    official = "/workspace/gdrnpp/pretrained_models/lmo_pbr/model_final_wo_optim.pth"
    result = _resource_gate(
        "require_lmo_full_imagenet_resources configs/research/train.py",
        train_protocol="lmo_full_imagenet",
        hierarchy=LMO_FULL_HIERARCHY,
        missing=(official,),
    )
    assert result.returncode == 0, result.stderr

    for missing, expected in (
        ((CONVNEXT_CHECKPOINT,), "convnext_base_1k_224_ema.pth"),
        ((LMO_FULL_HIERARCHY,), "reused_v1.npz"),
        (("/workspace/gdrnpp/datasets/BOP_DATASETS/lmo/test",), "lmo/test"),
    ):
        rejected = _resource_gate(
            "require_lmo_full_imagenet_resources configs/research/train.py",
            train_protocol="lmo_full_imagenet",
            hierarchy=LMO_FULL_HIERARCHY,
            missing=missing,
            check=False,
        )
        assert rejected.returncode != 0
        assert expected in rejected.stderr


def test_lmo_full_gate_stops_on_failed_model_preflight():
    rejected = _resource_gate(
        "fake_server_preflight=1\n"
        "require_lmo_full_imagenet_resources configs/research/train.py",
        train_protocol="lmo_full_imagenet",
        hierarchy=LMO_FULL_HIERARCHY,
        check=False,
    )
    assert rejected.returncode != 0
    assert "LM-O full-training preflight failed" in rejected.stderr


def test_resource_gate_precedes_run_directory_creation():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert source.index('require_profile_resources "${profile}" "${config}"') < source.index(
        'run_id="$(next_run_id "${mode}")"'
    )
    assert source.index("require_profile_resources()") < source.index("runtime_gate()")
    assert source.index("resolve_resource_profile()") < source.index("runtime_gate()")


def test_check_host_accepts_a_fresh_profile_without_runtime_directories(tmp_path):
    """`create` runs check_host before making outputs/cache/home, so it must not need them."""
    root = tmp_path / "docker_data/chx"
    root.mkdir(parents=True)
    for directory in ("datasets", "weights", "outputs", "cache", "home"):
        assert not (root / directory).exists()
    result = _source_and_run(
        'machine="$(id -un)"\n'
        "gpu_id=0\n"
        f"root={shlex.quote(str(root))}\n"
        "docker_bin=/bin/true\n"
        "nvidia-smi() { echo '0, GPU-uuid, NVIDIA L40, 0 MiB, 46068 MiB, 0 %'; }\n"
        "check_host"
    )
    assert result.returncode == 0
    assert result.stdout.strip().endswith("46068 MiB, 0 %")


def test_check_host_still_requires_the_project_root(tmp_path):
    result = _source_and_run(
        'machine="$(id -un)"\n'
        "gpu_id=0\n"
        f"root={shlex.quote(str(tmp_path / 'absent'))}\n"
        "docker_bin=/bin/true\n"
        "nvidia-smi() { echo gpu; }\n"
        "check_host",
        check=False,
    )
    assert result.returncode != 0
    assert "missing project root" in result.stderr


def test_create_owns_the_runtime_directories():
    source = LAUNCHER.read_text(encoding="utf-8")
    create_block = source[source.index("\n    create)"):source.index("\n    run)")]
    for target in (
        "${root}/datasets",
        "${root}/weights",
        "${root}/datasets/lm_imgn",
        "${root}/outputs",
        "${root}/cache",
        "${root}/cache/gdrnpp_datasets",
        "${root}/home",
        "${root}/home/.cache",
    ):
        assert f'"{target}"' in create_block, target

    # and create must not fabricate real dataset content
    for target in ("lm/test", "lm_imgn/imgn", "VOC2012/JPEGImages"):
        assert target not in create_block, target

    check_host_block = source[source.index("check_host()"):source.index("container_exists()")]
    assert "for directory in" not in check_host_block
    assert "${root}/outputs" not in check_host_block
