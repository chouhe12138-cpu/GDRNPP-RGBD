#!/usr/bin/env python3
"""Bind a clean Git release checkout to an immutable environment image."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, utc_now
from .docker_image import inspect_docker_image
from .manifest import (
    collect_git_provenance,
    read_json,
    sha256_file,
    sha256_json,
)


ENVIRONMENT_EXACT_PATHS = {
    ".dockerignore",
    "docker/l40/Dockerfile",
    "docker/l40/Dockerfile.dockerignore",
    "docker/l40/requirements.lock",
    "docker/l40/build_native.sh",
    "docker/l40/10_nvidia.json",
}
ENVIRONMENT_PREFIXES = ("docker/l40/vendor/",)
NATIVE_ROOTS = (
    "core/csrc/fps/",
    "core/csrc/flow/",
    "core/csrc/ransac_voting/",
    "core/csrc/torch_nndistance/",
    "core/csrc/uncertainty_pnp/",
    "lib/egl_renderer/",
)
NATIVE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".cmake",
    ".in",
}
NATIVE_ARTIFACT_ROOTS = (
    "core/csrc/fps",
    "core/csrc/flow",
    "core/csrc/ransac_voting",
    "core/csrc/torch_nndistance",
    "core/csrc/uncertainty_pnp",
    "lib/egl_renderer",
)
REQUIRED_ARTIFACT_PATTERNS = (
    "core/csrc/fps/_ext*.so",
    "core/csrc/flow/flow_cuda*.so",
    "core/csrc/ransac_voting/ransac_voting*.so",
    "core/csrc/torch_nndistance/torch_nndistance_aten*.so",
    "core/csrc/uncertainty_pnp/_ext*.so",
    "core/csrc/uncertainty_pnp/lib/libceres.so*",
    "core/csrc/uncertainty_pnp/lib/libglog.so*",
    "lib/egl_renderer/CppEGLRenderer*.so",
)
RUNTIME_MOUNTPOINTS = (
    "datasets/BOP_DATASETS",
    "datasets/VOCdevkit",
    "pretrained_models",
    "output",
)


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True
    ).strip()


def _is_environment_input(path: str) -> bool:
    if path in ENVIRONMENT_EXACT_PATHS or path.startswith(ENVIRONMENT_PREFIXES):
        return True
    if not path.startswith(NATIVE_ROOTS):
        return False
    name = Path(path).name
    if name in {"setup.py", "CMakeLists.txt"}:
        return True
    return Path(path).suffix.lower() in NATIVE_SUFFIXES


def environment_contract(repo_root: Path, commit: str = "HEAD") -> dict[str, Any]:
    """Hash only inputs whose change requires a new environment/native image."""

    resolved_commit = _git(repo_root, "rev-parse", f"{commit}^{{commit}}")
    raw = subprocess.check_output(
        ["git", "-C", str(repo_root), "ls-tree", "-r", "-z", resolved_commit]
    )
    entries: list[dict[str, str]] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, raw_path = item.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if _is_environment_input(path):
            entries.append(
                {
                    "mode": mode,
                    "object_type": object_type,
                    "object_id": object_id,
                    "path": path,
                }
            )
    if not entries:
        raise RuntimeError("environment contract selected no tracked inputs")
    material = {"schema_version": 1, "files": sorted(entries, key=lambda x: x["path"])}
    return {
        "commit": resolved_commit,
        "sha256": sha256_json(material),
        "file_count": len(entries),
        "files": material["files"],
    }


def _artifact_entry(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        return {"path": relative, "kind": "symlink", "target": os.readlink(path)}
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"unsupported native artifact type: {path}")
    return {
        "path": relative,
        "kind": "file",
        "sha256": sha256_file(path),
        "size_bytes": info.st_size,
    }


def _walk_artifacts(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for directory, subdirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in sorted(subdirs + files):
            path = base / name
            if path.is_symlink() or path.is_file():
                entries.append(_artifact_entry(root, path))
    entries.sort(key=lambda entry: entry["path"])
    return entries


def _validate_required_artifacts(root: Path) -> None:
    missing = [pattern for pattern in REQUIRED_ARTIFACT_PATTERNS if not list(root.glob(pattern))]
    if missing:
        raise RuntimeError(f"environment image is missing native artifacts: {missing}")


def _extract_artifacts(
    repo_root: Path, image_id: str, docker_bin: Path
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="gdrn-native-overlay-") as temporary:
        export_root = Path(temporary).resolve()
        roots = " ".join(NATIVE_ARTIFACT_ROOTS)
        script = (
            "set -Eeuo pipefail; cd /workspace/gdrnpp; "
            f"find {roots} \\( -type f -o -type l \\) "
            "\\( -name '*.so' -o -name '*.so.*' \\) "
            "-exec cp -P --parents '{}' /export/ \\;; "
            "if [[ -d core/csrc/uncertainty_pnp/lib ]]; then "
            "cp -RP --no-preserve=ownership --parents "
            "core/csrc/uncertainty_pnp/lib /export/; fi"
        )
        subprocess.run(
            [
                str(docker_bin),
                "run",
                "--rm",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--mount",
                f"type=bind,src={export_root},dst=/export",
                "--entrypoint",
                "/bin/bash",
                image_id,
                "-lc",
                script,
            ],
            check=True,
        )
        _validate_required_artifacts(export_root)
        entries = _walk_artifacts(export_root)

        for entry in entries:
            destination = repo_root / entry["path"]
            if destination.exists() or destination.is_symlink():
                if _artifact_entry(repo_root, destination) != entry:
                    raise RuntimeError(
                        "existing native artifact differs; use a fresh release checkout: "
                        f"{destination}"
                    )
        for entry in entries:
            source = export_root / entry["path"]
            destination = repo_root / entry["path"]
            if destination.exists() or destination.is_symlink():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if entry["kind"] == "symlink":
                destination.symlink_to(entry["target"])
            else:
                shutil.copy2(source, destination)
    return entries


def _native_abi(image_id: str, docker_bin: Path) -> dict[str, Any]:
    program = """
import hashlib, json, platform, sys, sysconfig
from pathlib import Path
import bop_renderer, detectron2, torch
bop_path = Path(bop_renderer.__file__).resolve()
print(json.dumps({
    "python_version": platform.python_version(),
    "python_soabi": sysconfig.get_config_var("SOABI"),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    "detectron2_version": getattr(detectron2, "__version__", "unknown"),
    "bop_renderer_module": str(bop_path),
    "bop_renderer_sha256": hashlib.sha256(bop_path.read_bytes()).hexdigest(),
}, sort_keys=True))
"""
    output = subprocess.check_output(
        [
            str(docker_bin),
            "run",
            "--rm",
            "--entrypoint",
            "python",
            image_id,
            "-c",
            program,
        ],
        text=True,
    )
    return json.loads(output.strip().splitlines()[-1])


def prepare_release(
    repo_root: Path, image_ref: str, docker_bin: Path = Path("/usr/bin/docker")
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    source = collect_git_provenance(repo_root)
    if not source["source_tree_clean"]:
        raise RuntimeError("release preparation requires a clean Git checkout")
    if not source["source_head_detached"]:
        raise RuntimeError("release preparation requires a detached Git checkout")
    image = inspect_docker_image(image_ref, docker_bin)
    build_source = image["build_source_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", build_source):
        raise RuntimeError("environment image has no valid build-source commit label")
    current_contract = environment_contract(repo_root, source["source_git_commit"])
    build_contract = environment_contract(repo_root, build_source)
    if current_contract["sha256"] != build_contract["sha256"]:
        current_paths = {item["path"]: item["object_id"] for item in current_contract["files"]}
        build_paths = {item["path"]: item["object_id"] for item in build_contract["files"]}
        changed = sorted(
            path
            for path in current_paths.keys() | build_paths.keys()
            if current_paths.get(path) != build_paths.get(path)
        )
        raise RuntimeError(
            "release changes environment/native inputs and requires a rebuilt image: "
            f"{changed[:20]}"
        )
    artifacts = _extract_artifacts(repo_root, image["image_id"], docker_bin)
    for relative in RUNTIME_MOUNTPOINTS:
        (repo_root / relative).mkdir(parents=True, exist_ok=True)
    native_artifact_sha = sha256_json({"artifacts": artifacts})
    abi = _native_abi(image["image_id"], docker_bin)
    binding = {
        "schema_version": 1,
        "created_at": utc_now(),
        "release": {
            "source_git_commit": source["source_git_commit"],
            "source_tree_clean": True,
            "source_head_detached": True,
            "environment_contract_sha256": current_contract["sha256"],
        },
        "environment": {
            "environment_image_ref": image_ref,
            "environment_image_id": image["image_id"],
            "environment_build_source_commit": build_source,
            "environment_contract_sha256": build_contract["sha256"],
            "native_artifact_manifest_sha256": native_artifact_sha,
            "native_abi": abi,
        },
        "native_artifacts": artifacts,
    }
    binding_path = repo_root / ".local/environment_binding.json"
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(binding_path, binding)
    verify_release_binding(repo_root, binding_path, image["image_id"])
    return {"status": "PASS", "binding_path": str(binding_path), **binding}


def verify_release_binding(
    repo_root: Path,
    binding_path: Path,
    expected_image_id: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    binding = read_json(binding_path.resolve())
    if binding.get("schema_version") != 1:
        raise ValueError("environment binding schema_version must be 1")
    source = collect_git_provenance(repo_root)
    if not source["source_tree_clean"]:
        raise RuntimeError("bound source checkout is not clean")
    if not source["source_head_detached"]:
        raise RuntimeError("bound source checkout is not detached")
    release = binding.get("release", {})
    environment = binding.get("environment", {})
    if release.get("source_git_commit") != source["source_git_commit"]:
        raise RuntimeError("environment binding source commit mismatch")
    contract = environment_contract(repo_root, source["source_git_commit"])
    if contract["sha256"] != release.get("environment_contract_sha256"):
        raise RuntimeError("release environment contract changed after binding")
    if contract["sha256"] != environment.get("environment_contract_sha256"):
        raise RuntimeError("release is incompatible with the environment image")
    if expected_image_id and environment.get("environment_image_id") != expected_image_id:
        raise RuntimeError("environment image ID does not match the release binding")
    artifacts = [_artifact_entry(repo_root, repo_root / item["path"]) for item in binding["native_artifacts"]]
    if artifacts != binding["native_artifacts"]:
        raise RuntimeError("native artifact paths or hashes changed after binding")
    artifact_sha = sha256_json({"artifacts": artifacts})
    if artifact_sha != environment.get("native_artifact_manifest_sha256"):
        raise RuntimeError("native artifact manifest hash mismatch")
    return {
        "status": "PASS",
        "source_git_commit": source["source_git_commit"],
        "environment_image_id": environment["environment_image_id"],
        "environment_build_source_commit": environment[
            "environment_build_source_commit"
        ],
        "environment_contract_sha256": contract["sha256"],
        "native_artifact_manifest_sha256": artifact_sha,
        "native_artifacts_checked": len(artifacts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", type=Path, required=True)
    prepare.add_argument("--image", required=True)
    prepare.add_argument("--docker-bin", type=Path, default=Path("/usr/bin/docker"))
    verify = subparsers.add_parser("verify")
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.add_argument("--binding", type=Path, required=True)
    verify.add_argument("--image-id")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_release(args.repo_root, args.image, args.docker_bin)
    else:
        result = verify_release_binding(args.repo_root, args.binding, args.image_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
