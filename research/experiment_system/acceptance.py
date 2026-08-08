"""Non-overwriting acceptance checks for legacy experiment evidence."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import read_json, sha256_file


ACCEPTANCE_STATUSES = {"PASS", "PASS_WITH_NOTE", "PENDING_EXTERNAL", "CONFLICT"}


def _repo_path(repo_root: Path, relative: str) -> Path:
    path = repo_root / relative
    if not path.exists():
        return path
    return path


def _read_value(check: dict[str, Any], repo_root: Path) -> Any:
    payload: Any = json.loads(
        _repo_path(repo_root, check["path"]).read_text(encoding="utf-8")
    )
    for key in check.get("pointer", []):
        payload = payload[key]
    selector = check.get("selector")
    if selector:
        matches = [
            item
            for item in payload
            if all(item.get(key) == value for key, value in selector["where"].items())
        ]
        if len(matches) != 1:
            raise ValueError(
                f"selector expected one match in {check['path']}, found {len(matches)}"
            )
        payload = matches[0][selector["field"]]
    return payload


def _check_json(check: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    actual = _read_value(check, repo_root)
    expected = check["expected"]
    tolerance = float(check.get("tolerance", 0.0))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        passed = abs(float(actual) - float(expected)) <= tolerance
    else:
        passed = actual == expected
    return {
        "kind": "json_value",
        "path": check["path"],
        "name": check["name"],
        "expected": expected,
        "actual": actual,
        "tolerance": tolerance,
        "status": "PASS" if passed else "FAIL",
    }


def _check_csv(check: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    path = _repo_path(repo_root, check["path"])
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = sum(1 for _row in csv.reader(handle)) - 1
    expected = int(check["expected_rows"])
    return {
        "kind": "csv_data_rows",
        "path": check["path"],
        "expected": expected,
        "actual": rows,
        "status": "PASS" if rows == expected else "FAIL",
    }


def _check_hash_manifest(check: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    manifest_path = _repo_path(repo_root, check["path"])
    manifest = read_json(manifest_path)
    base = _repo_path(repo_root, check["base"])
    failures: list[dict[str, str]] = []
    for relative, expected in sorted(manifest.items()):
        path = base / relative
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append({"path": relative, "reason": "sha256_mismatch"})
    return {
        "kind": "hash_manifest",
        "path": check["path"],
        "files_checked": len(manifest),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def _check_config(relative: str, repo_root: Path) -> dict[str, Any]:
    path = _repo_path(repo_root, relative)
    if not path.is_file():
        return {
            "kind": "config",
            "path": relative,
            "status": "FAIL",
            "reason": "missing",
        }
    try:
        from mmcv import Config

        resolved = Config.fromfile(str(path))
        output_dir = resolved.get("OUTPUT_DIR")
    except Exception as exc:  # pragma: no cover - exact third-party errors vary
        return {
            "kind": "config",
            "path": relative,
            "status": "FAIL",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "kind": "config",
        "path": relative,
        "sha256": sha256_file(path),
        "resolved_output_dir": output_dir,
        "status": "PASS",
    }


def _check_checkpoint(check: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    path = _repo_path(repo_root, check["path"])
    external = bool(check.get("external", False))
    if not path.is_file():
        return {
            "kind": "checkpoint",
            "id": check["id"],
            "path": check["path"],
            "status": "PENDING_EXTERNAL" if external else "FAIL",
            "reason": "not available in this workspace",
        }
    actual = sha256_file(path)
    expected = check["sha256"]
    return {
        "kind": "checkpoint",
        "id": check["id"],
        "path": check["path"],
        "size_bytes": path.stat().st_size,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "status": "PASS" if actual == expected else "FAIL",
    }


def evaluate_acceptance(
    spec_path: Path,
    repo_root: Path,
    git_commit: str,
) -> dict[str, Any]:
    spec = read_json(spec_path)
    if spec.get("schema_version") != 1:
        raise ValueError(f"invalid acceptance spec schema: {spec_path}")
    experiment_id = spec["experiment_id"]
    if spec_path.parent.name != experiment_id:
        raise ValueError(f"acceptance spec directory does not match {experiment_id}")
    status_on_pass = spec.get("status_on_pass", "PASS")
    if status_on_pass not in ACCEPTANCE_STATUSES - {"CONFLICT"}:
        raise ValueError(f"invalid status_on_pass: {status_on_pass}")

    checks: list[dict[str, Any]] = []
    checks.extend(_check_config(path, repo_root) for path in spec.get("configs", []))
    checks.extend(
        _check_checkpoint(check, repo_root) for check in spec.get("checkpoints", [])
    )
    for item in spec.get("required_paths", []):
        path = _repo_path(repo_root, item["path"])
        present = path.is_dir() if item.get("type") == "directory" else path.is_file()
        external = bool(item.get("external", False))
        checks.append(
            {
                "kind": "required_path",
                "path": item["path"],
                "status": (
                    "PASS" if present else "PENDING_EXTERNAL" if external else "FAIL"
                ),
            }
        )
    for check in spec.get("json_checks", []):
        try:
            checks.append(_check_json(check, repo_root))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            checks.append(
                {
                    "kind": "json_value",
                    "path": check["path"],
                    "name": check["name"],
                    "status": "FAIL",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    for check in spec.get("csv_checks", []):
        try:
            checks.append(_check_csv(check, repo_root))
        except (FileNotFoundError, ValueError) as exc:
            checks.append(
                {
                    "kind": "csv_data_rows",
                    "path": check["path"],
                    "status": "FAIL",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    for check in spec.get("hash_manifests", []):
        try:
            checks.append(_check_hash_manifest(check, repo_root))
        except (FileNotFoundError, ValueError) as exc:
            checks.append(
                {
                    "kind": "hash_manifest",
                    "path": check["path"],
                    "status": "FAIL",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    statuses = {check["status"] for check in checks}
    if "FAIL" in statuses:
        status = "CONFLICT"
    elif "PENDING_EXTERNAL" in statuses:
        status = "PENDING_EXTERNAL"
    else:
        status = status_on_pass
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": status,
        "normalized_decision": spec.get("normalized_decision"),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verified_git_commit": git_commit,
        "scope": spec["scope"],
        "checks": checks,
        "notes": spec.get("notes", []),
    }


def render_acceptance_markdown(result: dict[str, Any]) -> str:
    counts: dict[str, int] = {}
    for check in result["checks"]:
        counts[check["status"]] = counts.get(check["status"], 0) + 1
    lines = [
        f"# {result['experiment_id']} 历史验收",
        "",
        f"- 状态：`{result['status']}`",
        f"- 验收 commit：`{result['verified_git_commit']}`",
        f"- 范围：{result['scope']}",
        f"- 检查数：{len(result['checks'])}（"
        + "，".join(f"{key}={value}" for key, value in sorted(counts.items()))
        + "）",
        "",
        "原始 `RECORD.md` 和历史 `output/` 未被覆盖。详细机器可读结果见 `ACCEPTANCE.json`。",
    ]
    if result.get("normalized_decision"):
        lines.insert(3, f"- 规范化结论：`{result['normalized_decision']}`")
    if result["notes"]:
        lines.extend(["", "## 备注", ""])
        lines.extend(f"- {note}" for note in result["notes"])
    lines.append("")
    return "\n".join(lines)


def write_acceptance(spec_path: Path, result: dict[str, Any]) -> None:
    json_path = spec_path.parent / "ACCEPTANCE.json"
    markdown_path = spec_path.parent / "ACCEPTANCE_CN.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(
            f"acceptance output already exists for {result['experiment_id']}"
        )
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_acceptance_markdown(result), encoding="utf-8")


def current_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def render_acceptance_index(results: list[dict[str, Any]]) -> str:
    lines = [
        "# 历史实验验收总表",
        "",
        "历史原始记录与产物保持只读；本表只记录重新验证结果。",
        "",
        "| 实验 | 状态 | 验收范围 |",
        "|---|---|---|",
    ]
    for result in results:
        scope = result["scope"].replace("|", "\\|")
        lines.append(f"| {result['experiment_id']} | {result['status']} | {scope} |")
    lines.append("")
    return "\n".join(lines)
