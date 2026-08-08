"""Read and validate per-experiment metadata without touching raw outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import read_json, validate_experiment


def load_experiment_registry(experiments_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(experiments_root.glob("*/EXPERIMENT.json")):
        payload = read_json(path)
        validate_experiment(payload, path.parent.name)
        experiment_id = payload["experiment_id"]
        if experiment_id in seen:
            raise ValueError(f"duplicate experiment_id: {experiment_id}")
        seen.add(experiment_id)
        record = dict(payload)
        record["metadata_path"] = str(path.relative_to(experiments_root.parent.parent))
        records.append(record)
    return records


def registry_payload(experiments_root: Path) -> dict[str, Any]:
    records = load_experiment_registry(experiments_root)
    summaries = [
        {
            "experiment_id": record["experiment_id"],
            "title": record["title"],
            "stage": record["stage"],
            "role": record.get("role"),
            "status": record["status"],
            "decision": record.get("decision"),
            "legacy_import": record["legacy_import"],
            "metadata_path": record["metadata_path"],
        }
        for record in records
    ]
    return {
        "schema_version": 1,
        "experiment_count": len(summaries),
        "experiments": summaries,
    }


def render_registry_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# 实验索引",
        "",
        "本页由各实验目录中的 `EXPERIMENT.json` 生成；详细结果以 RECORD 和原始产物为准。",
        "",
        "| 实验 ID | 阶段 | 角色 | 状态 | 标题 |",
        "|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            "| {experiment_id} | {stage} | {role} | {status} | {title} |".format(
                experiment_id=record["experiment_id"],
                stage=record["stage"],
                role=record.get("role", "—") or "—",
                status=record["status"],
                title=record["title"].replace("|", "\\|"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def compare_generated_registry(
    experiments_root: Path,
    json_path: Path,
    markdown_path: Path,
) -> None:
    payload = registry_payload(experiments_root)
    expected_json = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    expected_markdown = render_registry_markdown(payload["experiments"])
    if json_path.read_text(encoding="utf-8") != expected_json:
        raise RuntimeError(f"experiment JSON index is stale: {json_path}")
    if markdown_path.read_text(encoding="utf-8") != expected_markdown:
        raise RuntimeError(f"experiment Markdown index is stale: {markdown_path}")
