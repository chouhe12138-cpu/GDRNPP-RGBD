from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def write_results(output_dir: str, metadata: Dict[str, Any], summaries: Dict[str, Any]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "diagnostics": summaries}
    (out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    lines = ["# Pose Structure Diagnostic Summary", "", "## Run metadata", ""]
    for k, v in metadata.items():
        lines.append(f"- **{k}**: `{_fmt(v)}`")
    for name, summary in summaries.items():
        lines += ["", f"## {name}", "", "```json", json.dumps(summary, indent=2, ensure_ascii=False), "```"]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
