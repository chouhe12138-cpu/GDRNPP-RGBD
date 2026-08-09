"""Compact, structured summaries derived from formal runtime logs."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, utc_now


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP_RE = re.compile(r"^(?P<timestamp>\d{8}_\d{6})\|")
WARNING_RE = re.compile(
    r"WRN\||\bWARNING\b|\b(?:User|Runtime|Future|Deprecation)Warning\b",
    re.IGNORECASE,
)


def _canonical_warning(line: str) -> tuple[str, str | None]:
    clean = ANSI_RE.sub("", line).strip()
    match = TIMESTAMP_RE.match(clean)
    timestamp = match.group("timestamp") if match else None
    if match:
        clean = clean[match.end() :]
    return clean, timestamp


def warning_summary(console_log: Path) -> dict[str, Any]:
    if not console_log.is_file():
        raise FileNotFoundError(console_log)
    records: dict[str, dict[str, Any]] = {}
    with console_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not WARNING_RE.search(line):
                continue
            message, timestamp = _canonical_warning(line)
            record = records.get(message)
            seen = timestamp or f"line:{line_number}"
            if record is None:
                records[message] = {
                    "message": message,
                    "count": 1,
                    "first_seen": seen,
                    "last_seen": seen,
                }
            else:
                record["count"] += 1
                record["last_seen"] = seen
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source": str(console_log),
        "unique_warnings": len(records),
        "total_warning_occurrences": sum(
            record["count"] for record in records.values()
        ),
        "warnings": list(records.values()),
    }


def write_warning_summary(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    payload = warning_summary(run_dir / "train" / "console.log")
    payload["source"] = "train/console.log"
    atomic_write_json(run_dir / "summary" / "warnings.json", payload)
    return payload


def compact_and_write_warning_summary(run_dir: Path) -> dict[str, Any]:
    """Keep the first copy of repeated warning lines and retain counts in JSON."""

    run_dir = run_dir.resolve()
    console_log = run_dir / "train" / "console.log"
    payload = warning_summary(console_log)
    seen: set[str] = set()
    temporary = console_log.with_name(f".{console_log.name}.compact.tmp")
    with console_log.open("r", encoding="utf-8", errors="replace") as source, temporary.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if WARNING_RE.search(line):
                message, _timestamp = _canonical_warning(line)
                if message in seen:
                    continue
                seen.add(message)
            target.write(line)
    temporary.replace(console_log)
    payload["source"] = "train/console.log"
    atomic_write_json(run_dir / "summary" / "warnings.json", payload)
    return payload


def load_or_write_warning_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir.resolve() / "summary" / "warnings.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return compact_and_write_warning_summary(run_dir)
