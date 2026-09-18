"""Describe an evaluation result well enough to trace it later.

The report a paper needs is rarely the `output/` directory name.  This records
which dataset and evaluation protocol produced a score, which bounding boxes
were used, and which checkpoint and toolkit versions were involved.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from mmcv import Config

from research.exp022.dataset_context import resolve_dataset_context
from research.exp022.preflight import resolve_config_path


def _git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        done = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip() or None


def build_manifest(cfg, context, *, checkpoint: str, bop_toolkit_root: Path | None = None,
                   extra: dict | None = None) -> dict:
    """Collect the provenance of one evaluation run."""

    train_protocol = dict(cfg.get("TRAIN_PROTOCOL", {}))
    eval_protocol = dict(cfg.get("EVAL_PROTOCOL", {}))
    dataset_protocol = str(train_protocol.get("NAME", "")) or None
    manifest = {
        "experiment_id": str(cfg.EXPERIMENT_ID),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "dataset_context": context.key,
        "dataset_protocol": dataset_protocol,
        "data_domain": train_protocol.get("DATA_DOMAIN"),
        "train_datasets": [str(name) for name in cfg.DATASETS.TRAIN],
        "test_dataset": context.test_dataset,
        "object_ids": list(context.object_ids),
        "eval_protocol": eval_protocol.get("NAME"),
        "bbox_source": eval_protocol.get("BBOX_SOURCE"),
        "test_bbox_type": str(cfg.TEST.TEST_BBOX_TYPE).lower(),
        "targets_filename": str(cfg.VAL.TARGETS_FILENAME),
        "bop_targets": str(context.bop_targets),
        "use_bop": bool(cfg.VAL.get("USE_BOP", False)),
        "pose_solver": str(cfg.TEST.get("PNP_TYPE", "")) if cfg.TEST.get("USE_PNP", False) else None,
        "checkpoint": str(checkpoint),
        "bop_toolkit_commit": _git_commit(bop_toolkit_root) if bop_toolkit_root else None,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(directory: Path, manifest: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "eval_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bop-toolkit-root", type=Path, default=Path(".local/bop_toolkit"))
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    cfg = Config.fromfile(str(config_path))
    context = resolve_dataset_context(cfg, require_hierarchy=False)
    manifest = build_manifest(cfg, context, checkpoint=args.checkpoint,
                              bop_toolkit_root=args.bop_toolkit_root,
                              extra={"config": str(config_path)})
    path = write_manifest(args.output, manifest)
    print(json.dumps({"status": "COMPLETE", "manifest": str(path),
                      "eval_protocol": manifest["eval_protocol"],
                      "bbox_source": manifest["bbox_source"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
