#!/usr/bin/env python3
"""Register and validate the Stage 3A PBR calibration dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.pbr_validation.pbr_dataset import build_validation_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbr-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--path-check-stride", default=50, type=int)
    args = parser.parse_args()
    records, summary = build_validation_records(args.pbr_root, args.manifest, args.models_root)
    missing = []
    checked_records = records[:: max(1, args.path_check_stride)]
    for record in checked_records:
        for path_key in ("file_name", "depth_file"):
            if not Path(record[path_key]).exists():
                missing.append(record[path_key])
        for annotation in record["annotations"]:
            for path_key in ("mask_visib_file", "mask_full_file"):
                if not Path(annotation[path_key]).exists():
                    missing.append(annotation[path_key])
    summary["missing_files"] = len(missing)
    summary["path_checked_images"] = len(checked_records)
    summary["path_check_stride"] = args.path_check_stride
    if missing:
        summary["first_missing"] = missing[:10]
    print(json.dumps(summary, indent=2))
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
