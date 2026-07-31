#!/usr/bin/env python3
"""Generate the tracked, deterministic LM-PBR Stage 3A split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.pbr_validation.split_protocol import build_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbr-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.pbr_root, args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_scenes": len(manifest["train_scenes"]),
                "validation_scenes": manifest["validation_scenes"],
                "diagnostic_images": manifest["diagnostic_images_total"],
                "currently_extracted_images": manifest["currently_extracted_images"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
