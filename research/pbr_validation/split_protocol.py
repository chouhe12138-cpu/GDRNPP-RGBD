"""Frozen scene-level split protocol for LM PBR validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Dict, Iterable, List
import zipfile


SEED = 20260731
ALL_SCENES = tuple(range(50))
VALIDATION_SCENES = (12, 13, 14)
TRAIN_SCENES = tuple(scene for scene in ALL_SCENES if scene not in VALIDATION_SCENES)
DIAGNOSTIC_IMAGES_PER_SCENE = 500


def select_diagnostic_images(
    image_ids: Iterable[int], scene_id: int, count: int = DIAGNOSTIC_IMAGES_PER_SCENE
) -> List[int]:
    ids = sorted(int(image_id) for image_id in image_ids)
    if len(ids) < count:
        raise ValueError(f"scene {scene_id} has {len(ids)} images, fewer than requested {count}")
    generator = random.Random(SEED + int(scene_id))
    return sorted(generator.sample(ids, count))


def archive_index_sha256(path: Path) -> str:
    """Hash ZIP metadata without re-reading the 21 GB archive payload."""

    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for item in sorted(archive.infolist(), key=lambda value: value.filename):
            digest.update(
                f"{item.filename}\t{item.CRC}\t{item.file_size}\t{item.compress_size}\n".encode("utf-8")
            )
    return digest.hexdigest()


def build_manifest(pbr_root: Path, archive_path: Path) -> Dict[str, object]:
    pbr_root = pbr_root.resolve()
    extracted_scenes = sorted(
        int(path.name)
        for path in pbr_root.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "scene_gt.json").exists()
    )
    missing_validation = sorted(set(VALIDATION_SCENES) - set(extracted_scenes))
    if missing_validation:
        raise FileNotFoundError(f"validation scenes are not extracted: {missing_validation}")

    diagnostic_images = {}
    for scene_id in VALIDATION_SCENES:
        gt_path = pbr_root / f"{scene_id:06d}" / "scene_gt.json"
        with gt_path.open(encoding="utf-8") as handle:
            image_ids = map(int, json.load(handle).keys())
        diagnostic_images[f"{scene_id:06d}"] = select_diagnostic_images(image_ids, scene_id)

    return {
        "protocol": "EXP-20260731-003-pbr-validation-calibration",
        "seed": SEED,
        "dataset": "LM train_pbr",
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_index_sha256": archive_index_sha256(archive_path),
        "archive_expected_scenes": list(ALL_SCENES),
        "archive_expected_images": 50000,
        "currently_extracted_scenes": extracted_scenes,
        "currently_extracted_images": len(extracted_scenes) * 1000,
        "train_scenes": list(TRAIN_SCENES),
        "train_images_when_fully_extracted": len(TRAIN_SCENES) * 1000,
        "validation_scenes": list(VALIDATION_SCENES),
        "validation_reserved_images": len(VALIDATION_SCENES) * 1000,
        "diagnostic_images_per_scene": DIAGNOSTIC_IMAGES_PER_SCENE,
        "diagnostic_images_total": len(VALIDATION_SCENES) * DIAGNOSTIC_IMAGES_PER_SCENE,
        "diagnostic_images": diagnostic_images,
        "scene_disjoint": True,
        "official_checkpoint_may_have_seen_validation_scenes": True,
        "formal_model_selection_requires_retraining_without_validation_scenes": True,
    }
