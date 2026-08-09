#!/usr/bin/env python3
"""Validate Stage 3C-0 data, split, checkpoint, and configuration invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mmcv import Config

from core.gdrn_modeling.datasets.lm_pbr import SPLITS_LM_PBR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_WEIGHT_SHA256 = "bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c"
LMO_OBJECT_IDS = {1, 5, 6, 8, 9, 10, 11, 12}
EXPECTED_PBR_ANNOTATIONS = 749_600
EXPECTED_LMO_ANNOTATIONS = 399_950
EXPECTED_VOC_JPEGS = 17_125
EXPECTED_VOC_TABLE_BACKGROUNDS = 538


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/gdrn/lmo_pbr/convnext_stage3c0_pnp_only_local_lmo.py",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    )
    parser.add_argument(
        "--pbr-root",
        type=Path,
        default=PROJECT_ROOT / "datasets/BOP_DATASETS/lmo/train_pbr",
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=PROJECT_ROOT / "datasets/VOCdevkit/VOC2012",
    )
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--expected-seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config.fromfile(str(args.config.resolve()))
    weights = args.weights.resolve()
    pbr_root = args.pbr_root.resolve()
    voc_root = args.voc_root.resolve()

    train_names = tuple(config.DATASETS.TRAIN)
    if train_names == ("lmo_pbr_train",):
        train_scenes = tuple(range(50))
    elif train_names == ("lmo_pbr_stage3_local_train",):
        train_scenes = tuple(SPLITS_LM_PBR["lmo_pbr_stage3_local_train"]["scene_ids"])
    else:
        raise RuntimeError(f"Unexpected control training dataset: {train_names}")

    summary = {
        "config": str(args.config.resolve()),
        "weights_sha256": sha256(weights),
        "train_scenes": list(train_scenes),
        "validation_scenes": [],
        "pbr_scenes": 0,
        "pbr_rgb_images": 0,
        "pbr_depth_images": 0,
        "pbr_annotations": 0,
        "pbr_lmo_annotations": 0,
        "mask_paths_checked": 0,
    }
    if summary["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"Unexpected checkpoint hash: {summary['weights_sha256']}")

    for scene_id in range(50):
        scene = pbr_root / f"{scene_id:06d}"
        if not scene.is_dir():
            raise FileNotFoundError(scene)
        with (scene / "scene_gt.json").open(encoding="utf-8") as handle:
            scene_gt = json.load(handle)
        with (scene / "scene_gt_info.json").open(encoding="utf-8") as handle:
            scene_info = json.load(handle)
        with (scene / "scene_camera.json").open(encoding="utf-8") as handle:
            scene_camera = json.load(handle)
        expected_keys = {str(image_id) for image_id in range(1000)}
        if set(scene_gt) != expected_keys or set(scene_info) != expected_keys:
            raise RuntimeError(f"Incomplete GT metadata in scene {scene_id:06d}")
        if set(scene_camera) != expected_keys:
            raise RuntimeError(f"Incomplete camera metadata in scene {scene_id:06d}")
        rgb_count = len(list((scene / "rgb").glob("*.jpg")))
        depth_count = len(list((scene / "depth").glob("*.png")))
        if rgb_count != 1000 or depth_count != 1000:
            raise RuntimeError(
                f"Scene {scene_id:06d} has RGB/depth counts {rgb_count}/{depth_count}"
            )
        summary["pbr_scenes"] += 1
        summary["pbr_rgb_images"] += rgb_count
        summary["pbr_depth_images"] += depth_count
        for image_key, annotations in scene_gt.items():
            if len(annotations) != len(scene_info[image_key]):
                raise RuntimeError(f"GT/info length mismatch: {scene_id}/{image_key}")
            summary["pbr_annotations"] += len(annotations)
            for annotation_index, annotation in enumerate(annotations):
                if int(annotation["obj_id"]) in LMO_OBJECT_IDS:
                    summary["pbr_lmo_annotations"] += 1
                if args.deep:
                    stem = f"{int(image_key):06d}_{annotation_index:06d}.png"
                    for mask_dir in ("mask", "mask_visib"):
                        path = scene / mask_dir / stem
                        if not path.is_file():
                            raise FileNotFoundError(path)
                        summary["mask_paths_checked"] += 1

    voc_list = voc_root / "ImageSets/Main/diningtable_trainval.txt"
    positive_backgrounds = [
        line.split()[0]
        for line in voc_list.read_text(encoding="utf-8").splitlines()
        if line.split()[1] == "1"
    ]
    missing_backgrounds = [
        name
        for name in positive_backgrounds
        if not (voc_root / "JPEGImages" / f"{name}.jpg").is_file()
    ]
    if missing_backgrounds:
        raise FileNotFoundError(missing_backgrounds[0])
    summary["voc_jpeg_images"] = len(list((voc_root / "JPEGImages").glob("*.jpg")))
    summary["voc_table_backgrounds"] = len(positive_backgrounds)
    expected_counts = {
        "pbr_scenes": 50,
        "pbr_rgb_images": 50_000,
        "pbr_depth_images": 50_000,
        "pbr_annotations": EXPECTED_PBR_ANNOTATIONS,
        "pbr_lmo_annotations": EXPECTED_LMO_ANNOTATIONS,
        "voc_jpeg_images": EXPECTED_VOC_JPEGS,
        "voc_table_backgrounds": EXPECTED_VOC_TABLE_BACKGROUNDS,
    }
    mismatches = {
        key: {"actual": summary[key], "expected": expected}
        for key, expected in expected_counts.items()
        if summary[key] != expected
    }
    if mismatches:
        raise RuntimeError(f"Dataset count mismatch: {mismatches}")

    if not config.MODEL.WEIGHTS:
        raise RuntimeError("PnP-only control must initialize from the official checkpoint")
    if not config.MODEL.POSE_NET.BACKBONE.FREEZE:
        raise RuntimeError("Backbone must be frozen")
    if not config.MODEL.POSE_NET.GEO_HEAD.FREEZE:
        raise RuntimeError("Geometry head must be frozen")
    if config.MODEL.POSE_NET.PNP_NET.FREEZE:
        raise RuntimeError("Patch-PnP must remain trainable")
    if config.MODEL.POSE_NET.BACKBONE.INIT_CFG.pretrained:
        raise RuntimeError("Offline control config must disable timm pretrained download")
    if train_names == ("lmo_pbr_train",):
        expected_schedule = {
            "IMS_PER_BATCH": 48,
            "REFERENCE_BS": 48,
            "TOTAL_EPOCHS": 40,
        }
    elif train_names == ("lmo_pbr_stage3_local_train",):
        expected_schedule = {
            "IMS_PER_BATCH": 4,
            "REFERENCE_BS": 48,
            "TOTAL_EPOCHS": 1,
        }
    else:
        raise RuntimeError(f"Unexpected control training split: {train_names}")
    schedule_mismatches = {
        key: {"actual": int(config.SOLVER[key]), "expected": expected}
        for key, expected in expected_schedule.items()
        if int(config.SOLVER[key]) != expected
    }
    if schedule_mismatches:
        raise RuntimeError(f"Control schedule mismatch: {schedule_mismatches}")
    if train_names == ("lmo_pbr_train",):
        if tuple(config.DATASETS.TEST) != ("lmo_bop_test",):
            raise RuntimeError("Formal control must evaluate LM-O")
        if int(config.TEST.EVAL_PERIOD) != 5 or config.TEST.TEST_BBOX_TYPE != "gt":
            raise RuntimeError("Formal control must evaluate GT-box LM-O every five epochs")
    optimizer = config.SOLVER.OPTIMIZER_CFG
    if (
        optimizer["type"] != "Ranger"
        or float(optimizer["lr"]) != 8e-5
        or float(optimizer["weight_decay"]) != 0.01
    ):
        raise RuntimeError(f"Unexpected control optimizer: {optimizer}")
    if int(config.SEED) != int(args.expected_seed):
        raise RuntimeError(f"Unexpected control seed: {config.SEED}")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
