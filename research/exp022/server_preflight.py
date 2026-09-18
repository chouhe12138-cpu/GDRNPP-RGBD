#!/usr/bin/env python3
"""Check the container-visible resources an EXP023 LM13 run depends on.

``exp022.preflight`` proves on CPU that the model, hierarchy and loss are
healthy, and the launcher's mount gate proves the bind mounts exist.  Neither
answers the question this script answers: with the container's own environment,
can the LM13 protocol find and register everything it names?  The launcher runs
it inside the container as part of the runtime gate, before any run directory is
created, so a missing render set or a wrong cache path fails before a 160-epoch
run starts rather than hours into it.

Tensor-level verification of the ConvNeXt checkpoint stays in
``exp022.preflight`` (``verify_imagenet_backbone``); this script only proves the
environment points the backbone at a readable file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from detectron2.data import DatasetCatalog
from mmcv import Config

from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.datasets.lm_dataset_d2 import SPLITS_LM
from core.gdrn_modeling.datasets.lm_pbr import LM_13_OBJECTS, SPLITS_LM_PBR
from core.gdrn_modeling.datasets.lm_syn_imgn import SPLITS_LM_IMGN
from research.exp022.dataset_context import resolve_dataset_context
from research.exp022.preflight import resolve_config_path


# The launcher maps these names to a resource profile; anything else is the
# legacy LM-O contract and is not this script's business.
PROFILES = {"lm13_gdrn": "lm13", "lm13_real_only": "lm13", "lm13_pbr": "lm13_pbr"}
SPLIT_TABLES = (SPLITS_LM, SPLITS_LM_IMGN, SPLITS_LM_PBR)
# Representative files: one object's train/test split, one scene's annotations
# and the CAD metadata every LM loader reads.  xyz_crop is deliberately absent --
# the LM13 splits run with require_xyz=False and evaluation does not need it.
LM_REAL_SAMPLE_FILES = (
    "image_set/ape_train.txt",
    "image_set/ape_test.txt",
    "test/000001/scene_gt.json",
    "test/000001/scene_gt_info.json",
    "test/000001/scene_camera.json",
    "models/obj_000001.ply",
    "models/models_info.json",
)
RENDER_SUFFIXES = ("-color.png", "-depth.png", "-pose.txt")


def require_path(path: Path, *, directory: bool = False, readable: bool = False) -> str:
    if directory:
        if not path.is_dir():
            raise FileNotFoundError(f"missing directory: {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")
    if readable and not os.access(path, os.R_OK):
        raise PermissionError(f"not readable: {path}")
    return str(path)


def known_split(name: str) -> dict:
    for table in SPLIT_TABLES:
        if name in table:
            return table[name]
    return {}


def split_config(name: str) -> dict:
    found = known_split(name)
    if not found:
        raise KeyError(f"no split config registered for {name}")
    return found


def lm_root(cfg) -> Path:
    """Every LM split points models_root at BOP_DATASETS/lm/models."""
    for name in list(cfg.DATASETS.TRAIN) + list(cfg.DATASETS.TEST):
        models_root = known_split(str(name)).get("models_root")
        if models_root:
            return Path(models_root).parent
    raise ValueError("config declares no LM split with a models_root")


def check_lm_real(cfg) -> dict:
    """LM real images, annotations and CAD; the BOP targets live beside them."""
    root = lm_root(cfg)
    for name in ("test", "image_set", "models"):
        require_path(root / name, directory=True)
    for relative in LM_REAL_SAMPLE_FILES:
        require_path(root / relative)
    return {"lm_root": str(root)}


def check_lm_imgn(cfg) -> dict:
    """DeepIM renders: the 13 train lists plus one object's render triple."""
    name = next((n for n in cfg.DATASETS.TRAIN if str(n).startswith("lm_imgn")), None)
    if name is None:
        return {}
    root = Path(split_config(str(name))["img_root"]).parent
    for relative in ("image_set", "imgn"):
        require_path(root / relative, directory=True)
    for obj in LM_13_OBJECTS:
        require_path(root / "image_set" / f"train_{obj}.txt")
    sample = root / "imgn" / "ape"
    require_path(sample, directory=True)
    for suffix in RENDER_SUFFIXES:
        if not any(sample.glob(f"*{suffix}")):
            raise FileNotFoundError(f"lm_imgn renders missing *{suffix}: {sample}")
    return {"lm_imgn_root": str(root)}


def check_lm_pbr(cfg) -> dict:
    """The BOP/PBR renders the lm13_pbr arm trains on instead of the DeepIM set."""
    name = next((n for n in cfg.DATASETS.TRAIN if str(n).startswith("lm_pbr")), None)
    if name is None:
        return {}
    root = Path(split_config(str(name))["dataset_root"])
    require_path(root, directory=True)
    if not any(root.glob("*/scene_gt.json")):
        raise FileNotFoundError(f"lm_pbr renders have no scenes: {root}")
    return {"lm_pbr_root": str(root)}


def check_voc(cfg) -> dict:
    root = Path(str(cfg.INPUT.BG_IMGS_ROOT))
    jpeg = root / "JPEGImages"
    require_path(jpeg, directory=True)
    if not any(jpeg.glob("*.jpg")):
        raise FileNotFoundError(f"VOC JPEGImages holds no images: {jpeg}")
    return {"voc_root": str(root)}


def check_convnext(cfg) -> dict:
    """The config reads the variable at load time, so compare the two directly."""
    configured = str(cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.get("checkpoint_path", ""))
    if not configured:
        raise RuntimeError(
            "GDRN_CONVNEXT_BASE_WEIGHTS is unset; the ConvNeXt backbone would "
            "initialise without its ImageNet weights"
        )
    env_value = os.environ.get("GDRN_CONVNEXT_BASE_WEIGHTS", "")
    if env_value != configured:
        raise ValueError(
            f"config checkpoint_path={configured!r} but "
            f"GDRN_CONVNEXT_BASE_WEIGHTS={env_value!r}"
        )
    path = Path(configured)
    require_path(path, readable=True)
    return {"convnext_checkpoint": str(path), "convnext_bytes": path.stat().st_size}


def check_records(cfg) -> dict:
    """Register the configured splits and report how many records each yields."""
    register_datasets_in_cfg(cfg)
    return {str(name): len(DatasetCatalog.get(str(name)))
            for name in list(cfg.DATASETS.TRAIN) + list(cfg.DATASETS.TEST)}


def build_report(cfg, *, counts: bool) -> dict:
    name = str(cfg.get("TRAIN_PROTOCOL", {}).get("NAME", ""))
    profile = PROFILES.get(name)
    if profile is None:
        raise ValueError(f"not an EXP023 LM13 protocol: TRAIN_PROTOCOL.NAME={name!r}")
    context = resolve_dataset_context(cfg)
    report = {
        "profile": profile,
        "train_protocol": name,
        "train_datasets": [str(value) for value in cfg.DATASETS.TRAIN],
        "test_dataset": context.test_dataset,
        "object_ids": list(context.object_ids),
        "hierarchy": require_path(context.hierarchy_path),
        "bop_targets": require_path(context.bop_targets, readable=True),
    }
    report.update(check_lm_real(cfg))
    report.update(check_convnext(cfg))
    report.update(check_voc(cfg))
    report.update(check_lm_imgn(cfg) if profile == "lm13" else check_lm_pbr(cfg))
    if counts:
        report["records"] = check_records(cfg)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-records", action="store_true",
                        help="check paths only; do not register the splits to count them")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    cfg = Config.fromfile(str(config_path))
    try:
        report = build_report(cfg, counts=not args.skip_records)
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "FAIL", "config": str(config_path),
                          "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "config": str(config_path), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
