#!/usr/bin/env python3
"""Check Stage 3C-1 configs, official weights, and identity initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.heads.quality_coverage_attention import (
    QualityCoverageAttention,
)
from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from core.gdrn_modeling.datasets.lm_pbr import SPLITS_LM_PBR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_WEIGHT_SHA256 = "bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c"


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
        / "configs/gdrn/lmo_pbr/convnext_stage3c1_quality_coverage_lmo.py",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config.fromfile(str(args.config.resolve()))
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    weight_hash = sha256(weights)
    if weight_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"Unexpected official checkpoint hash: {weight_hash}")

    pose_cfg = cfg.MODEL.POSE_NET
    expected_freeze = {
        "backbone": bool(pose_cfg.BACKBONE.FREEZE),
        "geometry": bool(pose_cfg.GEO_HEAD.FREEZE),
        "pnp": bool(pose_cfg.PNP_NET.FREEZE),
    }
    if expected_freeze != {"backbone": True, "geometry": True, "pnp": True}:
        raise RuntimeError(f"C1 must freeze every official component: {expected_freeze}")
    if not pose_cfg.QUALITY_COVERAGE.ENABLED or pose_cfg.QUALITY_COVERAGE.FREEZE:
        raise RuntimeError("C1 must enable and train only QUALITY_COVERAGE")
    if tuple(cfg.DATASETS.TRAIN) != ("lmo_pbr_train",):
        raise RuntimeError("Formal C1 must use all 50 LM-PBR scenes")
    configured_scenes = tuple(
        SPLITS_LM_PBR["lmo_pbr_train"].get("scene_ids", range(50))
    )
    if configured_scenes != tuple(range(50)):
        raise RuntimeError(f"lmo_pbr_train does not cover scenes 0-49: {configured_scenes}")
    if tuple(cfg.DATASETS.TEST) != ("lmo_bop_test",):
        raise RuntimeError("Formal C1 must evaluate LM-O")
    if cfg.TEST.TEST_BBOX_TYPE != "gt" or cfg.TEST.USE_PNP:
        raise RuntimeError("C1 must isolate direct R,t using GT boxes")

    expected_solver = {
        "IMS_PER_BATCH": 48,
        "REFERENCE_BS": 48,
        "TOTAL_EPOCHS": 40,
        "CHECKPOINT_PERIOD": 5,
        "MAX_TO_KEEP": 3,
    }
    actual_solver = {key: int(cfg.SOLVER[key]) for key in expected_solver}
    if actual_solver != expected_solver:
        raise RuntimeError(f"Unexpected formal schedule: {actual_solver}")
    if int(cfg.TEST.EVAL_PERIOD) != 5:
        raise RuntimeError("LM-O evaluation period must be five epochs")
    if not cfg.SOLVER.BEST_CHECKPOINT.ENABLED:
        raise RuntimeError("Best-one plus latest-two retention must be enabled")

    module = QualityCoverageAttention(
        coor_channels=5,
        num_regions=64,
        hidden_dim=pose_cfg.QUALITY_COVERAGE.HIDDEN_DIM,
        max_residual=pose_cfg.QUALITY_COVERAGE.MAX_RESIDUAL,
    )
    coor = torch.randn(2, 5, 16, 16)
    region = torch.softmax(torch.randn(2, 64, 16, 16), dim=1)
    mask = torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        output = module(coor, region, mask)
    if not torch.equal(output, region):
        raise RuntimeError("Quality/coverage module is not an exact identity at initialization")

    cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    model, _ = build_model_optimizer(cfg, is_test=True)
    checkpoint = torch.load(weights, map_location="cpu")
    official_state = checkpoint.get("model", checkpoint)
    official_state = {
        key.removeprefix("_module."): value
        for key, value in official_state.items()
    }
    incompatible = model.load_state_dict(official_state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if not missing or not all(key.startswith("quality_coverage_net.") for key in missing):
        raise RuntimeError(f"Unexpected missing checkpoint tensors: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected official checkpoint tensors: {unexpected}")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names or not all(name.startswith("quality_coverage_net.") for name in trainable_names):
        raise RuntimeError(f"Unexpected trainable tensors: {trainable_names}")

    summary = {
        "status": "PASS",
        "config": str(args.config.resolve()),
        "official_weight_sha256": weight_hash,
        "train_scenes": 50,
        "lmo_objects": 8,
        "identity_initialization": True,
        "trainable_module_parameters": sum(p.numel() for p in module.parameters()),
        "trainable_tensors": len(trainable_names),
        "official_shared_tensors_loaded": len(official_state),
        "new_checkpoint_tensors": len(missing),
        "unexpected_checkpoint_tensors": len(unexpected),
        "schedule": actual_solver,
        "eval_period_epochs": int(cfg.TEST.EVAL_PERIOD),
        "checkpoint_policy": "best_1_plus_latest_2",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
