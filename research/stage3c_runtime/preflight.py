#!/usr/bin/env python3
"""Validate the matched B/C2 formal or smoke configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_WEIGHT_SHA256 = "bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c"
CONFIGS = {
    "B": PROJECT_ROOT
    / "configs/gdrn/lmo_pbr/convnext_stage3c0_pnp_only_lmo.py",
    "C2": PROJECT_ROOT
    / "configs/gdrn/lmo_pbr/convnext_stage3c2_pnp_quality_coverage_lmo.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=sorted(CONFIGS))
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    )
    parser.add_argument("--build-model", action="store_true")
    return parser.parse_args()


def validate_config(role: str, cfg: Config) -> dict[str, object]:
    pose = cfg.MODEL.POSE_NET
    if not pose.BACKBONE.FREEZE or not pose.GEO_HEAD.FREEZE:
        raise RuntimeError("B and C2 must freeze the backbone and geometry head")
    if pose.PNP_NET.FREEZE:
        raise RuntimeError("B and C2 must train Patch-PnP")
    if int(cfg.SEED) != 20260731:
        raise RuntimeError(f"Unexpected formal seed: {cfg.SEED}")
    if cfg.TEST.TEST_BBOX_TYPE != "gt" or cfg.TEST.USE_PNP:
        raise RuntimeError("Formal pose must use GT boxes and direct R,t")

    is_smoke = tuple(cfg.DATASETS.TRAIN) == ("lmo_pbr_stage3_local_train",)
    expected_schedule = (
        {"IMS_PER_BATCH": 4, "REFERENCE_BS": 48, "TOTAL_EPOCHS": 1}
        if is_smoke
        else {"IMS_PER_BATCH": 48, "REFERENCE_BS": 48, "TOTAL_EPOCHS": 40}
    )
    actual_schedule = {key: int(cfg.SOLVER[key]) for key in expected_schedule}
    if actual_schedule != expected_schedule:
        raise RuntimeError(f"Unexpected schedule: {actual_schedule}")
    if not is_smoke:
        if tuple(cfg.DATASETS.TRAIN) != ("lmo_pbr_train",):
            raise RuntimeError("Formal B/C2 must use all 50 LM-PBR scenes")
        if tuple(cfg.DATASETS.TEST) != ("lmo_bop_test",):
            raise RuntimeError("Formal B/C2 must evaluate LM-O")
        if int(cfg.TEST.EVAL_PERIOD) != 5:
            raise RuntimeError("Formal B/C2 must evaluate every five epochs")

    quality = pose.get("QUALITY_COVERAGE", {})
    if role == "B":
        if quality.get("ENABLED", False):
            raise RuntimeError("B must not enable quality/coverage")
        expected_base_lr = 8e-5
        expected_pnp_lr = 8e-5
    else:
        if not quality.get("ENABLED", False) or quality.get("FREEZE", False):
            raise RuntimeError("C2 must enable and train quality/coverage")
        if float(quality.LR_MULT) != 1.0 or float(pose.PNP_NET.LR_MULT) != 0.1:
            raise RuntimeError("C2 learning-rate multipliers must be 1.0 and 0.1")
        expected_base_lr = 8e-4
        expected_pnp_lr = 8e-5

    optimizer = cfg.SOLVER.OPTIMIZER_CFG
    if (
        optimizer.type != "Ranger"
        or float(optimizer.lr) != expected_base_lr
        or float(optimizer.weight_decay) != 0.01
    ):
        raise RuntimeError(f"Unexpected optimizer: {optimizer}")

    artifacts = cfg.RUN_ARTIFACTS
    expected_artifacts = {
        "STRUCTURED_LAYOUT": True,
        "COMPACT_LOG": True,
        "TENSORBOARD": False,
        "SKIP_DUPLICATE_FINAL_EVAL": True,
    }
    actual_artifacts = {key: artifacts.get(key) for key in expected_artifacts}
    if actual_artifacts != expected_artifacts:
        raise RuntimeError(f"Unexpected artifact policy: {actual_artifacts}")

    return {
        "role": role,
        "mode": "smoke" if is_smoke else "formal",
        "schedule": actual_schedule,
        "base_lr": expected_base_lr,
        "pnp_lr": expected_pnp_lr,
        "quality_lr": None if role == "B" else 8e-4,
        "output_dir": cfg.OUTPUT_DIR,
    }


def validate_model(role: str, cfg: Config, weights: Path) -> dict[str, object]:
    cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    model, optimizer = build_model_optimizer(cfg, is_test=False)
    checkpoint = torch.load(weights, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    state = {key.removeprefix("_module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint tensors: {unexpected}")
    if role == "B" and missing:
        raise RuntimeError(f"B must strictly load the official model: {missing}")
    if role == "C2" and (
        not missing
        or not all(name.startswith("quality_coverage_net.") for name in missing)
    ):
        raise RuntimeError(f"Unexpected C2 missing tensors: {missing}")

    trainable = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    allowed = ("pnp_net.",) if role == "B" else ("pnp_net.", "quality_coverage_net.")
    if not trainable or not all(name.startswith(allowed) for name in trainable):
        raise RuntimeError(f"Unexpected trainable tensors: {trainable}")
    if role == "C2" and not any(name.startswith("quality_coverage_net.") for name in trainable):
        raise RuntimeError("C2 has no trainable quality/coverage tensors")

    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    group_lrs: dict[str, set[float]] = {}
    for group in optimizer.param_groups:
        names = [
            parameter_names[id(parameter)]
            for parameter in group["params"]
            if id(parameter) in parameter_names
        ]
        prefixes = {
            "quality" if name.startswith("quality_coverage_net.") else "pnp"
            for name in names
        }
        for prefix in prefixes:
            group_lrs.setdefault(prefix, set()).add(float(group["lr"]))
    expected_lrs = {"pnp": {8e-5}}
    if role == "C2":
        expected_lrs["quality"] = {8e-4}
    if group_lrs != expected_lrs:
        raise RuntimeError(f"Unexpected optimizer parameter-group LRs: {group_lrs}")

    return {
        "trainable_tensors": len(trainable),
        "missing_official_tensors": len(missing),
        "optimizer_group_lrs": {
            key: sorted(values) for key, values in group_lrs.items()
        },
    }


def main() -> int:
    args = parse_args()
    config_path = (args.config or CONFIGS[args.role]).resolve()
    weights = args.weights.resolve()
    cfg = Config.fromfile(str(config_path))
    result = validate_config(args.role, cfg)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    weight_hash = sha256(weights)
    if weight_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"Unexpected official checkpoint hash: {weight_hash}")
    result["official_weight_sha256"] = weight_hash
    result["config"] = str(config_path)
    if args.build_model:
        result.update(validate_model(args.role, cfg, weights))
    result["status"] = "PASS"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
