#!/usr/bin/env python3
"""CPU, synthetic-input EXP020 preflight; never consumes training data.

Builds the full GDRN_DoubleMask with the official checkpoint, freezes
backbone/PNP_NET, keeps only the geometry head trainable, and runs one
do_loss forward/backward + optimizer step for the selected arm (A: REPROJ_LW=0,
B: REPROJ_LW=1). It verifies loss keys, finiteness, gradient routing to the
geometry head, the frozen-parameter contract, and official-checkpoint
compatibility (no added parameters, no strict-load failures).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from detectron2.utils.events import EventStorage
from mmcv import Config

from core.gdrn_modeling.engine.engine_utils import get_K_crop_resize
from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from research.exp013.preflight import (
    PROJECT_ROOT,
    checkpoint_model_state,
    synthetic_full_inputs,
)
from research.run_contract import validate_research_run_config

ARMS = {
    "A": ("configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr/control.py", 0.0),
    "B": ("configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr/reproj.py", 1.0),
}
EXPERIMENT_ID = "EXP-20260909-020-geometry-aware-correspondence-loss"


def validate_config(cfg: Config, expected_reproj_lw: float) -> dict[str, object]:
    contract = validate_research_run_config(
        cfg, mode="formal", expected_experiment_id=EXPERIMENT_ID
    )
    pose = cfg.MODEL.POSE_NET
    loss = pose.LOSS_CFG
    if pose.XYZ_RENDERER != "cpp":
        raise RuntimeError("EXP020 requires the cpp online XYZ renderer")
    if pose.GEO_HEAD.FREEZE or not pose.GEO_HEAD.TRAIN_SUPERVISION:
        raise RuntimeError("EXP020 must train the geometry head with supervision")
    if not pose.BACKBONE.FREEZE or not pose.PNP_NET.FREEZE:
        raise RuntimeError("EXP020 must freeze backbone and PNP_NET")
    if pose.BACKBONE.INIT_CFG.pretrained:
        raise RuntimeError("EXP020 must not download backbone weights")
    if pose.QUALITY_COVERAGE.get("ENABLED", False):
        raise RuntimeError("EXP020 must disable quality/coverage")
    if loss.XYZ_LOSS_TYPE != "L1":
        raise RuntimeError("EXP020 phase 1 requires continuous L1 XYZ")
    for key in ("PM_LW", "CENTROID_LW", "Z_LW", "ROT_LW", "TRANS_LW", "BIND_LW"):
        if float(loss[key]) != 0.0:
            raise RuntimeError(f"EXP020 must zero pose-level loss {key}")
    if float(loss.get("REPROJ_LW", 0.0)) != expected_reproj_lw:
        raise RuntimeError(
            f"Arm expected REPROJ_LW={expected_reproj_lw}, got {loss.get('REPROJ_LW')}"
        )
    return contract


def synthetic_geometry_targets(
    inputs: dict[str, torch.Tensor], out_res: int, batch: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """GT masks/XYZ/region/pose/crop-K for a synthetic online geometry batch."""
    b = batch
    fg = torch.zeros(b, out_res, out_res, device=device)
    fg[:, out_res // 4 : 3 * out_res // 4, out_res // 4 : 3 * out_res // 4] = 1.0

    gt_xyz = (0.35 + 0.3 * torch.rand(b, 3, out_res, out_res, device=device))
    gt_xyz = gt_xyz * fg[:, None]
    gt_region = torch.zeros(b, out_res, out_res, dtype=torch.long, device=device)

    extents = inputs["roi_extents"]
    gt_rot = torch.eye(3, device=device).repeat(b, 1, 1)
    gt_trans = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(b, 1)

    # Build the crop-resized camera matrix exactly like engine_utils does:
    # crop a square of side roi_wh centred on the ROI centre, resize to out_res.
    scale = inputs["roi_whs"][:, 0]
    crop_xy = inputs["roi_centers"] - scale.view(b, -1) / 2
    resize_ratio = out_res / scale.view(b, -1)
    roi_zoom_cams = get_K_crop_resize(inputs["roi_cams"], crop_xy, resize_ratio)
    return {
        "gt_xyz": gt_xyz,
        "gt_mask_trunc": fg.clone(),
        "gt_mask_visib": fg.clone(),
        "gt_mask_obj": fg.clone(),
        "gt_mask_full": fg.clone(),
        "gt_region": gt_region,
        "gt_ego_rot": gt_rot,
        "gt_trans": gt_trans,
        "gt_trans_ratio": torch.zeros(b, 3, device=device),
        "roi_zoom_cams": roi_zoom_cams,
    }


def load_official_state(model: torch.nn.Module, weights_path: Path) -> dict:
    state = checkpoint_model_state(weights_path)
    incompatible = model.load_state_dict(dict(state), strict=False)
    # EXP020 does not add any parameters, so the official checkpoint must load
    # with no missing and no unexpected keys.
    if incompatible.unexpected_keys or incompatible.missing_keys:
        raise RuntimeError(
            "Official checkpoint compatibility broken: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return dict(state)


def check_optimizer(model, optimizer):
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    registered = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if not trainable or any(not name.startswith("geo_head_net.") for name in trainable):
        raise RuntimeError(f"Unexpected trainable tensors: {sorted(trainable)}")
    if any(id(parameter) not in registered for parameter in trainable.values()):
        raise RuntimeError("Trainable geometry parameters missing from the optimizer")
    frozen = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith(("backbone.", "pnp_net."))
    ]
    if not frozen or any(id(parameter) in registered for name, parameter in
                         model.named_parameters() if name in frozen):
        raise RuntimeError("Frozen backbone/PNP parameters leaked into the optimizer")
    return sorted(trainable)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    config_path, expected_lw = ARMS[args.arm]
    cfg = Config.fromfile(str(PROJECT_ROOT / config_path))
    contract = validate_config(cfg, expected_lw)
    cfg.MODEL.DEVICE = args.device
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    cfg.SOLVER.OPTIMIZER_NAME = cfg.SOLVER.OPTIMIZER_CFG.type
    cfg.SOLVER.WEIGHT_DECAY = float(cfg.SOLVER.OPTIMIZER_CFG.weight_decay)
    cfg.DATALOADER.PERSISTENT_WORKERS = False

    torch.manual_seed(42)
    model, optimizer = build_model_optimizer(cfg, is_test=False)
    load_official_state(model, args.weights)
    trainable_names = check_optimizer(model, optimizer)
    model.to(args.device).train()

    batch = 1
    device = torch.device(args.device)
    out_res = int(cfg.MODEL.POSE_NET.OUTPUT_RES)
    inputs = synthetic_full_inputs(device, batch)
    targets = synthetic_geometry_targets(inputs, out_res, batch, device)
    kwargs = dict(inputs, **targets, do_loss=True)

    # REPROJ_LW>0 (arm B) or REPROJ_LW=0 (arm A) must both forward/backward.
    forward_before = _snapshot(model.geo_head_net)
    frozen_before = _snapshot(
        dict(model.named_parameters())
    )

    losses = {}
    with EventStorage():
        _out, loss_dict = model(**kwargs)
    losses["step"] = {k: float(v.detach()) for k, v in loss_dict.items()}
    if expected_lw > 0:
        expected_keys = {
            "loss_coor_x",
            "loss_coor_y",
            "loss_coor_z",
            "loss_mask",
            "loss_mask_full",
            "loss_region",
            "loss_xyz_reproj",
        }
        if not set(loss_dict) == expected_keys:
            raise RuntimeError(f"Unexpected B-mode loss keys: {sorted(loss_dict)}")
    else:
        if "loss_xyz_reproj" in loss_dict:
            raise RuntimeError("A-mode must not produce loss_xyz_reproj")

    total = sum(loss_dict.values())
    if not bool(torch.isfinite(total)):
        raise RuntimeError(f"Non-finite EXP020 loss: {loss_dict}")
    total.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"Missing/non-finite gradient: {name}")
        elif parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter received a gradient: {name}")
    geo_grad_scale = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.geo_head_net.parameters()
        if parameter.grad is not None
    )
    if geo_grad_scale <= 0:
        raise RuntimeError("Geometry head received no gradient")

    optimizer.step()
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise RuntimeError("Optimizer step produced non-finite parameters")
    changed = _value_changed(forward_before, model.geo_head_net)
    if not changed:
        raise RuntimeError("Optimizer step did not change trainable geometry tensors")
    frozen_after = _snapshot(dict(model.named_parameters()))
    frozen_drift = _frozen_value_changed(frozen_before, frozen_after, trainable_names)
    if frozen_drift:
        raise RuntimeError(f"Frozen tensors changed after optimizer step: {frozen_drift}")

    # REPROJ_LW=0 compatibility on the same checkpoint: toggle the config (the
    # model reads self.cfg) and confirm the legacy key set and no reproj key.
    cfg.MODEL.POSE_NET.LOSS_CFG.REPROJ_LW = 0.0
    legacy_kwargs = dict(inputs, **targets)
    legacy_kwargs["roi_zoom_cams"] = None
    legacy_kwargs["do_loss"] = True
    with EventStorage():
        _out, legacy = model(**legacy_kwargs)
    if "loss_xyz_reproj" in legacy:
        raise RuntimeError("REPROJ_LW=0 must not emit the reprojection key")

    print(
        json.dumps(
            {
                "status": "PASS",
                "arm": args.arm,
                "config": str(config_path),
                "device": args.device,
                "dtype": "float32",
                "base_reproj_lw": expected_lw,
                "trainable_geometry_head_tensors": trainable_names,
                "geo_grad_l1_scale": round(float(geo_grad_scale), 6),
                "losses": losses,
                "official_checkpoint_compat_missing_zero": True,
                "reproj_lw_zero_legacy_keys_match": set(legacy) == {
                    "loss_coor_x",
                    "loss_coor_y",
                    "loss_coor_z",
                    "loss_mask",
                    "loss_mask_full",
                    "loss_region",
                },
                "real_data": False,
                "formal_training": False,
                "contract": contract,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _snapshot(module_or_params):
    if isinstance(module_or_params, dict):
        named = module_or_params
    else:
        named = dict(module_or_params.named_parameters())
    return {name: parameter.detach().clone() for name, parameter in named.items()}


def _value_changed(before, module):
    return [
        name
        for name, parameter in module.named_parameters()
        if name in before and not torch.equal(parameter.detach(), before[name])
    ]


def _frozen_value_changed(before, after, trainable_names):
    return [
        name
        for name, value in after.items()
        if name not in trainable_names and not torch.equal(value, before[name])
    ]


if __name__ == "__main__":
    raise SystemExit(main())
