#!/usr/bin/env python3
"""Check dataset context, backbone initialization, and a CPU PCC step."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_PCC import build_model_optimizer
from research.exp013.preflight import PROJECT_ROOT, checkpoint_model_state
from research.run_contract import validate_research_run_config
from research.exp022.dataset_context import resolve_dataset_context


CONFIG_ROOT = PROJECT_ROOT / "configs/gdrn/lmo_pbr/research/exp022_progressive_pcc"
EXPERIMENT_ID = "EXP-20260916-022-progressive-pcc"


def load_official_backbone(model, weights: Path):
    state = checkpoint_model_state(weights)
    current = model.backbone.state_dict()
    source = {name.removeprefix("backbone."): tensor for name, tensor in state.items()
              if name.startswith("backbone.")}
    if current.keys() != source.keys():
        raise RuntimeError(f"Official backbone key mismatch: missing={current.keys()-source.keys()}, "
                           f"unexpected={source.keys()-current.keys()}")
    for name in current:
        if current[name].shape != source[name].shape:
            raise RuntimeError(f"Official backbone shape mismatch: {name}")
    model.backbone.load_state_dict(source, strict=True)
    return len(source)


def verify_imagenet_backbone(model, checkpoint: Path) -> int:
    """Check that all feature tensors match the supplied Facebook ConvNeXt file."""
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    current = model.backbone.state_dict()
    converted = {}
    for name, tensor in raw.items():
        if name.startswith(("head.", "norm.")):
            continue
        name = name.replace("downsample_layers.0.0.", "stem_0.")
        name = name.replace("downsample_layers.0.1.", "stem_1.")
        name = re.sub(r"downsample_layers\.(\d+)\.(\d+)",
                      lambda match: f"stages_{match.group(1)}.downsample.{match.group(2)}", name)
        name = re.sub(r"stages\.(\d+)\.(\d+)",
                      lambda match: f"stages_{match.group(1)}.blocks.{match.group(2)}", name)
        name = name.replace("dwconv", "conv_dw").replace("pwconv", "mlp.fc")
        if name not in current:
            raise RuntimeError(f"Unmapped ImageNet tensor: {name}")
        converted[name] = tensor.reshape(current[name].shape)
    if current.keys() != converted.keys():
        raise RuntimeError(f"ImageNet backbone tensor mismatch: {current.keys()-converted.keys()}")
    if any(not torch.equal(current[name].cpu(), converted[name]) for name in current):
        raise RuntimeError("ImageNet backbone weights were not loaded exactly")
    return len(converted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_ROOT / "train_reused.py")
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config_path = args.config if args.config.is_absolute() or args.config.exists() else CONFIG_ROOT / args.config
    cfg = Config.fromfile(str(config_path))
    mode = "smoke" if config_path.stem.startswith("smoke") else (
        "prepare" if cfg.get("RESEARCH_PROTOCOL", {}).get("SCHEDULE") == "configurable" else "formal"
    )
    validate_research_run_config(cfg, mode=mode, expected_experiment_id=cfg.EXPERIMENT_ID)
    context = resolve_dataset_context(cfg)
    cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    model, optimizer = build_model_optimizer(cfg)
    if cfg.MODEL.POSE_NET.BACKBONE.FREEZE:
        loaded = load_official_backbone(model, args.weights or Path(cfg.MODEL.WEIGHTS))
    else:
        loaded = verify_imagenet_backbone(
            model, Path(cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.checkpoint_path)
        )
    model.train()
    assert model.backbone.training == (not cfg.MODEL.POSE_NET.BACKBONE.FREEZE)
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    registered = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    allowed = ("pcc_head.",) if cfg.MODEL.POSE_NET.BACKBONE.FREEZE else ("pcc_head.", "backbone.")
    if not trainable or any(not name.startswith(allowed) for name in trainable):
        raise RuntimeError("Unexpected EXP022 trainable tensors")
    if not cfg.MODEL.POSE_NET.BACKBONE.FREEZE and not any(
        name.startswith("backbone.") for name in trainable
    ):
        raise RuntimeError("EXP022 full training has no trainable backbone")
    if registered != {id(parameter) for parameter in trainable.values()}:
        raise RuntimeError("EXP022 optimizer does not exactly match PCC tensors")
    image = torch.randn(1, 3, 256, 256)
    xyz = torch.full((1, 3, 64, 64), 0.5)
    mask = torch.ones(1, 1, 64, 64)
    _, losses = model(image, roi_classes=torch.tensor([0]), gt_xyz=xyz,
                      gt_mask_visib=mask, do_loss=True)
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise RuntimeError("EXP022 non-finite preflight loss")
    total.backward()
    if any(value.grad is None or not torch.isfinite(value.grad).all() for value in trainable.values()):
        raise RuntimeError("EXP022 missing/non-finite trainable gradient")
    optimizer.step()
    with torch.no_grad():
        output = model(image, roi_classes=torch.tensor([0]))
    if any(tuple(output[key].shape) != (1, 1, 64, 64) for key in
           ("coor_x", "coor_y", "coor_z", "mask")):
        raise RuntimeError("EXP022 inference output shape mismatch")
    print(json.dumps({"status": "PASS", "config": str(config_path), "formal_training": False,
                      "dataset": context.key, "num_objects": context.num_objects,
                      "object_ids": context.object_ids,
                      "hierarchy_path": str(context.hierarchy_path),
                      "backbone_checkpoint": str(args.weights or cfg.MODEL.WEIGHTS or
                                                 cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.get("checkpoint_path", "")),
                      "backbone_tensors": loaded,
                      "trainable_parameters": sum(p.numel() for p in trainable.values()),
                      "symmetry_counts": model.pcc_head.symmetry_counts.tolist(),
                      "max_symmetry_count": int(model.pcc_head.symmetry_counts.max().item()),
                      "losses": {name: float(value.detach()) for name, value in losses.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
