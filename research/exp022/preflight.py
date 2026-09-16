#!/usr/bin/env python3
"""Check official frozen-backbone loading and one CPU PCC optimization step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_PCC import build_model_optimizer
from research.exp013.preflight import PROJECT_ROOT, checkpoint_model_state
from research.run_contract import validate_research_run_config


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("train_reused.py", "smoke_reused.py", "smoke_independent.py"),
                        default="train_reused.py")
    parser.add_argument("--weights", type=Path,
                        default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = Config.fromfile(str(CONFIG_ROOT / args.config))
    validate_research_run_config(cfg, mode="formal" if args.config == "train_reused.py" else "smoke",
                                 expected_experiment_id=EXPERIMENT_ID)
    cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    model, optimizer = build_model_optimizer(cfg)
    loaded = load_official_backbone(model, args.weights)
    model.train()
    assert not model.backbone.training
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    registered = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if not trainable or any(not name.startswith("pcc_head.") for name in trainable):
        raise RuntimeError("Unexpected EXP022 trainable tensors")
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
    print(json.dumps({"status": "PASS", "config": args.config, "formal_training": False,
                      "official_backbone_tensors": loaded,
                      "trainable_parameters": sum(p.numel() for p in trainable.values()),
                      "losses": {name: float(value.detach()) for name, value in losses.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
