#!/usr/bin/env python3
"""CPU synthetic preflight for EXP021 arms B/C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from detectron2.utils.events import EventStorage
from mmcv import Config

from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from research.exp013.preflight import PROJECT_ROOT, checkpoint_model_state, synthetic_full_inputs
from research.exp020.preflight import synthetic_geometry_targets
from research.run_contract import validate_research_run_config


EXPERIMENT_ID = "EXP-20260914-021-global-guided-hierarchical-cad-correspondence"
CONFIG_ROOT = PROJECT_ROOT / "configs/gdrn/lmo_pbr/research/exp021_global_hierarchical_cad"


def _build(arm: str, device: torch.device):
    torch.manual_seed(42)
    cfg = Config.fromfile(str(CONFIG_ROOT / ("b_hierarchical.py" if arm == "B" else "c_global.py")))
    validate_research_run_config(cfg, mode="formal", expected_experiment_id=EXPERIMENT_ID)
    cfg.MODEL.DEVICE = str(device)
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    return cfg, *build_model_optimizer(cfg)


def _load_official(model, checkpoint: Path):
    state = checkpoint_model_state(checkpoint)
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys or any(
        not name.startswith("cad_head.") for name in incompatible.missing_keys
    ):
        raise RuntimeError(f"Official checkpoint incompatibility: {incompatible}")
    for name, value in model.state_dict().items():
        if name.startswith(("backbone.", "geo_head_net.", "pnp_net.")):
            if not torch.equal(value.cpu(), state[name].cpu()):
                raise RuntimeError(f"Inherited tensor changed while loading official checkpoint: {name}")
    return incompatible.missing_keys


def _optimizer_contract(model, optimizer):
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    registered = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if not trainable or any(not name.startswith("cad_head.") for name in trainable):
        raise RuntimeError(f"Unexpected trainable tensors: {sorted(trainable)}")
    if {id(value) for value in trainable.values()} != registered:
        raise RuntimeError("Optimizer parameters do not exactly match trainable CAD parameters")
    return trainable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("B", "C"), required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    )
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    cfg, model, optimizer = _build(args.arm, device)
    missing = _load_official(model, args.weights)
    trainable = _optimizer_contract(model, optimizer)
    inputs = synthetic_full_inputs(device, 1)
    inputs.update(synthetic_geometry_targets(inputs, cfg.MODEL.POSE_NET.OUTPUT_RES, 1, device))
    before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    with EventStorage():
        output, losses = model(**inputs, do_loss=True, cad_beam_ks=(4,))
    if set(losses) != {"loss_cad_coarse", "loss_cad_fine", "loss_cad_xyz"}:
        raise RuntimeError(f"Unexpected EXP021 losses: {sorted(losses)}")
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise RuntimeError("Non-finite EXP021 loss")
    total.backward()
    missing_grad = [name for name, value in trainable.items() if value.grad is None]
    if missing_grad:
        raise RuntimeError(f"Trainable tensors without gradients: {missing_grad}")
    optimizer.step()
    changed_frozen = [
        name
        for name, value in model.named_parameters()
        if name in before and not torch.equal(before[name], value.detach())
    ]
    if changed_frozen:
        raise RuntimeError(f"Frozen tensors changed: {changed_frozen[:5]}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "arm": args.arm,
                "device": str(device),
                "formal_training": False,
                "trainable_parameters": sum(value.numel() for value in trainable.values()),
                "official_missing_keys": len(missing),
                "only_cad_head_trainable": True,
                "frozen_tensors_unchanged": True,
                "losses": {name: float(value.detach()) for name, value in losses.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
