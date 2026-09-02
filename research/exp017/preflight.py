#!/usr/bin/env python3
"""Validate EXP017 before any real-data smoke or formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from core.gdrn_modeling.models.heads.exp013_geometry_pnp_net import (
    XYZResidualBypassPnPNet,
)
from core.gdrn_modeling.models.heads.exp017_rotation_residual_pnp_net import (
    SupportAwareRotationResidualPnPNet,
)
from core.gdrn_modeling.models.model_utils import get_pnp_net
from research.exp013.preflight import (
    PROJECT_ROOT,
    checkpoint_model_state,
    load_official_shared_state,
    profile_head,
    synthetic_full_inputs,
)


CONFIG = (
    "configs/gdrn/lmo_pbr/research/exp017/"
    "support_aware_rotation_residual/train.py"
)
A_CONFIG = "configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual/train.py"
EXPERIMENT_ID = "EXP-20260902-017-support-aware-rotation-residual"
EXPECTED_ADAPTER_PARAMETERS = 13_000
MAX_ADAPTER_PARAMETERS = 15_000


def _config_dict(value):
    return value.to_dict() if hasattr(value, "to_dict") else dict(value)


def validate_config(cfg: Config, a_cfg: Config) -> None:
    if cfg.EXPERIMENT_ID != EXPERIMENT_ID:
        raise RuntimeError(f"Unexpected EXP017 ID: {cfg.EXPERIMENT_ID!r}")
    if cfg.SEED != 42:
        raise RuntimeError("EXP017 requires seed 42")
    for name in ("DATASETS", "DATALOADER", "SOLVER", "TEST", "TRAIN"):
        if _config_dict(cfg[name]) != _config_dict(a_cfg[name]):
            raise RuntimeError(f"EXP017 {name} must remain matched to EXP013A")

    pose, a_pose = cfg.MODEL.POSE_NET, a_cfg.MODEL.POSE_NET
    if cfg.MODEL.WEIGHTS != a_cfg.MODEL.WEIGHTS:
        raise RuntimeError("EXP017 must use EXP013A's official checkpoint protocol")
    for name in ("BACKBONE", "GEO_HEAD", "QUALITY_COVERAGE", "LOSS_CFG"):
        if _config_dict(pose[name]) != _config_dict(a_pose[name]):
            raise RuntimeError(f"EXP017 MODEL.POSE_NET.{name} differs from EXP013A")
    if not pose.BACKBONE.FREEZE or not pose.GEO_HEAD.FREEZE or pose.PNP_NET.FREEZE:
        raise RuntimeError("EXP017 must freeze backbone/geometry and train only pnp_net")

    pnp, a_pnp = pose.PNP_NET, a_pose.PNP_NET
    for key in pnp:
        if key == "INIT_CFG":
            continue
        if pnp[key] != a_pnp[key]:
            raise RuntimeError(f"EXP017 PNP_NET.{key} differs from EXP013A")
    init_cfg = _config_dict(pnp.INIT_CFG)
    a_init_cfg = _config_dict(a_pnp.INIT_CFG)
    if init_cfg.pop("type") != SupportAwareRotationResidualPnPNet.__name__:
        raise RuntimeError("EXP017 head type is incorrect")
    adapter_cfg = {
        "adapter_token_channels": init_cfg.pop("adapter_token_channels"),
        "adapter_score_channels": init_cfg.pop("adapter_score_channels"),
        "alpha_r_init": init_cfg.pop("alpha_r_init"),
    }
    a_init_cfg.pop("type")
    if init_cfg != a_init_cfg:
        raise RuntimeError("EXP017 changed the inherited EXP013A head structure")
    if adapter_cfg != {
        "adapter_token_channels": 64,
        "adapter_score_channels": 32,
        "alpha_r_init": 1.0,
    }:
        raise RuntimeError(f"Unexpected adapter config: {adapter_cfg}")
    if cfg.SOLVER.TOTAL_EPOCHS != 40 or cfg.SOLVER.IMS_PER_BATCH != 48:
        raise RuntimeError("EXP017 formal protocol requires 40 epochs and batch 48")
    if cfg.SOLVER.OPTIMIZER_CFG.type != "Ranger":
        raise RuntimeError("EXP017 requires Ranger")
    if (
        cfg.SOLVER.OPTIMIZER_CFG.lr != 8e-4
        or cfg.SOLVER.OPTIMIZER_CFG.weight_decay != 0.01
        or cfg.SOLVER.WARMUP_ITERS != 200
    ):
        raise RuntimeError("EXP017 solver must remain matched to EXP013A")


def _head_inputs(device: torch.device, batch: int = 2):
    torch.manual_seed(1702)
    coor = torch.rand(batch, 5, 64, 64, device=device)
    region = torch.softmax(
        torch.randn(batch, 64, 64, 64, device=device), dim=1
    )
    extents = torch.rand(batch, 3, device=device) + 0.05
    support = torch.ones(batch, 1, 64, 64, device=device)
    support[:, :, :16, :24] = 0
    return coor, region, extents, support


def _load_a_state_into_exp017(
    a_head: XYZResidualBypassPnPNet,
    exp017_head: SupportAwareRotationResidualPnPNet,
) -> None:
    incompatible = exp017_head.load_state_dict(a_head.state_dict(), strict=False)
    expected_missing = {
        f"rotation_adapter.{name}"
        for name in exp017_head.rotation_adapter.state_dict()
    } | {"alpha_r"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "A-to-EXP017 migration mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def head_contract_checks(
    cfg: Config, a_cfg: Config, device: torch.device
) -> dict[str, object]:
    exp017_head, _ = get_pnp_net(cfg)
    a_head, _ = get_pnp_net(a_cfg)
    exp017_head.to(device).eval()
    a_head.to(device).eval()
    _load_a_state_into_exp017(a_head, exp017_head)

    a_parameters = sum(parameter.numel() for parameter in a_head.parameters())
    total_parameters = sum(parameter.numel() for parameter in exp017_head.parameters())
    adapter_parameters = exp017_head.adapter_parameter_count()
    if adapter_parameters != EXPECTED_ADAPTER_PARAMETERS:
        raise RuntimeError(
            f"Expected {EXPECTED_ADAPTER_PARAMETERS} adapter parameters, got {adapter_parameters}"
        )
    if adapter_parameters > MAX_ADAPTER_PARAMETERS:
        raise RuntimeError("EXP017 adapter exceeds the 15k hard gate")
    if total_parameters - a_parameters != adapter_parameters:
        raise RuntimeError("EXP017 changed parameters outside the adapter")

    inputs = _head_inputs(device)
    with torch.no_grad():
        expected_r, expected_t = a_head(*inputs)
        actual_r, actual_t, info = exp017_head.forward_with_adapter_intervention(
            *inputs
        )
    if torch.count_nonzero(info["delta_r"]) != 0:
        raise RuntimeError("EXP017 delta_r is not exactly zero at initialization")
    if not torch.equal(actual_r, expected_r):
        raise RuntimeError("EXP017 initial raw rotation is not value-exact to A")
    if not torch.equal(actual_t, expected_t):
        raise RuntimeError("EXP017 initial raw translation is not bitwise equal to A")
    if not torch.equal(exp017_head.alpha_r, torch.ones_like(exp017_head.alpha_r)):
        raise RuntimeError("EXP017 alpha_r must initialize to exactly 1")

    exp017_head.zero_grad(set_to_none=True)
    _r, raw_t = exp017_head(*inputs)
    raw_t.square().mean().backward()
    if any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) != 0
        for parameter in exp017_head.adapter_parameters()
    ):
        raise RuntimeError("Translation-only loss reached EXP017 adapter parameters")

    exp017_head.zero_grad(set_to_none=True)
    raw_r, _t = exp017_head(*inputs)
    raw_r.sum().backward()
    output_gradient = exp017_head.rotation_adapter.delta_output.weight.grad
    if (
        output_gradient is None
        or not bool(torch.isfinite(output_gradient).all())
        or torch.count_nonzero(output_gradient) == 0
    ):
        raise RuntimeError("Rotation loss did not reach delta_r output layer")

    optimizer = torch.optim.SGD(exp017_head.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    raw_r, _t = exp017_head(*inputs)
    raw_r.square().mean().backward()
    internal = {
        "token_projection": exp017_head.rotation_adapter.token_projection.weight,
        "pool_score_in": exp017_head.rotation_adapter.pool_score[0].weight,
        "pool_score_out": exp017_head.rotation_adapter.pool_score[2].weight,
        "position_embedding": exp017_head.rotation_adapter.position_embedding,
        "delta_hidden": exp017_head.rotation_adapter.delta_hidden.weight,
    }
    for name, parameter in internal.items():
        if (
            parameter.grad is None
            or not bool(torch.isfinite(parameter.grad).all())
            or torch.count_nonzero(parameter.grad) == 0
        ):
            raise RuntimeError(f"Adapter internal parameter lacks post-step gradient: {name}")

    weights, valid = info["weights"], info["valid"]
    if torch.count_nonzero(weights[~valid]) != 0:
        raise RuntimeError("Invalid EXP017 token received non-zero pooling weight")
    if not torch.allclose(
        weights.sum(dim=1), torch.ones(weights.shape[0], device=device), atol=1e-7, rtol=0
    ):
        raise RuntimeError("EXP017 pooling weights do not sum to one")

    return {
        "a_head_parameters": a_parameters,
        "exp017_head_parameters": total_parameters,
        "adapter_parameters": adapter_parameters,
        "adapter_parameter_hard_gate": MAX_ADAPTER_PARAMETERS,
        "alpha_r_initial": 1.0,
        "delta_r_initial_exact_zero": True,
        "raw_rotation_initial_value_exact_to_a": True,
        "raw_translation_initial_bitwise_equal_to_a": True,
        "translation_only_adapter_gradient_zero": True,
        "rotation_output_gradient_nonzero": True,
        "post_step_internal_gradients_nonzero": True,
        "invalid_pool_weight_zero": True,
        "valid_pool_weight_sum_one": True,
    }


def full_model_optimizer_probe(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, object]:
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any(not name.startswith("pnp_net.") for name in trainable):
        raise RuntimeError(f"Unexpected trainable tensors: {trainable}")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("Backbone is not frozen")
    if any(parameter.requires_grad for parameter in model.geo_head_net.parameters()):
        raise RuntimeError("Geometry head is not frozen")

    model.eval()
    optimizer.zero_grad(set_to_none=True)
    captured = {}

    def hook(_module, _inputs, output):
        captured["raw"] = output

    handle = model.pnp_net.register_forward_hook(hook)
    try:
        output = model(**synthetic_full_inputs(device, batch=1))
    finally:
        handle.remove()
    if output["rot"].shape != (1, 3, 3) or output["trans"].shape != (1, 3):
        raise RuntimeError("Full-model pose shapes are incorrect")
    raw_r, raw_t = captured["raw"]
    loss = raw_r.square().mean() + raw_t.square().mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Full-model preflight loss is non-finite")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.pnp_net.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(bool(torch.isfinite(grad).all()) for grad in gradients):
        raise RuntimeError("Full-model preflight gradients are invalid")
    output_gradient = model.pnp_net.rotation_adapter.delta_output.weight.grad
    if output_gradient is None or torch.count_nonzero(output_gradient) == 0:
        raise RuntimeError("Full model did not train the rotation residual output")
    optimizer.step()
    return {
        "only_pnp_net_trainable": True,
        "backbone_frozen": True,
        "geometry_head_frozen": True,
        "full_model_forward_backward_optimizer_step": True,
        "full_model_loss": float(loss.detach().cpu()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (args.config or PROJECT_ROOT / CONFIG).resolve()
    a_config_path = (PROJECT_ROOT / A_CONFIG).resolve()
    weights_path = args.weights.resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cfg = Config.fromfile(str(config_path))
    a_cfg = Config.fromfile(str(a_config_path))
    validate_config(cfg, a_cfg)
    for current in (cfg, a_cfg):
        current.SOLVER.BASE_LR = float(current.SOLVER.OPTIMIZER_CFG.lr)
    cfg.MODEL.DEVICE = args.device
    a_cfg.MODEL.DEVICE = args.device
    device = torch.device(args.device)

    contracts = head_contract_checks(cfg, a_cfg, device)
    model, optimizer = build_model_optimizer(cfg, is_test=False)
    if optimizer is None or not isinstance(
        model.pnp_net, SupportAwareRotationResidualPnPNet
    ):
        raise RuntimeError("Training build returned the wrong model or no optimizer")
    load_official_shared_state(model, checkpoint_model_state(weights_path))
    model.to(device)
    full_model = full_model_optimizer_probe(model, optimizer, device)
    profile = profile_head(model.pnp_net, device)
    print(
        json.dumps(
            {
                "status": "PASS",
                "experiment_id": EXPERIMENT_ID,
                "config": str(config_path),
                "weights": str(weights_path),
                "device": args.device,
                **contracts,
                **full_model,
                **profile,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
