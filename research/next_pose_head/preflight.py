#!/usr/bin/env python3
"""Validate EXP012 integration, warm-start, isolation, and checkpoint reload."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from core.gdrn_modeling.models.heads.hierarchical_corr_pnp_net import (
    HierarchicalCorrespondencePnPNet,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/gdrn/lmo_pbr/research/exp012_hierarchical_corr_head/train.py"
)
DEFAULT_WEIGHTS = PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth"
EXPECTED_WEIGHT_SHA256 = (
    "bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_model_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    return {key.removeprefix("_module."): value for key, value in state.items()}


def validate_config(cfg: Config) -> None:
    pose = cfg.MODEL.POSE_NET
    pnp = pose.PNP_NET
    if not pose.BACKBONE.FREEZE or not pose.GEO_HEAD.FREEZE or pnp.FREEZE:
        raise RuntimeError("EXP012 must freeze backbone/geometry and train only PnP")
    if pose.BACKBONE.INIT_CFG.pretrained:
        raise RuntimeError("EXP012 must not download backbone weights")
    if pnp.INIT_CFG.type != "HierarchicalCorrespondencePnPNet":
        raise RuntimeError(f"Unexpected pose head: {pnp.INIT_CFG.type}")
    if not pnp.WITH_2D_COORD or pnp.COORD_2D_TYPE != "abs":
        raise RuntimeError("EXP012 requires absolute ROI 2D coordinates")
    if not pnp.REGION_ATTENTION or not pnp.INIT_CFG.use_region_aux:
        raise RuntimeError("EXP012 requires zero-start auxiliary Region input")
    if pnp.MASK_ATTENTION != "mul":
        raise RuntimeError("EXP012 requires multiplicative visible-mask support")
    if pose.QUALITY_COVERAGE.ENABLED:
        raise RuntimeError("EXP012 must not enable quality/coverage attention")


def clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def states_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left
    )


def load_official_shared_state(
    model: torch.nn.Module, official_state: dict[str, torch.Tensor]
) -> dict[str, int]:
    initial_pnp = clone_state(model.pnp_net)
    official_pnp_count = sum(key.startswith("pnp_net.") for key in official_state)
    incompatible = model.load_state_dict(dict(official_state), strict=False)
    expected_missing = {f"pnp_net.{key}" for key in model.pnp_net.state_dict()}
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            f"Official migration has unexpected missing keys: {incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Official migration has unexpected tensors: {incompatible.unexpected_keys}"
        )
    if not states_equal(initial_pnp, clone_state(model.pnp_net)):
        raise RuntimeError("Official checkpoint changed EXP012 head initialization")

    current = model.state_dict()
    shared = {
        key: value
        for key, value in official_state.items()
        if not key.startswith("pnp_net.")
    }
    missing_shared = sorted(set(shared) - set(current))
    mismatched_shared = [
        key
        for key, value in shared.items()
        if key in current and not torch.equal(current[key].detach().cpu(), value)
    ]
    if missing_shared or mismatched_shared:
        raise RuntimeError(
            "Official shared-state migration failed: "
            f"missing={missing_shared}, mismatched={mismatched_shared}"
        )
    return {
        "official_shared_tensors": len(shared),
        "legacy_pnp_tensors_filtered": official_pnp_count,
        "new_pose_head_tensors": len(expected_missing),
    }


def synthetic_full_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    axis = torch.linspace(0.0, 1.0, 64, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return {
        "x": torch.rand(1, 3, 256, 256, device=device),
        "roi_classes": torch.tensor([0], dtype=torch.long, device=device),
        "roi_cams": torch.tensor(
            [[[572.0, 0.0, 320.0], [0.0, 572.0, 240.0], [0.0, 0.0, 1.0]]],
            device=device,
        ),
        "roi_whs": torch.tensor([[160.0, 160.0]], device=device),
        "roi_centers": torch.tensor([[320.0, 240.0]], device=device),
        "resize_ratios": torch.tensor([1.0], device=device),
        "roi_coord_2d": torch.stack([xx, yy], dim=0).unsqueeze(0),
        "roi_extents": torch.tensor([[0.102, 0.102, 0.140]], device=device),
    }


def run_full_forward(model: torch.nn.Module, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        output = model(**synthetic_full_inputs(device))
    if output["rot"].shape != (1, 3, 3) or output["trans"].shape != (1, 3):
        raise RuntimeError("Full-model EXP012 forward returned unexpected pose shapes")
    if not all(torch.isfinite(output[key]).all() for key in ("rot", "trans")):
        raise RuntimeError("Full-model EXP012 forward produced non-finite pose")


def run_optimizer_step(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    head = model.pnp_net
    head.train()
    coor = torch.rand(2, 5, 64, 64, device=device)
    region = torch.softmax(torch.randn(2, 64, 64, 64, device=device), dim=1)
    mask = torch.rand(2, 1, 64, 64, device=device)
    extents = torch.rand(2, 3, device=device) + 0.1
    before = clone_state(head)
    optimizer.zero_grad(set_to_none=True)
    rotation, translation = head(coor, region, extents, mask)
    loss = rotation.square().mean() + translation.square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in head.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("EXP012 optimizer probe produced missing/non-finite gradients")
    optimizer.step()
    if states_equal(before, clone_state(head)):
        raise RuntimeError("EXP012 optimizer probe changed no pose-head tensor")


def validate_round_trip(cfg: Config, model: torch.nn.Module) -> None:
    coor = torch.rand(2, 5, 16, 16)
    region = torch.softmax(torch.randn(2, 64, 16, 16), dim=1)
    mask = torch.rand(2, 1, 16, 16)
    extents = torch.rand(2, 3) + 0.1
    model.cpu().pnp_net.eval()
    with torch.no_grad():
        reference = model.pnp_net(coor, region, extents, mask)

    with tempfile.TemporaryDirectory(prefix="exp012_roundtrip_") as directory:
        path = Path(directory) / "model.pth"
        torch.save({"model": model.state_dict()}, path)
        reloaded, optimizer = build_model_optimizer(cfg, is_test=True)
        if optimizer is not None:
            raise RuntimeError("Test-mode EXP012 build unexpectedly created optimizer")
        state = torch.load(path, map_location="cpu")["model"]
        incompatible = reloaded.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"EXP012 strict reload was incomplete: {incompatible}")
        reloaded.cpu().pnp_net.eval()
        with torch.no_grad():
            actual = reloaded.pnp_net(coor, region, extents, mask)
    if not all(torch.equal(left, right) for left, right in zip(reference, actual)):
        raise RuntimeError("EXP012 checkpoint roundtrip changed pose-head output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-round-trip", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    weights_path = args.weights.resolve()
    if sha256(weights_path) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Unexpected official checkpoint hash")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but CUDA is unavailable")

    cfg = Config.fromfile(str(config_path))
    validate_config(cfg)
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    cfg.MODEL.DEVICE = args.device
    device = torch.device(args.device)
    model, optimizer = build_model_optimizer(cfg, is_test=False)
    if optimizer is None:
        raise RuntimeError("Training-mode EXP012 build did not create optimizer")
    if not isinstance(model.pnp_net, HierarchicalCorrespondencePnPNet):
        raise RuntimeError(f"Unexpected built pose head: {type(model.pnp_net)}")

    migration = load_official_shared_state(model, checkpoint_model_state(weights_path))
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or not all(name.startswith("pnp_net.") for name in trainable):
        raise RuntimeError(f"Unexpected trainable tensors: {trainable}")
    run_full_forward(model, device)
    run_optimizer_step(model, optimizer, device)
    if not args.skip_round_trip:
        if args.device != "cpu":
            raise RuntimeError("Strict roundtrip is defined on CPU; use --skip-round-trip")
        validate_round_trip(cfg, model)

    print(
        json.dumps(
            {
                "status": "PASS",
                "config": str(config_path),
                "device": args.device,
                "dtype": "float32",
                "official_weight_sha256": EXPECTED_WEIGHT_SHA256,
                "trainable_parameters": sum(
                    value.numel() for value in model.parameters() if value.requires_grad
                ),
                "trainable_tensors": len(trainable),
                "full_forward": True,
                "optimizer_step": True,
                "checkpoint_round_trip": not args.skip_round_trip,
                **migration,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
