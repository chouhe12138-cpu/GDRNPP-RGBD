#!/usr/bin/env python3
"""Validate CPM integration, official migration, freezing, and FP32 forward."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from core.gdrn_modeling.models.heads.cpm_pnp_net import (
    CorrespondenceAwareMomentPnPNet,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/gdrn/lmo_pbr/convnext_cpm_head_local_lmo.py"
)
DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth"
)
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
        raise RuntimeError("CPM must freeze backbone/geometry and train only its pose head")
    if pose.BACKBONE.INIT_CFG.pretrained:
        raise RuntimeError("CPM integration config must not download backbone weights")
    if pnp.INIT_CFG.type != "CorrespondenceAwareMomentPnPNet":
        raise RuntimeError(f"Unexpected pose head: {pnp.INIT_CFG.type}")
    if not pnp.WITH_2D_COORD or pnp.COORD_2D_TYPE != "abs":
        raise RuntimeError("CPM requires absolute ROI 2D coordinates")
    if not pnp.REGION_ATTENTION or pnp.MASK_ATTENTION != "mul":
        raise RuntimeError("CPM requires Region posterior and visible-mask support")
    if pose.QUALITY_COVERAGE.ENABLED:
        raise RuntimeError("CPM integration must not enable quality/coverage attention")


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
    incompatible = model.load_state_dict(official_state, strict=False)
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
        raise RuntimeError("Official checkpoint changed CPM initialization")

    current = model.state_dict()
    shared = {key: value for key, value in official_state.items() if not key.startswith("pnp_net.")}
    missing_shared = sorted(set(shared) - set(current))
    if missing_shared:
        raise RuntimeError(f"Official shared tensors are absent from CPM model: {missing_shared}")
    mismatched_shared = [
        key
        for key, value in shared.items()
        if not torch.equal(current[key].detach().cpu(), value.detach().cpu())
    ]
    if mismatched_shared:
        raise RuntimeError(f"Official shared tensors differ after load: {mismatched_shared}")
    return {
        "official_shared_tensors": len(shared),
        "legacy_pnp_tensors_filtered": sum(
            key.startswith("pnp_net.") for key in official_state
        ),
        "new_cpm_tensors": len(expected_missing),
    }


def synthetic_inputs(device: torch.device) -> dict[str, torch.Tensor]:
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
    inputs = synthetic_inputs(device)
    model.eval()
    with torch.no_grad():
        output = model(**inputs)
    if output["rot"].shape != (1, 3, 3) or output["trans"].shape != (1, 3):
        raise RuntimeError(
            f"Unexpected full-model output shapes: {output['rot'].shape}/{output['trans'].shape}"
        )
    if not bool(torch.isfinite(output["rot"]).all()) or not bool(
        torch.isfinite(output["trans"]).all()
    ):
        raise RuntimeError("Full-model CPM forward produced non-finite pose")


def validate_round_trip(cfg: Config, model: torch.nn.Module) -> None:
    coor = torch.rand(2, 5, 16, 16)
    region = torch.softmax(torch.randn(2, 64, 16, 16), dim=1)
    mask = torch.rand(2, 1, 16, 16)
    extents = torch.rand(2, 3) + 0.1
    model.pnp_net.eval()
    with torch.no_grad():
        reference = model.pnp_net(coor, region, extents, mask)

    with tempfile.TemporaryDirectory(prefix="cpm_roundtrip_") as directory:
        path = Path(directory) / "model.pth"
        torch.save({"model": model.state_dict()}, path)
        reloaded, optimizer = build_model_optimizer(cfg, is_test=True)
        if optimizer is not None:
            raise RuntimeError("Test-mode CPM build unexpectedly created an optimizer")
        state = torch.load(path, map_location="cpu")["model"]
        reloaded.load_state_dict(state, strict=True)
        reloaded.pnp_net.eval()
        with torch.no_grad():
            changed = reloaded.pnp_net(coor, region, extents, mask)
    if not all(torch.equal(a, b) for a, b in zip(reference, changed)):
        raise RuntimeError("CPM checkpoint save/reload changed pose-head output")


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
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    weight_hash = sha256(weights_path)
    if weight_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"Unexpected official checkpoint hash: {weight_hash}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")

    cfg = Config.fromfile(str(config_path))
    validate_config(cfg)
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    cfg.MODEL.DEVICE = args.device
    device = torch.device(args.device)
    model, optimizer = build_model_optimizer(cfg, is_test=True)
    if optimizer is not None:
        raise RuntimeError("Test-mode CPM build unexpectedly created an optimizer")
    if not isinstance(model.pnp_net, CorrespondenceAwareMomentPnPNet):
        raise RuntimeError(f"Unexpected built pose head: {type(model.pnp_net)}")

    official_state = checkpoint_model_state(weights_path)
    migration = load_official_shared_state(model, official_state)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or not all(name.startswith("pnp_net.") for name in trainable):
        raise RuntimeError(f"Unexpected trainable tensors: {trainable}")
    run_full_forward(model, device)
    if not args.skip_round_trip:
        if args.device != "cpu":
            raise RuntimeError("Full checkpoint round-trip is defined on CPU only")
        validate_round_trip(cfg, model)

    summary = {
        "status": "PASS",
        "config": str(config_path),
        "device": args.device,
        "dtype": "float32",
        "official_weight_sha256": weight_hash,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "trainable_tensors": len(trainable),
        "full_forward": True,
        "checkpoint_round_trip": not args.skip_round_trip,
        **migration,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
