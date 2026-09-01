#!/usr/bin/env python3
"""Build and validate EXP014-D (ImageNet end-to-end unfrozen) configuration.

Kept deliberately small: reuses EXP013 preflight helpers and only adds the
checks that matter for the full-unfreeze regime (nothing frozen, no official
checkpoint, backbone pretrained from timm).

Rollback: deleting this file removes all EXP014-D gate machinery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from core.gdrn_modeling.models.heads.exp013_geometry_pnp_net import (
    RTDecoupledGeometryPnPNet,
)
from research.exp013.preflight import (
    PROJECT_ROOT,
    profile_head,
    synthetic_full_inputs,
)

CONFIG = "configs/gdrn/lmo_pbr/research/exp013/d_fulltrain/train.py"
EXPECTED_HEAD = RTDecoupledGeometryPnPNet
PRETRAINED_FILE = "convnext_base_1k_224_ema.pth"


def validate_config(cfg: Config) -> None:
    pose = cfg.MODEL.POSE_NET
    pnp = pose.PNP_NET
    if cfg.MODEL.WEIGHTS != "":
        raise RuntimeError(f"EXP014-D requires MODEL.WEIGHTS=''; got {cfg.MODEL.WEIGHTS!r}")
    if pose.BACKBONE.FREEZE or not pose.BACKBONE.INIT_CFG.pretrained:
        raise RuntimeError("EXP014-D requires an unfrozen ImageNet-pretrained backbone")
    if pose.BACKBONE.get("PRETRAINED", "") != "timm":
        raise RuntimeError("EXP014-D requires BACKBONE.PRETRAINED='timm'")
    if pose.GEO_HEAD.FREEZE or not pose.GEO_HEAD.get("TRAIN_SUPERVISION", True):
        raise RuntimeError("EXP014-D requires unfrozen geometry head with supervision")
    if pnp.FREEZE:
        raise RuntimeError("EXP014-D requires an unfrozen pnp_net")
    if pnp.INIT_CFG.type != EXPECTED_HEAD.__name__:
        raise RuntimeError(f"Unexpected pose head: {pnp.INIT_CFG.type}")
    if not pnp.WITH_2D_COORD or pnp.COORD_2D_TYPE != "abs":
        raise RuntimeError("EXP014-D requires absolute ROI2D")
    if not pnp.REGION_ATTENTION or not pnp.INIT_CFG.use_region_aux:
        raise RuntimeError("EXP014-D keeps the EXP012 Region main path")
    if pnp.MASK_ATTENTION != "mul":
        raise RuntimeError("EXP014-D requires visible-mask multiplication")
    if cfg.SOLVER.TOTAL_EPOCHS != 40 or cfg.SOLVER.IMS_PER_BATCH != 48:
        raise RuntimeError(
            "EXP014-D formal protocol requires 40 epochs and batch size 48"
        )
    if cfg.SOLVER.CHECKPOINT_PERIOD != 5 or cfg.TEST.EVAL_PERIOD != 5:
        raise RuntimeError("EXP014-D requires checkpoint/evaluation every five epochs")
    if cfg.SEED != 42:
        raise RuntimeError("EXP014-D requires seed 42")
    if cfg.SOLVER.WARMUP_ITERS < 1000:
        raise RuntimeError("EXP014-D requires WARMUP_ITERS >= 1000")
    # The engine only reads MODEL.POSE_NET.XYZ_RENDERER (engine_utils.get_renderer);
    # a top-level XYZ_RENDERER key is inert, so validate the nested value and
    # enforce the preregistered egl renderer for the full-training formal run.
    if pose.get("XYZ_RENDERER", "") != "egl":
        raise RuntimeError(
            "EXP014-D preregisters MODEL.POSE_NET.XYZ_RENDERER='egl'; "
            f"got {pose.get('XYZ_RENDERER', '')!r}"
        )


def bucket_parameters(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    buckets: dict[str, list[torch.nn.Parameter]] = {
        "backbone": [],
        "pnp_net": [],
        "other_head": [],  # geometry head and any remaining trainable tensors
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            buckets["backbone"].append(parameter)
        elif name.startswith("pnp_net."):
            buckets["pnp_net"].append(parameter)
        else:
            buckets["other_head"].append(parameter)
    return buckets


def full_forward_backward_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, int]:
    buckets = bucket_parameters(model)
    empty = [name for name, params in buckets.items() if not params]
    if empty:
        raise RuntimeError(f"No trainable parameters in buckets: {empty}")
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    raw_pose: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def capture_raw_pose(_module, _inputs, output):
        raw_pose["value"] = output

    handle = model.pnp_net.register_forward_hook(capture_raw_pose)
    try:
        output = model(**synthetic_full_inputs(device))
    finally:
        handle.remove()
    if output["rot"].shape != (1, 3, 3) or output["trans"].shape != (1, 3):
        raise RuntimeError("Full-model forward returned unexpected pose shapes")
    if "value" not in raw_pose:
        raise RuntimeError("Full-model forward did not execute pnp_net")
    raw_rotation, raw_translation = raw_pose["value"]
    loss = raw_rotation.square().mean() + raw_translation.square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Full-model loss is non-finite")
    loss.backward()
    for name, parameters in buckets.items():
        gradients = [value.grad for value in parameters if value.grad is not None]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError(f"Invalid optimizer gradients in {name}")
    if isinstance(model.pnp_net, RTDecoupledGeometryPnPNet):
        for scale_name in ("geometry_scale_r", "geometry_scale_t"):
            gradient = getattr(model.pnp_net, scale_name).grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"Missing or non-finite D gradient: {scale_name}")
    optimizer.step()
    return {name: len(params) for name, params in buckets.items()}


def locate_pretrained_file() -> dict[str, str | int] | None:
    """Best-effort report of the timm ImageNet weight file location."""
    hub_checkpoints = Path(torch.hub.get_dir()) / "checkpoints" / PRETRAINED_FILE
    if hub_checkpoints.is_file():
        return {"path": str(hub_checkpoints), "bytes": hub_checkpoints.stat().st_size}
    hf_root = Path.home() / ".cache/huggingface/hub"
    if hf_root.is_dir():
        for match in hf_root.glob("models--*convnext*"):
            if not match.is_dir():
                continue
            for blob in sorted(
                match.glob("snapshots/*/*"),
                key=lambda p: p.stat().st_size,
                reverse=True,
            ):
                if blob.is_file() and blob.stat().st_size > 100 * 1024 * 1024:
                    return {"path": str(blob), "bytes": blob.stat().st_size}
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (args.config or PROJECT_ROOT / CONFIG).resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cfg = Config.fromfile(str(config_path))
    validate_config(cfg)
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    cfg.MODEL.DEVICE = args.device
    model, optimizer = build_model_optimizer(cfg, is_test=False)
    if optimizer is None or not isinstance(model.pnp_net, EXPECTED_HEAD):
        raise RuntimeError("Training build returned the wrong model or no optimizer")
    device = torch.device(args.device)
    model.to(device)
    transport = full_forward_backward_step(model, optimizer, device)
    profile = profile_head(model.pnp_net, device)
    tracker = {
        "status": "PASS",
        "config": str(config_path),
        "device": args.device,
        "dtype": "float32",
        "pretrained_backbone_file": locate_pretrained_file(),
        "model_total_parameters": sum(p.numel() for p in model.parameters()),
        "model_trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "trainable_buckets": transport,
        "full_model_forward_backward_optimizer_step": True,
        # profile_head (reused from EXP013) reports pnp_net-only metrics and
        # itself carries a "trainable_parameters" key; it is expanded last and
        # therefore intentionally wins for that name.
        **profile,
    }
    print(json.dumps(tracker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
