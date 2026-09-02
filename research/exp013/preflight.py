#!/usr/bin/env python3
"""Build and validate one EXP013 variant against the official checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from core.gdrn_modeling.models.heads.exp013_geometry_pnp_net import (
    GeometryAttentionResidualPnPNet,
    RTDecoupledGeometryPnPNet,
    XYZResidualBypassPnPNet,
)
from core.gdrn_modeling.models.heads.glm_pose_net import GLMPoseLNet
from core.gdrn_modeling.models.heads.official_head_random_init import (
    OfficialConvPnPNetRandomInit,
)
from research.run_contract import validate_research_run_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "A": (
        "configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual/train.py",
        XYZResidualBypassPnPNet,
    ),
    "B": (
        "configs/gdrn/lmo_pbr/research/exp013/b_geometry_attention/train.py",
        GeometryAttentionResidualPnPNet,
    ),
    "C": (
        "configs/gdrn/lmo_pbr/research/exp013/c_rt_decoupled/train.py",
        RTDecoupledGeometryPnPNet,
    ),
    "E": (
        "configs/gdrn/lmo_pbr/research/exp013/e_official_head_random/train.py",
        OfficialConvPnPNetRandomInit,
    ),
    "F": (
        "configs/gdrn/lmo_pbr/research/exp013/f_glm_pose_l/train.py",
        GLMPoseLNet,
    ),
}


def checkpoint_model_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    return {key.removeprefix("_module."): value for key, value in state.items()}


def validate_config(cfg: Config, expected_type: type[torch.nn.Module]) -> None:
    validate_research_run_config(cfg, mode="formal")
    pose = cfg.MODEL.POSE_NET
    pnp = pose.PNP_NET
    if not pose.BACKBONE.FREEZE or not pose.GEO_HEAD.FREEZE or pnp.FREEZE:
        raise RuntimeError(
            "EXP013 must freeze backbone/geometry and train only pnp_net"
        )
    if pose.BACKBONE.INIT_CFG.pretrained:
        raise RuntimeError("EXP013 must not download backbone weights")
    if pnp.INIT_CFG.type != expected_type.__name__:
        raise RuntimeError(f"Unexpected pose head: {pnp.INIT_CFG.type}")
    if not pnp.WITH_2D_COORD or pnp.COORD_2D_TYPE != "abs":
        raise RuntimeError("EXP013 requires absolute ROI2D")
    if expected_type is OfficialConvPnPNetRandomInit:
        # EXP013E rebuilds the official head with its native flags; it must
        # not inherit any EXP013-family head settings.
        if pnp.INIT_CFG.get("act") != "gelu" or pnp.INIT_CFG.get("norm") != "GN":
            raise RuntimeError("EXP013E must mirror the official head norm/act")
        if pnp.INIT_CFG.get("flat_op") != "flatten" or not pnp.INIT_CFG.get(
            "denormalize_by_extent"
        ):
            raise RuntimeError("EXP013E must keep the official flatten-fc1 design")
        if "use_region_aux" in pnp.INIT_CFG or "geometry_scale_init" in pnp.INIT_CFG:
            raise RuntimeError("EXP013E must not inherit EXP013 head settings")
        if pnp.MASK_ATTENTION != "none":
            raise RuntimeError("EXP013E keeps the official no-mask-gating design")
        if not pnp.REGION_ATTENTION:
            raise RuntimeError("EXP013E keeps the official region attention")
    else:
        if not pnp.REGION_ATTENTION or not pnp.INIT_CFG.use_region_aux:
            raise RuntimeError("EXP013 keeps the EXP012 Region main path")
        if pnp.MASK_ATTENTION != "mul":
            raise RuntimeError("EXP013 requires visible-mask multiplication")
    if cfg.SOLVER.TOTAL_EPOCHS != 40 or cfg.SOLVER.IMS_PER_BATCH != 48:
        raise RuntimeError(
            "EXP013 formal protocol requires 40 epochs and batch size 48"
        )
    if cfg.SOLVER.CHECKPOINT_PERIOD != 5 or cfg.TEST.EVAL_PERIOD != 5:
        raise RuntimeError("EXP013 requires checkpoint/evaluation every five epochs")
    if cfg.SEED != 42:
        raise RuntimeError("EXP013 requires seed 42")
    if expected_type is RTDecoupledGeometryPnPNet:
        if pose.GEO_HEAD.get("TRAIN_SUPERVISION", True):
            raise RuntimeError("A-based EXP013C must disable frozen geometry supervision")
        if "attention_scale_init" in pnp.INIT_CFG:
            raise RuntimeError("A-based EXP013C must not inherit B attention settings")
    if expected_type is OfficialConvPnPNetRandomInit:
        if pose.GEO_HEAD.get("TRAIN_SUPERVISION", True):
            raise RuntimeError(
                "EXP013E must disable frozen geometry supervision so no CPP/EGL "
                "training renderer is ever constructed"
            )
    if expected_type is GLMPoseLNet:
        if pose.GEO_HEAD.get("TRAIN_SUPERVISION", True):
            raise RuntimeError(
                "EXP013F must disable frozen geometry supervision so no CPP/EGL "
                "training renderer is ever constructed"
            )
        if not pnp.INIT_CFG.get("use_depth_stats", False):
            raise RuntimeError("EXP013F preregisters use_depth_stats=True")
        if not cfg.INPUT.get("HEAD_DEPTH", False):
            raise RuntimeError(
                "EXP013F requires INPUT.HEAD_DEPTH=True to feed the depth statistics"
            )


def load_official_shared_state(
    model: torch.nn.Module, official_state: dict[str, torch.Tensor]
) -> None:
    incompatible = model.load_state_dict(dict(official_state), strict=False)
    expected_missing = {f"pnp_net.{key}" for key in model.pnp_net.state_dict()}
    if (
        set(incompatible.missing_keys) != expected_missing
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Official migration mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def synthetic_full_inputs(
    device: torch.device, batch: int = 1
) -> dict[str, torch.Tensor]:
    axis = torch.linspace(0.0, 1.0, 64, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return {
        "x": torch.rand(batch, 3, 256, 256, device=device),
        "roi_classes": torch.zeros(batch, dtype=torch.long, device=device),
        "roi_cams": torch.tensor(
            [[[572.0, 0.0, 320.0], [0.0, 572.0, 240.0], [0.0, 0.0, 1.0]]],
            device=device,
        ).repeat(batch, 1, 1),
        "roi_whs": torch.tensor([[160.0, 160.0]], device=device).repeat(batch, 1),
        "roi_centers": torch.tensor([[320.0, 240.0]], device=device).repeat(batch, 1),
        "resize_ratios": torch.ones(batch, device=device),
        "roi_coord_2d": torch.stack([xx, yy], dim=0)
        .unsqueeze(0)
        .repeat(batch, 1, 1, 1),
        "roi_extents": torch.tensor([[0.102, 0.102, 0.140]], device=device).repeat(
            batch, 1
        ),
    }


def full_forward_backward_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    # Eval mode keeps frozen running-stat buffers unchanged while autograd and
    # the optimizer remain fully active for pnp_net parameters.
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
    gradients = [
        parameter.grad
        for parameter in model.pnp_net.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("Pose-head optimizer probe produced invalid gradients")
    if isinstance(model.pnp_net, RTDecoupledGeometryPnPNet):
        for scale_name in ("geometry_scale_r", "geometry_scale_t"):
            gradient = getattr(model.pnp_net, scale_name).grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"Missing or non-finite C gradient: {scale_name}")
    optimizer.step()


def profile_head(
    head: torch.nn.Module, device: torch.device
) -> dict[str, float | int | None]:
    coor = torch.rand(1, 5, 64, 64, device=device)
    region = torch.softmax(torch.randn(1, 64, 64, 64, device=device), dim=1)
    extents = torch.rand(1, 3, device=device) + 0.1
    mask = torch.rand(1, 1, 64, 64, device=device)
    inputs = (coor, region, extents, mask)
    flops = None
    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(head, inputs)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        flops = int(analysis.total())
    except Exception:
        flops = None
    head.eval()
    for _ in range(3):
        with torch.no_grad():
            head(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            head(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 100.0
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return {
        "trainable_parameters": sum(
            p.numel() for p in head.parameters() if p.requires_grad
        ),
        "fvcore_supported_flops_per_64x64_sample": flops,
        "head_forward_latency_ms_batch1": round(latency_ms, 4),
        "peak_vram_bytes_batch1": peak_vram,
    }


def depth_stats_sensitivity_check(
    head: GLMPoseLNet, device: torch.device
) -> dict[str, bool]:
    """EXP013F: translation must consume depth statistics while rotation must
    not, and the zero padding must equal an explicit zero vector."""
    head = head.to(device).eval()
    coor = torch.rand(1, 5, 64, 64, device=device)
    region = torch.softmax(torch.randn(1, 64, 64, 64, device=device), dim=1)
    extents = torch.rand(1, 3, device=device) + 0.1
    mask = torch.rand(1, 1, 64, 64, device=device)
    with torch.no_grad():
        rot_none, t_none = head(coor, region=region, extents=extents, mask_attention=mask)
        zeros = torch.zeros(1, head.depth_stats_dim, device=device)
        rot_zero, t_zero = head(
            coor, region=region, extents=extents, mask_attention=mask, depth_stats=zeros
        )
        probe = torch.full((1, head.depth_stats_dim), 0.5, device=device)
        rot_probe, t_probe = head(
            coor, region=region, extents=extents, mask_attention=mask, depth_stats=probe
        )
    if not torch.equal(t_none, t_zero) or not torch.equal(rot_none, rot_zero):
        raise RuntimeError("EXP013F zero-padding is not equivalent to depth_stats=None")
    if torch.equal(t_zero, t_probe):
        raise RuntimeError("EXP013F translation ignores depth statistics")
    if not torch.equal(rot_zero, rot_probe):
        raise RuntimeError("EXP013F rotation must not depend on depth statistics")
    return {
        "depth_stats_zero_padding_equivalent": True,
        "depth_stats_translation_sensitive": True,
        "depth_stats_rotation_independent": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
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
    relative_config, expected_type = VARIANTS[args.variant]
    config_path = (args.config or PROJECT_ROOT / relative_config).resolve()
    weights_path = args.weights.resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cfg = Config.fromfile(str(config_path))
    validate_config(cfg, expected_type)
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    cfg.MODEL.DEVICE = args.device
    model, optimizer = build_model_optimizer(cfg, is_test=False)
    if optimizer is None or not isinstance(model.pnp_net, expected_type):
        raise RuntimeError("Training build returned the wrong model or no optimizer")
    load_official_shared_state(model, checkpoint_model_state(weights_path))
    trainable = [
        name for name, value in model.named_parameters() if value.requires_grad
    ]
    if not trainable or any(not name.startswith("pnp_net.") for name in trainable):
        raise RuntimeError(f"Unexpected trainable tensors: {trainable}")
    device = torch.device(args.device)
    model.to(device)
    full_forward_backward_step(model, optimizer, device)
    depth_stats_check = None
    if isinstance(model.pnp_net, GLMPoseLNet):
        depth_stats_check = depth_stats_sensitivity_check(model.pnp_net, device)
        head_params = sum(p.numel() for p in model.pnp_net.parameters())
        if not 800_000 <= head_params <= 1_100_000:
            raise RuntimeError(
                f"EXP013F head parameter budget violated: {head_params}"
            )
    profile = profile_head(model.pnp_net, device)
    print(
        json.dumps(
            {
                "status": "PASS",
                "variant": args.variant,
                "config": str(config_path),
                "device": args.device,
                "dtype": "float32",
                "only_pnp_net_trainable": True,
                "full_model_forward_backward_optimizer_step": True,
                **(depth_stats_check or {}),
                **profile,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
