#!/usr/bin/env python3
"""Run a bounded real LM-PBR optimizer smoke for EXP017 (never formal)."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
from detectron2.utils.events import EventStorage
from mmcv import Config

from core.gdrn_modeling.datasets.data_loader import build_gdrn_train_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data
from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from core.gdrn_modeling.models.heads.exp017_rotation_residual_pnp_net import (
    SupportAwareRotationResidualPnPNet,
)
from core.gdrn_modeling.models.model_utils import get_pnp_net
from research.diagnostics.pose_structure.model_access import (
    capture_model_pose_call,
    make_model_kwargs,
    model_input_from_batch,
)
from research.diagnostics.pose_structure.runtime import set_seed
from research.exp013.preflight import (
    PROJECT_ROOT,
    checkpoint_model_state,
    load_official_shared_state,
)
from research.exp017.preflight import CONFIG, EXPECTED_ADAPTER_PARAMETERS, validate_config


A_CONFIG = "configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual/train.py"


def _versions(named_parameters):
    return {name: parameter._version for name, parameter in named_parameters}


def _version_changed(before, named_parameters):
    return [
        name
        for name, parameter in named_parameters
        if parameter._version != before[name]
    ]


def _snapshot(named_parameters):
    return {
        name: parameter.detach().clone() for name, parameter in named_parameters
    }


def _value_changed(before, named_parameters):
    return [
        name
        for name, parameter in named_parameters
        if not torch.equal(parameter.detach(), before[name])
    ]


def _all_finite(values) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def _adapter_named(head):
    return [
        (name, parameter)
        for name, parameter in head.named_parameters()
        if name == "alpha_r" or name.startswith("rotation_adapter.")
    ]


def _a_named(head):
    return [
        (name, parameter)
        for name, parameter in head.named_parameters()
        if name != "alpha_r" and not name.startswith("rotation_adapter.")
    ]


def _training_step(model, optimizer, cfg, batch, storage_iteration: int):
    optimizer.zero_grad(set_to_none=True)
    with EventStorage(storage_iteration):
        _out, loss_dict = model(
            model_input_from_batch(cfg, batch),
            **make_model_kwargs(batch, do_loss=True),
        )
    loss = sum(loss_dict.values())
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"Non-finite real smoke loss: {loss_dict}")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.pnp_net.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not _all_finite(gradients):
        raise RuntimeError("Real smoke produced missing or non-finite pose gradients")
    optimizer.step()
    return float(loss.detach().cpu()), {
        name: float(value.detach().cpu()) for name, value in loss_dict.items()
    }


def run_smoke(
    config_path: Path,
    weights_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, object]:
    cfg = Config.fromfile(str(config_path))
    a_cfg = Config.fromfile(str(PROJECT_ROOT / A_CONFIG))
    # Validate the inherited formal config rather than the one-epoch smoke
    # overrides, then apply bounded local loader settings below.
    formal_cfg = Config.fromfile(str(PROJECT_ROOT / CONFIG))
    validate_config(formal_cfg, a_cfg)
    set_seed(seed)
    cfg.MODEL.DEVICE = str(device)
    cfg.DATALOADER.NUM_WORKERS = int(num_workers)
    cfg.DATALOADER.PERSISTENT_WORKERS = False
    cfg.SOLVER.IMS_PER_BATCH = int(batch_size)
    cfg.SOLVER.REFERENCE_BS = int(batch_size)
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    cfg.SOLVER.OPTIMIZER_NAME = cfg.SOLVER.OPTIMIZER_CFG.type
    cfg.SOLVER.WEIGHT_DECAY = float(cfg.SOLVER.OPTIMIZER_CFG.weight_decay)
    register_datasets_in_cfg(cfg)

    model, optimizer = build_model_optimizer(cfg, is_test=False)
    if optimizer is None or not isinstance(
        model.pnp_net, SupportAwareRotationResidualPnPNet
    ):
        raise RuntimeError("EXP017 real smoke built the wrong head or no optimizer")
    load_official_shared_state(model, checkpoint_model_state(weights_path))
    model.to(device).train()
    head = model.pnp_net
    if head.adapter_parameter_count() != EXPECTED_ADAPTER_PARAMETERS:
        raise RuntimeError("EXP017 real smoke adapter parameter count changed")

    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    frozen_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(("backbone.", "geo_head_net."))
    ]
    if not frozen_named or any(parameter.requires_grad for _, parameter in frozen_named):
        raise RuntimeError("Backbone/geometry freeze contract is broken")
    if any(id(parameter) in optimizer_ids for _, parameter in frozen_named):
        raise RuntimeError("Optimizer contains frozen backbone/geometry parameters")
    frozen_versions = _versions(frozen_named)

    loader = build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN)
    renderer = None
    try:
        raw_data = next(iter(loader))
        batch = batch_data(
            cfg,
            raw_data,
            renderer=renderer,
            device=str(device),
            phase="train",
        )

        # Capture the exact real head inputs once.  Translation-only backward
        # and update must leave every adapter tensor untouched.
        model.eval()
        with torch.no_grad():
            _out, captured = capture_model_pose_call(
                model, cfg, batch, do_loss=False
            )
        model.train()
        adapter_named = _adapter_named(head)
        adapter_before_translation = _snapshot(adapter_named)
        optimizer.zero_grad(set_to_none=True)
        _raw_r, raw_t = head(
            captured.coor_feat,
            region=captured.region,
            extents=captured.extents,
            mask_attention=captured.mask_attention,
        )
        translation_probe_loss = raw_t.square().mean()
        translation_probe_loss.backward()
        if any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad) != 0
            for _, parameter in adapter_named
        ):
            raise RuntimeError("Translation-only real probe reached adapter gradients")
        optimizer.step()
        if _value_changed(adapter_before_translation, adapter_named):
            raise RuntimeError("Translation-only optimizer step changed adapter parameters")

        a_named = _a_named(head)
        a_before_training = _snapshot(a_named)
        adapter_before_training = _snapshot(adapter_named)
        first_loss, first_loss_dict = _training_step(model, optimizer, cfg, batch, 0)
        second_loss, second_loss_dict = _training_step(model, optimizer, cfg, batch, 1)

        changed_a = _value_changed(a_before_training, a_named)
        changed_adapter = _value_changed(adapter_before_training, adapter_named)
        if not changed_a:
            raise RuntimeError("Real smoke did not update the inherited A pose head")
        if not changed_adapter:
            raise RuntimeError("Real smoke did not update EXP017 adapter")
        internal_names = (
            "rotation_adapter.token_projection.weight",
            "rotation_adapter.position_embedding",
            "rotation_adapter.pool_score.0.weight",
            "rotation_adapter.pool_score.2.weight",
        )
        internal_gradients = {
            name: dict(head.named_parameters())[name].grad for name in internal_names
        }
        if any(
            gradient is None
            or not bool(torch.isfinite(gradient).all())
            or torch.count_nonzero(gradient) == 0
            for gradient in internal_gradients.values()
        ):
            raise RuntimeError("Adapter internal tensors lack second-step real gradients")
        if _version_changed(frozen_versions, frozen_named):
            raise RuntimeError("Real smoke changed frozen backbone/geometry parameters")
        if any(parameter.grad is not None for _, parameter in frozen_named):
            raise RuntimeError("Frozen backbone/geometry unexpectedly received gradients")

        # Mechanism statistics are descriptive smoke checks only.
        head.eval()
        with torch.no_grad():
            raw_r, raw_t, info = head.forward_with_adapter_intervention(
                captured.coor_feat,
                region=captured.region,
                extents=captured.extents,
                mask_attention=captured.mask_attention,
            )
        weights, valid = info["weights"], info["valid"]
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1).mean()
        valid_mass = (weights * valid.to(weights.dtype)).sum(dim=1).mean()
        baseline_rms = info["raw_r_a"].float().square().mean().sqrt()
        residual = head.alpha_r.to(info["delta_r"].dtype) * info["delta_r"]
        residual_rms = residual.float().square().mean().sqrt()
        if not _all_finite((raw_r, raw_t, entropy, valid_mass, baseline_rms, residual_rms)):
            raise RuntimeError("Real smoke outputs or mechanism statistics are non-finite")
        if torch.count_nonzero(weights[~valid]) != 0:
            raise RuntimeError("Real smoke assigned weight to invalid support tokens")

        # In-memory strict checkpoint round-trip of the trained pose head.
        payload = io.BytesIO()
        torch.save(head.state_dict(), payload)
        payload.seek(0)
        restored, _ = get_pnp_net(cfg)
        restored.to(device).eval()
        restored.load_state_dict(torch.load(payload, map_location=device), strict=True)
        with torch.no_grad():
            roundtrip_r, roundtrip_t = restored(
                captured.coor_feat,
                region=captured.region,
                extents=captured.extents,
                mask_attention=captured.mask_attention,
            )
        if not torch.equal(roundtrip_r, raw_r) or not torch.equal(roundtrip_t, raw_t):
            raise RuntimeError("EXP017 real smoke checkpoint round-trip is not value-exact")

        return {
            "status": "PASS",
            "formal_training": False,
            "dataset": cfg.DATASETS.TRAIN[0],
            "batch_size": batch_size,
            "samples": len(raw_data),
            "device": str(device),
            "optimizer_steps": 3,
            "translation_only_optimizer_steps": 1,
            "rotation_inclusive_optimizer_steps": 2,
            "translation_probe_loss": float(translation_probe_loss.detach().cpu()),
            "first_full_loss": first_loss,
            "second_full_loss": second_loss,
            "first_loss_dict": first_loss_dict,
            "second_loss_dict": second_loss_dict,
            "outputs_finite": True,
            "losses_finite": True,
            "gradients_finite": True,
            "backbone_geometry_frozen_unchanged": True,
            "a_pose_head_updated": True,
            "adapter_updated": True,
            "translation_only_adapter_unchanged": True,
            "checkpoint_roundtrip_value_exact": True,
            "adapter_parameters": head.adapter_parameter_count(),
            "alpha_r": float(head.alpha_r.detach().cpu()),
            "pooling_entropy": float(entropy.detach().cpu()),
            "valid_support_mass": float(valid_mass.detach().cpu()),
            "raw_r_baseline_rms": float(baseline_rms.detach().cpu()),
            "rotation_residual_rms": float(residual_rms.detach().cpu()),
            "changed_a_parameter_tensors": len(changed_a),
            "changed_adapter_parameter_tensors": len(changed_adapter),
            "internal_real_gradients_nonzero": True,
        }
    finally:
        if hasattr(renderer, "close"):
            renderer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/gdrn/lmo_pbr/research/exp017/"
        "support_aware_rotation_residual/smoke.py",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config_path = args.config.resolve()
    weights_path = args.weights.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    result = run_smoke(
        config_path,
        weights_path,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
