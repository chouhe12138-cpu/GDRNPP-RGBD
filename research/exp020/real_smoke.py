#!/usr/bin/env python3
"""Bounded real LM-PBR optimizer smoke for EXP020 (never formal).

Loads a genuine online-geometry batch (cpp XYZ renderer) and runs a few
forward/backward/optimizer steps for the selected arm(s). It exercises the
REPROJ_LW=0 legacy path (A) and the REPROJ_LW>0 path (B) with the actual
batch_data -> GDRN_DoubleMask -> gdrn_loss wiring including roi_zoom_K.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.events import EventStorage, get_event_storage
from mmcv import Config

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_train_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, get_renderer
from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from research.diagnostics.pose_structure.model_access import (
    make_model_kwargs,
    model_input_from_batch,
)
from research.diagnostics.pose_structure.runtime import set_seed
from research.exp013.preflight import PROJECT_ROOT, checkpoint_model_state

ARMS = {
    "A": (
        "configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr/smoke_control.py",
        0.0,
    ),
    "B": (
        "configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr/smoke_reproj.py",
        1.0,
    ),
}
FORMAL = {
    "A": "configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr/control.py",
    "B": "configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr/reproj.py",
}
EXPERIMENT_ID = "EXP-20260909-020-geometry-aware-correspondence-loss"


def _snapshot(named_parameters):
    return {
        name: parameter.detach().clone() for name, parameter in named_parameters
    }


def _value_changed(before, named_parameters):
    return [
        name
        for name, parameter in named_parameters
        if name in before and not torch.equal(parameter.detach(), before[name])
    ]


def run_arm(
    arm: str,
    config_path: Path,
    formal_path: Path,
    weights_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    steps: int,
    expected_lw: float,
) -> dict[str, object]:
    from research.run_contract import validate_research_run_config

    cfg = Config.fromfile(str(config_path))
    formal_cfg = Config.fromfile(str(formal_path))
    validate_research_run_config(
        formal_cfg, mode="formal", expected_experiment_id=EXPERIMENT_ID
    )
    validate_research_run_config(
        cfg, mode="smoke", expected_experiment_id=EXPERIMENT_ID
    )
    if float(cfg.MODEL.POSE_NET.LOSS_CFG.REPROJ_LW) != expected_lw:
        raise RuntimeError("Smoke config REPROJ_LW does not match the arm")

    set_seed(42)
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
    state = checkpoint_model_state(weights_path)
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys or incompatible.missing_keys:
        raise RuntimeError(
            "Official checkpoint compatibility broken: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device).train()

    trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any(not name.startswith("geo_head_net.") for name in trainable):
        raise RuntimeError(f"Real smoke found unexpected trainable tensors: {trainable}")

    train_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    data_ref = ref.__dict__[train_meta.ref_key]
    renderer = get_renderer(
        cfg,
        data_ref,
        obj_names=train_meta.objs,
        gpu_id=device.index if device.type == "cuda" else 0,
    )
    try:
        loader = build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN)
        raw_data = next(iter(loader))
        batch = batch_data(
            cfg,
            raw_data,
            renderer=renderer,
            device=str(device),
            phase="train",
        )
        if "roi_zoom_K" not in batch:
            raise RuntimeError("Online geometry batch is missing roi_zoom_K")

        frozen_named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith(("backbone.", "pnp_net."))
        ]
        frozen_before = _snapshot(frozen_named)
        geo_before = _snapshot(model.geo_head_net.named_parameters())

        recorded = []
        for step in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            kwargs = make_model_kwargs(batch, do_loss=True)
            kwargs["roi_zoom_cams"] = batch.get("roi_zoom_K", None)
            reproj_diag = None
            with EventStorage(step):
                _out, loss_dict = model(model_input_from_batch(cfg, batch), **kwargs)
                # gdrn_loss pushes the loss module's own diagnostics through
                # EventStorage; read the latest values (loss value stays in
                # loss_dict). ``vis/reproj_px`` is the module-computed Euclidean
                # mean pixel error -- never ``loss_xyz_reproj * 64``.
                if expected_lw > 0:
                    latest = get_event_storage().latest()
                    reproj_diag = {
                        "reproj_loss": float(
                            loss_dict["loss_xyz_reproj"].detach().cpu()
                        ),
                        "mean_reproj_px": latest.get("vis/reproj_px", (None, 0))[0],
                        "valid_ratio": latest.get(
                            "vis/reproj_valid_ratio", (None, 0)
                        )[0],
                        "gt_foreground_count": latest.get(
                            "vis/reproj_gt_fg_count", (None, 0)
                        )[0],
                        "positive_depth_ratio_on_gt_fg": latest.get(
                            "vis/reproj_positive_depth_ratio", (None, 0)
                        )[0],
                        "behind_camera_ratio_on_gt_fg": latest.get(
                            "vis/reproj_behind_camera_ratio", (None, 0)
                        )[0],
                    }
            keys = set(loss_dict)
            if expected_lw > 0:
                if "loss_xyz_reproj" not in keys:
                    raise RuntimeError(f"Arm B step {step} lost reprojection loss")
            elif "loss_xyz_reproj" in keys:
                raise RuntimeError(f"Arm A step {step} unexpectedly has reprojection loss")
            total = sum(loss_dict.values())
            if not bool(torch.isfinite(total)):
                raise RuntimeError(f"Non-finite real smoke loss: {loss_dict}")
            total.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is None or not bool(
                        torch.isfinite(parameter.grad).all()
                    ):
                        raise RuntimeError(
                            f"Missing/non-finite gradient on {name}"
                        )
                elif parameter.grad is not None:
                    raise RuntimeError(f"Frozen parameter got a gradient: {name}")
            optimizer.step()
            if not all(
                bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
            ):
                raise RuntimeError(f"Optimizer step {step} made parameters non-finite")
            geo_changed = _value_changed(geo_before, model.geo_head_net.named_parameters())
            if not geo_changed:
                raise RuntimeError(f"Real smoke step {step} did not update geometry head")
            frozen_changed = _value_changed(frozen_before, frozen_named)
            if frozen_changed:
                raise RuntimeError(f"Frozen tensors changed after step: {frozen_changed}")
            recorded.append(
                {
                    "step": step,
                    "total_loss": float(total.detach().cpu()),
                    "loss_keys": sorted(keys),
                    "reproj": reproj_diag,
                }
            )
        return {
            "arm": arm,
            "steps": recorded,
            "batch_size": int(batch_size),
            "trainable_geometry_only": True,
            "frozen_backbone_and_pnp": True,
            "roi_zoom_K_present": True,
        }
    finally:
        renderer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B", "both"), default="both")
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    selected = ("A", "B") if args.arm == "both" else (args.arm,)
    results = []
    for arm in selected:
        config_path, expected_lw = ARMS[arm]
        formal_path = PROJECT_ROOT / FORMAL[arm]
        results.append(
            run_arm(
                arm,
                PROJECT_ROOT / config_path,
                formal_path,
                args.weights,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                steps=args.steps,
                expected_lw=expected_lw,
            )
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "device": str(device),
                "arms": results,
                "real_data": True,
                "formal_training": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
