#!/usr/bin/env python3
"""Fixed-real-batch CUDA/EGL optimization smoke for EXP021 B/C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.events import EventStorage
from mmcv import Config

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_train_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, get_renderer
from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from research.diagnostics.pose_structure.model_access import make_model_kwargs, model_input_from_batch
from research.diagnostics.pose_structure.runtime import set_seed
from research.exp013.preflight import PROJECT_ROOT, checkpoint_model_state
from research.exp021.preflight import EXPERIMENT_ID
from research.exp021.precision import (
    PRECISION_CHOICES,
    autocast_context,
    grad_scaler,
    gradients_are_finite,
    resolve_precision,
)
from research.run_contract import validate_research_run_config


CONFIG_ROOT = PROJECT_ROOT / "configs/gdrn/lmo_pbr/research/exp021_global_hierarchical_cad"


def run_arm(
    arm: str,
    weights: Path,
    device: torch.device,
    batch_size: int,
    renderer_type: str = "egl",
    precision: str = "config",
    steps: int = 1,
) -> dict:
    config = CONFIG_ROOT / ("b_hierarchical.py" if arm == "B" else "c_global.py")
    cfg = Config.fromfile(str(config))
    validate_research_run_config(cfg, mode="formal", expected_experiment_id=EXPERIMENT_ID)
    cfg.MODEL.POSE_NET.XYZ_RENDERER = renderer_type
    set_seed(42)
    cfg.MODEL.DEVICE = str(device)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.PERSISTENT_WORKERS = False
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.REFERENCE_BS = batch_size
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    effective_precision = resolve_precision(cfg, precision)
    register_datasets_in_cfg(cfg)
    model, optimizer = build_model_optimizer(cfg)
    incompatible = model.load_state_dict(dict(checkpoint_model_state(weights)), strict=False)
    if incompatible.unexpected_keys or any(not key.startswith("cad_head.") for key in incompatible.missing_keys):
        raise RuntimeError(f"Official checkpoint incompatibility: {incompatible}")
    model.train()
    scaler = grad_scaler(effective_precision)
    frozen_before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    data_ref = ref.__dict__[metadata.ref_key]
    renderer = get_renderer(cfg, data_ref, obj_names=metadata.objs, gpu_id=device.index or 0)
    try:
        raw = next(iter(build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN)))
        batch = batch_data(cfg, raw, renderer=renderer, device=str(device), phase="train")
        kwargs = make_model_kwargs(batch, do_loss=True)
        kwargs["roi_zoom_cams"] = batch.get("roi_zoom_K")
        kwargs["cad_beam_ks"] = (4,)
        torch.cuda.reset_peak_memory_stats(device)
        scale_before = float(scaler.get_scale())
        trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
        trainable_before = {
            name: value.detach().clone() for name, value in trainable.items()
        }
        loss_history = []
        active = []
        with EventStorage() as storage:
            for step in range(steps):
                storage.iter = step
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(effective_precision, device):
                    _output, losses = model(
                        model_input_from_batch(cfg, batch), **kwargs
                    )
                if set(losses) != {
                    "loss_cad_coarse",
                    "loss_cad_fine",
                    "loss_cad_xyz",
                }:
                    raise RuntimeError(f"Unexpected losses: {sorted(losses)}")
                total = sum(losses.values())
                if not torch.isfinite(total):
                    raise RuntimeError(f"Non-finite EXP021 loss at step {step}")
                loss_history.append(
                    {
                        **{
                            name: float(value.detach())
                            for name, value in losses.items()
                        },
                        "total": float(total.detach()),
                    }
                )
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                active = [
                    name
                    for name, value in trainable.items()
                    if value.grad is not None
                    and torch.count_nonzero(value.grad).item() > 0
                ]
                if not active or any(
                    not name.startswith("cad_head.") for name in active
                ):
                    raise RuntimeError(f"Invalid active gradients: {active}")
                if not gradients_are_finite(trainable.values()):
                    raise RuntimeError(
                        f"EXP021 smoke produced non-finite gradients at step {step}"
                    )
                scaler.step(optimizer)
                scaler.update()
        optimizer_step_applied = any(
            not torch.equal(trainable_before[name], value.detach())
            for name, value in trainable.items()
        )
        if not optimizer_step_applied:
            raise RuntimeError("EXP021 smoke optimizer step did not update CAD parameters")
        changed_frozen = [
            name
            for name, value in model.named_parameters()
            if name in frozen_before and not torch.equal(frozen_before[name], value.detach())
        ]
        if changed_frozen:
            raise RuntimeError(f"Frozen parameters changed: {changed_frozen[:5]}")
        if arm == "C" and not any(name.startswith("cad_head.global_") for name in active):
            raise RuntimeError("C smoke did not route gradients through global guidance")
        loss_descent = None
        if steps > 1:
            window = min(5, steps // 2)
            first_mean = sum(row["total"] for row in loss_history[:window]) / window
            last_mean = sum(row["total"] for row in loss_history[-window:]) / window
            loss_descent = {
                "window": window,
                "first_mean": first_mean,
                "last_mean": last_mean,
                "passed": last_mean < first_mean,
            }
            if not loss_descent["passed"]:
                raise RuntimeError(
                    "EXP021 fixed-batch loss did not decrease: "
                    f"first={first_mean}, last={last_mean}"
                )
        return {
            "status": "PASS",
            "arm": arm,
            "real_data": True,
            "formal_training": False,
            "batch_size": batch_size,
            "steps": steps,
            "renderer_type": renderer_type,
            "formal_renderer_match": renderer_type == "egl",
            "precision": effective_precision,
            "amp_enabled": effective_precision == "amp-fp16",
            "grad_scale_before": scale_before,
            "grad_scale_after": float(scaler.get_scale()),
            "optimizer_step_applied": optimizer_step_applied,
            "trainable_parameters": sum(value.numel() for value in trainable.values()),
            "active_gradient_tensors": len(active),
            "losses": loss_history[-1],
            "loss_history": loss_history,
            "loss_descent": loss_descent,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "frozen_parameters_unchanged": True,
        }
    finally:
        renderer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("B", "C", "both"), default="both")
    parser.add_argument(
        "--weights", type=Path, default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--renderer-type", choices=("egl", "cpp"), default="egl")
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="config")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1:
        parser.error("--batch-size and --steps must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("EXP021 real smoke requires CUDA/EGL")
    arms = ("B", "C") if args.arm == "both" else (args.arm,)
    report = {
        arm: run_arm(
            arm,
            args.weights,
            device,
            args.batch_size,
            args.renderer_type,
            args.precision,
            args.steps,
        )
        for arm in arms
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
