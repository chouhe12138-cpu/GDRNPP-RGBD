#!/usr/bin/env python3
"""One-real-batch gradient calibration for EXP021's three fixed losses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
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
    resolve_precision,
)
from research.run_contract import validate_research_run_config


CONFIG = PROJECT_ROOT / "configs/gdrn/lmo_pbr/research/exp021_global_hierarchical_cad/b_hierarchical.py"
LOSS_KEYS = ("loss_cad_coarse", "loss_cad_fine", "loss_cad_xyz")


def _grad_norm(module: torch.nn.Module, divisor: float = 1.0) -> float:
    return math.sqrt(
        sum(
            float((parameter.grad.detach().float() / divisor).square().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
    )


def recommended_weights(norms: dict[str, float]) -> dict[str, float]:
    values = np.asarray(list(norms.values()), dtype=np.float64)
    if np.any(values <= 0) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid EXP021 gradient norms: {norms}")
    median = float(np.median(values))
    ratios = values / median
    if np.all((ratios >= 0.1) & (ratios <= 10.0)):
        return {key: 1.0 for key in norms}
    output = {}
    for key, value in norms.items():
        exponent = round(math.log2(median / value))
        output[key] = float(np.clip(2.0**exponent, 1.0 / 16.0, 16.0))
    return output


def run(
    config: Path,
    weights: Path,
    device: torch.device,
    batch_size: int,
    renderer_type: str = "egl",
    precision: str = "config",
) -> dict:
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
    model, _optimizer = build_model_optimizer(cfg)
    scaler = grad_scaler(effective_precision)
    incompatible = model.load_state_dict(dict(checkpoint_model_state(weights)), strict=False)
    if incompatible.unexpected_keys or any(not key.startswith("cad_head.") for key in incompatible.missing_keys):
        raise RuntimeError(f"Official checkpoint incompatibility: {incompatible}")
    model.train()
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    data_ref = ref.__dict__[metadata.ref_key]
    renderer = get_renderer(cfg, data_ref, obj_names=metadata.objs, gpu_id=device.index or 0)
    try:
        raw = next(iter(build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN)))
        batch = batch_data(cfg, raw, renderer=renderer, device=str(device), phase="train")
        kwargs = make_model_kwargs(batch, do_loss=True)
        kwargs["roi_zoom_cams"] = batch.get("roi_zoom_K")
        kwargs["cad_beam_ks"] = (4,)
        with EventStorage(), autocast_context(effective_precision, device):
            _output, losses = model(model_input_from_batch(cfg, batch), **kwargs)
        if set(losses) != set(LOSS_KEYS):
            raise RuntimeError(f"Unexpected calibration losses: {sorted(losses)}")
        configured_weights = {
            "loss_cad_coarse": model.cad_head.coarse_loss_weight,
            "loss_cad_fine": model.cad_head.fine_loss_weight,
            "loss_cad_xyz": model.cad_head.xyz_loss_weight,
        }
        norms = {}
        for index, key in enumerate(LOSS_KEYS):
            model.zero_grad(set_to_none=True)
            raw_loss = losses[key] / configured_weights[key]
            scaler.scale(raw_loss).backward(
                retain_graph=index < len(LOSS_KEYS) - 1
            )
            scale = float(scaler.get_scale())
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
                for parameter in model.cad_head.parameters()
            ):
                raise RuntimeError(
                    f"Non-finite scaled gradients for {key} at scale {scale}"
                )
            norms[key] = _grad_norm(model.cad_head, divisor=scale)
        recommendation = recommended_weights(norms)
        weighted = {key: norms[key] * recommendation[key] for key in LOSS_KEYS}
        weighted_median = float(np.median(list(weighted.values())))
        weighted_ratios = {key: value / weighted_median for key, value in weighted.items()}
        status = "PASS" if all(0.25 <= value <= 4.0 for value in weighted_ratios.values()) else "FAIL"
        return {
            "status": status,
            "real_data": True,
            "formal_training": False,
            "config": str(config),
            "device": str(device),
            "batch_size": batch_size,
            "renderer_type": renderer_type,
            "formal_renderer_match": renderer_type == "egl",
            "precision": effective_precision,
            "amp_enabled": effective_precision == "amp-fp16",
            "grad_scale": float(scaler.get_scale()),
            "configured_weights": configured_weights,
            "raw_losses": {
                key: float(losses[key].detach()) / configured_weights[key]
                for key in LOSS_KEYS
            },
            "raw_gradient_norms": norms,
            "recommended_weights": recommendation,
            "weighted_gradient_ratios_to_median": weighted_ratios,
        }
    finally:
        renderer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--weights", type=Path, default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--renderer-type", choices=("egl", "cpp"), default="egl")
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="config")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("EXP021 calibration requires a real CUDA/EGL batch")
    report = run(
        args.config,
        args.weights,
        device,
        args.batch_size,
        args.renderer_type,
        args.precision,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if report["status"] != "PASS":
        raise RuntimeError("EXP021 calibrated weighted gradients remain outside [0.25,4]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
