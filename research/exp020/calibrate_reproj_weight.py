#!/usr/bin/env python3
"""REPROJ_LW gradient-scale calibration for EXP020 (never a lambda sweep).

On one genuine online-geometry batch (EGL XYZ renderer), one model state
(official checkpoint, backbone/PNP_NET frozen, geometry head trainable), the
script compares the geometry-head gradient scale induced by

- Pass XYZ: only the three continuous XYZ supervision terms
  (``loss_coor_x + loss_coor_y + loss_coor_z``), and
- Pass REPROJ: only the raw per-pixel correspondence reprojection loss
  (``loss_xyz_reproj / REPROJ_LW``, i.e. without the loss weight).

It reports the global L2 gradient norm of ``geo_head_net`` for each pass plus
``ratio_raw = g_reproj_raw / g_xyz`` so a huge mismatch (tens to hundreds of
times) can be caught before a formal run.  This is a scale check only: it does
not change the formal ``REPROJ_LW`` and does not run an optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
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
from research.diagnostics.pose_structure.model_access import (
    make_model_kwargs,
    model_input_from_batch,
)
from research.diagnostics.pose_structure.runtime import set_seed
from research.exp013.preflight import PROJECT_ROOT, checkpoint_model_state
from research.exp020.real_smoke import EXPERIMENT_ID

REPROJ_FORMAL_CONFIG = (
    "configs/gdrn/lmo_pbr/research/exp020_geometry_aware_corr/reproj.py"
)
XYZ_KEYS = ("loss_coor_x", "loss_coor_y", "loss_coor_z")


def _l2_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().pow(2).sum())
    return math.sqrt(total)


def _grouped_grad_norms(module: torch.nn.Module) -> dict[str, float | None]:
    """Optional per-group L2 norms; ``None`` when no param matches the group.

    ``named_parameters()`` on the geo head yields module-relative names such
    as ``features.0.weight`` / ``out_layer.weight``.
    """
    groups = {
        "shared_trunk": lambda name: name.startswith("features."),
        "xyz_output_layer": lambda name: name.startswith("out_layer.")
        or name.startswith("xyz_out_layer."),
    }
    result: dict[str, float | None] = {}
    for group, match in groups.items():
        total = 0.0
        matched = 0
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            if not match(name):
                continue
            total += float(parameter.grad.detach().pow(2).sum())
            matched += 1
        result[group] = math.sqrt(total) if matched else None
    return result


def run(
    config_path: Path,
    weights_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, object]:
    from research.run_contract import validate_research_run_config

    cfg = Config.fromfile(str(config_path))
    validate_research_run_config(cfg, mode="formal", expected_experiment_id=EXPERIMENT_ID)
    reproj_lw = float(cfg.MODEL.POSE_NET.LOSS_CFG.get("REPROJ_LW", 0.0))
    if reproj_lw <= 0:
        raise RuntimeError("Calibration needs the REPROJ_LW>0 formal reproj config")

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

    model, _optimizer = build_model_optimizer(cfg, is_test=False)
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
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any(not name.startswith("geo_head_net.") for name in trainable):
        raise RuntimeError(f"Unexpected trainable tensors: {trainable}")

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
        batch = batch_data(cfg, raw_data, renderer=renderer, device=str(device), phase="train")
        if "roi_zoom_K" not in batch:
            raise RuntimeError("Online geometry batch is missing roi_zoom_K")

        kwargs = make_model_kwargs(batch, do_loss=True)
        kwargs["roi_zoom_cams"] = batch.get("roi_zoom_K", None)
        with EventStorage():
            _out, loss_dict = model(model_input_from_batch(cfg, batch), **kwargs)
        missing = [key for key in XYZ_KEYS + ("loss_xyz_reproj",) if key not in loss_dict]
        if missing:
            raise RuntimeError(f"Calibration forward missed loss keys: {missing}")
        total = sum(loss_dict.values())
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"Non-finite calibration loss: {loss_dict}")

        # Pass XYZ: only the three continuous XYZ supervision terms.
        model.zero_grad(set_to_none=True)
        xyz_total = sum(loss_dict[key] for key in XYZ_KEYS)
        xyz_total.backward(retain_graph=True)
        g_xyz = _l2_grad_norm(model.geo_head_net)
        g_xyz_groups = _grouped_grad_norms(model.geo_head_net)

        # Pass REPROJ: only the raw (unweighted) reprojection loss.
        model.zero_grad(set_to_none=True)
        raw_reproj = loss_dict["loss_xyz_reproj"] / reproj_lw
        raw_reproj.backward()
        g_reproj_raw = _l2_grad_norm(model.geo_head_net)
        g_reproj_groups = _grouped_grad_norms(model.geo_head_net)

        ratio_raw = g_reproj_raw / max(g_xyz, 1e-12)
        return {
            "status": "PASS",
            "device": str(device),
            "config": str(config_path),
            "reproj_lw_config": reproj_lw,
            "loss_scales": {
                "xyz_sum": float(xyz_total.detach().cpu()),
                "reproj_raw": float(raw_reproj.detach().cpu()),
            },
            "g_xyz": g_xyz,
            "g_reproj_raw": g_reproj_raw,
            "ratio_raw": ratio_raw,
            "groups": {
                "xyz": g_xyz_groups,
                "reproj_raw": g_reproj_groups,
            },
            "real_data": True,
            "formal_training": False,
        }
    finally:
        renderer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / REPROJ_FORMAL_CONFIG,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    report = run(
        args.config,
        args.weights,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
