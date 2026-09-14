#!/usr/bin/env python3
"""Profile EXP021 training phases on genuine online-geometry batches."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
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
from research.exp021.precision import (
    PRECISION_CHOICES,
    autocast_context,
    grad_scaler,
    gradients_are_finite,
    resolve_precision,
)


CONFIG_ROOT = PROJECT_ROOT / "configs/gdrn/lmo_pbr/research/exp021_global_hierarchical_cad"
FILE_LIMIT = 500_000


def _raise_file_limit() -> None:
    _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(FILE_LIMIT, hard), hard))


def _timed(function, *, synchronize: bool = False):
    if synchronize:
        torch.cuda.synchronize()
    started = time.perf_counter()
    value = function()
    if synchronize:
        torch.cuda.synchronize()
    return value, (time.perf_counter() - started) * 1000.0


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def profile(args: argparse.Namespace) -> dict:
    _raise_file_limit()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("EXP021 training profiling requires CUDA")
    config = CONFIG_ROOT / ("smoke_b.py" if args.arm == "B" else "smoke_c.py")
    cfg = Config.fromfile(str(config))
    cfg.MODEL.DEVICE = str(device)
    cfg.MODEL.POSE_NET.XYZ_RENDERER = args.renderer_type
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.PERSISTENT_WORKERS = args.num_workers > 0
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.REFERENCE_BS = args.batch_size
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    effective_precision = resolve_precision(cfg, args.precision)
    set_seed(42)
    register_datasets_in_cfg(cfg)

    model, optimizer = build_model_optimizer(cfg)
    incompatible = model.load_state_dict(
        dict(checkpoint_model_state(args.weights)), strict=False
    )
    if incompatible.unexpected_keys or any(
        not key.startswith("cad_head.") for key in incompatible.missing_keys
    ):
        raise RuntimeError(f"Official checkpoint incompatibility: {incompatible}")
    model.train()
    scaler = grad_scaler(effective_precision)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    data_ref = ref.__dict__[metadata.ref_key]
    renderer = get_renderer(
        cfg, data_ref, obj_names=metadata.objs, gpu_id=device.index or 0
    )
    iterator = iter(build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN))
    rows = []
    try:
        torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(args.warmup + args.steps):
            raw, raw_ms = _timed(lambda: next(iterator))
            batch, batch_ms = _timed(
                lambda: batch_data(
                    cfg, raw, renderer=renderer, device=str(device), phase="train"
                ),
                synchronize=True,
            )
            model_input = model_input_from_batch(cfg, batch)
            kwargs = make_model_kwargs(batch, do_loss=True)
            kwargs["roi_zoom_cams"] = batch.get("roi_zoom_K")
            kwargs["cad_beam_ks"] = (4,)
            optimizer.zero_grad(set_to_none=True)

            def forward():
                with EventStorage(), autocast_context(effective_precision, device):
                    return model(model_input, **kwargs)[1]

            losses, forward_ms = _timed(forward, synchronize=True)
            _, backward_ms = _timed(
                lambda: scaler.scale(sum(losses.values())).backward(), synchronize=True
            )

            def optimizer_step():
                scaler.unscale_(optimizer)
                if not gradients_are_finite(model.parameters()):
                    raise RuntimeError("EXP021 profile produced non-finite unscaled gradients")
                scaler.step(optimizer)
                scaler.update()

            _, optimizer_ms = _timed(optimizer_step, synchronize=True)
            if iteration >= args.warmup:
                rows.append(
                    {
                        "data_loader_ms": raw_ms,
                        "batch_geometry_ms": batch_ms,
                        "forward_ms": forward_ms,
                        "backward_ms": backward_ms,
                        "optimizer_ms": optimizer_ms,
                    }
                )
    finally:
        renderer.close()

    phases = {name: _summary([row[name] for row in rows]) for name in rows[0]}
    totals = [sum(row.values()) for row in rows]
    total = _summary(totals)
    total["samples_per_second"] = args.batch_size / (total["mean_ms"] / 1000.0)
    return {
        "status": "PASS",
        "arm": args.arm,
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "renderer_type": args.renderer_type,
        "precision": effective_precision,
        "amp_enabled": effective_precision == "amp-fp16",
        "final_grad_scale": float(scaler.get_scale()),
        "nonfinite_or_skipped_steps": 0,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "phases": phases,
        "total": total,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("B", "C"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--renderer-type", choices=("cpp", "egl"), default="cpp")
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="config")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.steps < 1 or args.num_workers < 0:
        parser.error("batch-size/steps must be positive; warmup/num-workers must be non-negative")
    report = profile(args)
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
