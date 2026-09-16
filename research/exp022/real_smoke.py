#!/usr/bin/env python3
"""One genuine LM-O online-geometry CUDA batch; optional short timing sample."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from detectron2.data import MetadataCatalog
from mmcv import Config

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_train_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, get_renderer
from core.gdrn_modeling.models.GDRN_PCC import build_model_optimizer
from research.exp022.preflight import CONFIG_ROOT, load_official_backbone
from research.run_contract import validate_research_run_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("smoke_reused.py", "smoke_independent.py"),
                        default="smoke_reused.py")
    parser.add_argument("--weights", type=Path,
                        default=Path("pretrained_models/lmo_pbr/model_final_wo_optim.pth"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", choices=("cpp", "egl"), default="cpp")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Real EXP022 smoke requires CUDA")
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("--steps and --batch-size must be positive")
    cfg = Config.fromfile(str(CONFIG_ROOT / args.config))
    validate_research_run_config(cfg, mode="smoke",
                                 expected_experiment_id="EXP-20260916-022-progressive-pcc")
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.POSE_NET.XYZ_RENDERER = args.renderer
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.PERSISTENT_WORKERS = False
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.REFERENCE_BS = args.batch_size
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    register_datasets_in_cfg(cfg)
    model, optimizer = build_model_optimizer(cfg)
    loaded = load_official_backbone(model, args.weights)
    model.train()
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    data_ref = ref.__dict__[metadata.ref_key]
    renderer = get_renderer(cfg, data_ref, obj_names=metadata.objs,
                            gpu_id=torch.device(args.device).index or 0)
    timings, history = [], []
    try:
        iterator = iter(build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN))
        raw = next(iterator)
        batch = batch_data(cfg, raw, renderer=renderer, device=args.device, phase="train")
        image, classes = batch["roi_img"], batch["roi_cls"]
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=True):
                _, losses = model(image, roi_classes=classes, gt_xyz=batch["roi_xyz"],
                                  gt_mask_visib=batch["roi_mask_visib"], do_loss=True)
                total = sum(losses.values())
            if not torch.isfinite(total):
                raise RuntimeError("Non-finite EXP022 smoke loss")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in model.pcc_head.parameters()):
                raise RuntimeError("Non-finite EXP022 smoke gradient")
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                raise RuntimeError("EXP022 AMP skipped an optimizer step")
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000)
            history.append({name: float(value.detach()) for name, value in losses.items()})
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            output = model(image, roi_classes=classes, return_pcc_debug=True)
        if not torch.isfinite(output["xyz_norm"]).all():
            raise RuntimeError("Non-finite EXP022 inference XYZ")
        if output["residual_norm"].max() > 1.00001:
            raise RuntimeError("EXP022 residual bound violated")
        print(json.dumps({"status": "PASS", "config": args.config, "renderer": args.renderer,
                          "batch_size": args.batch_size, "steps": args.steps,
                          "official_backbone_tensors": loaded,
                          "trainable_parameters": sum(p.numel() for p in model.pcc_head.parameters()),
                          "losses": history, "step_median_ms": statistics.median(timings),
                          "step_times_ms": timings,
                          "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}, indent=2))
    finally:
        if hasattr(renderer, "close"):
            renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
