#!/usr/bin/env python3
"""One genuine EXP022 online-geometry CUDA batch; optional short timing sample."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from detectron2.data import MetadataCatalog
from mmcv import Config

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_train_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, get_renderer
from core.gdrn_modeling.models.GDRN_PCC import build_model_optimizer
from research.exp022.preflight import CONFIG_ROOT, load_official_backbone
from research.exp022.dataset_context import resolve_dataset_context
from research.run_contract import validate_research_run_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_ROOT / "smoke_reused.py")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", choices=("cpp", "egl"), default="cpp")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostics", choices=("on", "off"), default="on")
    batch_files = parser.add_mutually_exclusive_group()
    batch_files.add_argument("--save-batch", type=Path)
    batch_files.add_argument("--load-batch", type=Path)
    args = parser.parse_args()
    warmup_steps = min(1, args.steps - 1) if args.warmup_steps is None else args.warmup_steps
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Real EXP022 smoke requires CUDA")
    if args.steps < 1 or args.batch_size < 1 or not 0 <= warmup_steps < args.steps:
        raise ValueError("Require positive steps/batch-size and 0 <= warmup-steps < steps")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config_path = args.config if args.config.is_absolute() or args.config.exists() else CONFIG_ROOT / args.config
    cfg = Config.fromfile(str(config_path))
    validate_research_run_config(cfg, mode="smoke",
                                 expected_experiment_id=cfg.EXPERIMENT_ID)
    context = resolve_dataset_context(cfg)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.POSE_NET.XYZ_RENDERER = args.renderer
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.PERSISTENT_WORKERS = False
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.REFERENCE_BS = args.batch_size
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    model, optimizer = build_model_optimizer(cfg)
    loaded = (load_official_backbone(model, args.weights or Path(cfg.MODEL.WEIGHTS))
              if cfg.MODEL.POSE_NET.BACKBONE.FREEZE else
              sum(1 for _ in model.backbone.state_dict()))
    model.train()
    renderer = None
    timings, history, phases = [], [], []
    try:
        if args.load_batch is not None:
            saved = torch.load(args.load_batch, map_location="cpu")
            required = {"roi_img", "roi_cls", "roi_xyz", "roi_mask_visib"}
            if not required.issubset(saved) or saved["roi_img"].shape[0] != args.batch_size:
                raise ValueError("Saved EXP022 batch keys or batch size mismatch")
            if saved.get("dataset_key") != context.key or tuple(saved.get("object_ids", ())) != context.object_ids:
                if context.key != "lmo" or "dataset_key" in saved:
                    raise ValueError("Saved EXP022 batch dataset/object order mismatch")
            batch = {key: saved[key].to(args.device) for key in required}
        else:
            register_datasets_in_cfg(cfg)
            metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            data_ref = ref.__dict__[metadata.ref_key]
            renderer = get_renderer(cfg, data_ref, obj_names=metadata.objs,
                                    gpu_id=torch.device(args.device).index or 0)
            iterator = iter(build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN))
            raw = next(iterator)
            batch = batch_data(cfg, raw, renderer=renderer, device=args.device, phase="train")
            if args.save_batch is not None:
                if args.save_batch.exists():
                    raise FileExistsError(args.save_batch)
                args.save_batch.parent.mkdir(parents=True, exist_ok=True)
                serializable = {key: batch[key].detach().cpu() for key in
                                ("roi_img", "roi_cls", "roi_xyz", "roi_mask_visib")}
                serializable.update(dataset_key=context.key, object_ids=context.object_ids)
                torch.save(serializable, args.save_batch)
        image, classes = batch["roi_img"], batch["roi_cls"]
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=True):
                output_stats, losses = model(image, roi_classes=classes, gt_xyz=batch["roi_xyz"],
                                             gt_mask_visib=batch["roi_mask_visib"], do_loss=True,
                                             collect_pcc_diagnostics=args.diagnostics == "on")
                total = sum(losses.values())
            torch.cuda.synchronize()
            forward_end = time.perf_counter()
            if not torch.isfinite(total):
                raise RuntimeError("Non-finite EXP022 smoke loss")
            scaler.scale(total).backward()
            torch.cuda.synchronize()
            backward_end = time.perf_counter()
            scaler.unscale_(optimizer)
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad):
                raise RuntimeError("Non-finite EXP022 smoke gradient")
            torch.cuda.synchronize()
            audit_end = time.perf_counter()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                raise RuntimeError("EXP022 AMP skipped an optimizer step")
            torch.cuda.synchronize()
            finished = time.perf_counter()
            timings.append((finished - started) * 1000)
            phases.append({
                "forward_ms": (forward_end - started) * 1000,
                "backward_ms": (backward_end - forward_end) * 1000,
                "unscale_and_gradient_audit_ms": (audit_end - backward_end) * 1000,
                "optimizer_ms": (finished - audit_end) * 1000,
            })
            history.append({name: float(value.detach()) for name, value in losses.items()})
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            output = model(image, roi_classes=classes, return_pcc_debug=True)
        if not torch.isfinite(output["xyz_norm"]).all():
            raise RuntimeError("Non-finite EXP022 inference XYZ")
        if output["residual_norm"].max() > 1.00001:
            raise RuntimeError("EXP022 residual bound violated")
        print(json.dumps({"status": "PASS", "config": str(config_path), "renderer": args.renderer,
                          "dataset": context.key, "num_objects": context.num_objects,
                          "object_ids": context.object_ids,
                          "hierarchy_path": str(context.hierarchy_path),
                          "batch_source": "saved" if args.load_batch else "online",
                          "batch_size": args.batch_size, "steps": args.steps, "seed": args.seed,
                          "diagnostics": args.diagnostics,
                          "backbone_tensors": loaded,
                          "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                          "total_parameters": sum(p.numel() for p in model.parameters()),
                          "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
                          "class_histogram": torch.bincount(classes, minlength=context.num_objects).cpu().tolist(),
                          "symmetric_instances": int((model.pcc_head.symmetry_counts[classes] > 1).sum()),
                          "losses": history, "step_median_ms": statistics.median(timings),
                          "warmup_steps": warmup_steps,
                          "measured_step_median_ms": statistics.median(timings[warmup_steps:]),
                          "step_times_ms": timings,
                          "phase_times_ms": phases,
                          "measured_phase_medians_ms": {
                              key: statistics.median(row[key] for row in phases[warmup_steps:])
                              for key in phases[0]
                          },
                          "amp_skipped_steps": 0,
                          "train_stats_last": {name: float(value.detach())
                                               for name, value in output_stats["_train_stats"].items()},
                          "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                          "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}, indent=2))
    finally:
        if hasattr(renderer, "close"):
            renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
