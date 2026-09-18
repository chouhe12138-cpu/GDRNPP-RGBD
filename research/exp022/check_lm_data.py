#!/usr/bin/env python3
"""Numeric checks on the LM real and lm_imgn training data.

The LM loaders follow conventions that a shape-only smoke cannot catch: poses in
metres, a fixed camera for the renders, masks derived from depth, and a colour
variant directory that has to map back to `benchvise`.  This script compares the
stored annotations against the images they point at, then renders one real batch
to confirm that pose, camera and CAD id still agree after the crop.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from mmcv import Config

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_train_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, get_renderer
from lib.utils.mask_utils import cocosegm2mask, mask2bbox_xywh
from research.exp022.dataset_context import resolve_dataset_context
from research.exp022.preflight import resolve_config_path
from research.run_contract import validate_research_run_config


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {"min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean())}


def check_split(name: str, limit: int) -> dict:
    """Compare each stored annotation against the image it points at."""
    records = DatasetCatalog.get(name)[:limit]
    if not records:
        raise RuntimeError(f"{name} produced no records")
    expected_cam = np.array(ref.lm_full.camera_matrix, dtype=np.float32)
    ref_objs = ref.lm_full
    t_norms, cam_errors, bbox_errors, diameter_errors, areas = [], [], [], [], []
    img_types, lengths = set(), set()
    for record in records:
        img_types.add(record["img_type"])
        lengths.add((record["height"], record["width"]))
        cam_errors.append(float(np.abs(np.asarray(record["cam"]) - expected_cam).max()))
        for inst in record["annotations"]:
            t_norms.append(float(np.linalg.norm(inst["trans"])))
            mask = cocosegm2mask(inst["segmentation"], record["height"], record["width"])
            areas.append(float(mask.sum()))
            x, y, w, h = inst["bbox"]
            mx, my, mw, mh = mask2bbox_xywh(mask)
            bbox_errors.append(float(max(abs(x - mx), abs(y - my), abs(w - mw), abs(h - mh))))
            corners = np.asarray(inst["bbox3d_and_center"])[:8]
            # bbox3d_and_center is built from the same mesh, so its side lengths
            # must equal models_info's millimetre sizes in metres exactly
            size = np.array([inst["model_info"][f"size_{axis}"] for axis in "xyz"], dtype=np.float64)
            expected = size * float(ref_objs.vertex_scale)
            actual = corners.max(axis=0) - corners.min(axis=0)
            diameter_errors.append(float((np.abs(actual - expected) / expected).max()))
    sym_infos = getattr(MetadataCatalog.get(name), "sym_infos", {})
    return {
        "records": len(records),
        "img_types": sorted(img_types),
        "image_hw": sorted(lengths),
        "translation_norm_m": _stats(t_norms),
        "camera_max_abs_error_px": float(max(cam_errors)),
        "segmentation_area_px": _stats(areas),
        "bbox_vs_mask_max_error_px": float(max(bbox_errors)),
        "bbox3d_extent_vs_models_info_rel_error": float(max(diameter_errors)),
        "symmetric_classes": [int(label) for label, info in sorted(sym_infos.items())
                              if info is not None],
    }


def check_online_render(cfg, device: str, renderer_name: str, batch_size: int) -> dict:
    """Render one real batch and measure how well it agrees with the source masks."""
    cfg.MODEL.DEVICE = device
    cfg.MODEL.POSE_NET.XYZ_RENDERER = renderer_name
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.PERSISTENT_WORKERS = False
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.REFERENCE_BS = batch_size
    register_datasets_in_cfg(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    data_ref = ref.__dict__[metadata.ref_key]
    renderer = get_renderer(cfg, data_ref, obj_names=metadata.objs,
                            gpu_id=torch.device(device).index or 0)
    iterator = iter(build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN))
    raw = next(iterator)
    batch = batch_data(cfg, raw, renderer=renderer, device=device, phase="train")
    # The PCC batch path intersects the warped source mask with the render and
    # does not keep the render mask itself, so recover both sides from the pair
    # (mapper output, batch).  A mis-scaled pose or camera drives the overlap to
    # zero while both masks stay non-empty, which is the failure this catches.
    source = torch.stack([item["roi_mask_visib"] for item in raw], dim=0).to(device)
    covered = batch["roi_mask_visib"]
    source_area = source.reshape(source.shape[0], -1).sum(dim=1)
    covered_area = covered.reshape(covered.shape[0], -1).sum(dim=1)
    xyz = batch["roi_xyz"]
    extent = batch["roi_extent"].view(xyz.shape[0], 3, 1, 1)
    de_normalized = (xyz - 0.5) * extent
    return {
        "batch_size": int(xyz.shape[0]),
        "classes": [int(value) for value in batch["roi_cls"]],
        "source_mask_area_px": [float(value) for value in source_area],
        "source_mask_covered_by_render": [float(a / b) if b > 0 else None
                                          for a, b in zip(covered_area, source_area)],
        "empty_source_masks": int((source_area <= 0).sum()),
        "xyz_normalized": _stats(xyz.cpu().numpy()),
        "xyz_de_normalized_m": _stats(de_normalized.cpu().numpy()),
        "extent_m": [float(value) for value in batch["roi_extent"].flatten()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/gdrn/research/exp022_progressive_pcc/train_lm13_gdrn.py"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", choices=("cpp", "egl"), default="cpp")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config_path = resolve_config_path(args.config)
    cfg = Config.fromfile(str(config_path))
    mode = "smoke" if config_path.stem.startswith("smoke") else (
        "prepare" if cfg.get("RESEARCH_PROTOCOL", {}).get("SCHEDULE") == "configurable" else "formal"
    )
    validate_research_run_config(cfg, mode=mode, expected_experiment_id=cfg.EXPERIMENT_ID)
    context = resolve_dataset_context(cfg)
    register_datasets_in_cfg(cfg)

    report = {"config": str(config_path), "dataset": context.key,
              "splits": {str(name): check_split(str(name), args.limit)
                         for name in cfg.DATASETS.TRAIN}}
    if not args.skip_render:
        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("Rendering the online batch requires CUDA")
        report["online_render"] = check_online_render(
            cfg, args.device, args.renderer, args.batch_size)
    print(json.dumps({"status": "COMPLETE", "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
