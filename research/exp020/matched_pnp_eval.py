#!/usr/bin/env python3
"""EXP020 matched classical PnP/RANSAC evaluator (never EPro, never alpha).

This is the primary downstream evaluation for the EXP020 A/B correspondence
supervision: it consumes the trained correspondence producer with the EXP019-
verified *matched classical PnP/RANSAC* solver and reports how well each arm's
predicted XYZ is usable by an ordinary PnP solver.

Protocol (BLOCKER 2 of the review fix): the two arms A and B are different
checkpoints, so the evaluation support must NOT be rebuilt from each arm's own
predicted mask/XYZ validity (the geometry-head trunk is shared and the new
reprojection gradient can change mask/region/XYZ together).  Instead one
``reference`` checkpoint (default: the official checkpoint) generates

    S_fixed = reference_pred_visible INTERSECT gt_visible INTERSECT valid_depth

once per target, and that support mask, its flat indices, the subsample
indices, the 2D points, the camera matrix K, the RANSAC seed / threshold /
iterations are frozen.  A and B only supply their own predicted XYZ decoded on
those exact fixed indices.  A native-support result is allowed only as an
explicitly labelled secondary analysis.

EXP019 helpers are reused verbatim where possible (``roi2d_norm_to_pixels``,
``xyz_norm_to_metric``, ``historical_gt_reprojection_errors`` and
``solve_ransac_pnp``); the EXP019 evaluator, its EPro worker and its alpha
sweep are never imported or started here.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import inference_context
from mmcv import Config

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import (
    batch_data,
    get_out_coor,
    get_out_mask,
)
from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from core.utils.my_checkpoint import MyCheckpointer
from lib.utils.mask_utils import cocosegm2mask

from research.diagnostics.exp019_epro.correspondence import (
    Correspondences,
    historical_gt_reprojection_errors,
    project_points,
    roi2d_norm_to_pixels,
    xyz_norm_to_metric,
)
from research.diagnostics.exp019_epro.ransac_solver import solve_ransac_pnp
from research.diagnostics.exp019_epro.repo_adapter import (
    _configure as _exp019_configure,
    _dataset_lookups,
    _depth_to_object,
    _prediction_valid_mask,
    _sample_nearest,
    _target_counts,
)
from research.diagnostics.exp019_epro.types import PoseResult, failed_pose

ROOT = Path(__file__).resolve().parents[2]

# Fixed RANSAC protocol identical to the EXP019 matched solver (EPNP, 3 px,
# 100 iterations, confidence 0.99, seed 20260730 base).
SEED_BASE = 20260730
RANSAC_REPROJ_ERR_PX = 3.0
RANSAC_CONFIDENCE = 0.99
RANSAC_ITERATIONS = 100
MIN_FIXED_SUPPORT = 6
DEFAULT_REFERENCE = ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth"
DEFAULT_CONFIG = ROOT / "configs/gdrn/lmo_pbr/research/_base_/lmo_gt_eval.py"
ARMS = ("a", "b")


@dataclass
class FixedPlan:
    """Frozen Stage-1 support for one LM-O target (never per-arm)."""

    target_id: str
    scene_id: int
    im_id: int
    obj_id: int
    instance_id: int
    map_h: int
    map_w: int
    flat_indices: np.ndarray  # full S_fixed flat indices into the [H,W] map
    selected: np.ndarray  # indices INTO flat_indices used by the solver
    xy2d_px: np.ndarray  # [S,2] full-image 2D points of the selected set
    xyz_gt_m: np.ndarray  # [S,3] GT object-metric XYZ of the selected set
    K: np.ndarray  # [3,3] full-image camera matrix
    extent_m: np.ndarray  # [3] object extents (metric)
    R_gt: np.ndarray  # [3,3] GT object-to-camera rotation
    t_gt: np.ndarray  # [3] GT object-to-camera translation

    @property
    def num_fixed_support(self) -> int:
        return int(self.flat_indices.size)

    @property
    def num_selected(self) -> int:
        return int(self.selected.size)


def _to_bool_mask(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=bool)


def build_fixed_plan(
    *,
    target_id: str,
    scene_id: int,
    im_id: int,
    obj_id: int,
    instance_id: int,
    support_mask: np.ndarray,
    roi2d_px_map: np.ndarray,
    xyz_gt_m: np.ndarray,
    K: np.ndarray,
    extent_m: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    max_correspondences: int = 0,
    min_support: int = MIN_FIXED_SUPPORT,
) -> FixedPlan:
    """Freeze S_fixed and all solver inputs into an immutable FixedPlan."""
    support = _to_bool_mask(support_mask)
    if support.ndim != 2:
        raise ValueError(f"support_mask must be [H,W], got {support.shape}")
    if roi2d_px_map.shape[:2] != support.shape or xyz_gt_m.shape[:2] != support.shape:
        raise ValueError("support/roi2d_px_map/xyz_gt_m spatial shape mismatch")
    valid = (
        support
        & np.isfinite(roi2d_px_map).all(axis=2)
        & np.isfinite(xyz_gt_m).all(axis=2)
    )
    map_h, map_w = support.shape
    flat = np.flatnonzero(valid.reshape(-1))
    if flat.size >= int(min_support) and max_correspondences and flat.size > int(max_correspondences):
        choose = np.linspace(0, flat.size - 1, int(max_correspondences), dtype=np.int64)
        selected = np.asarray(choose, dtype=np.int64)
    else:
        selected = np.arange(flat.size, dtype=np.int64)
    sel_flat = flat[selected] if selected.size else np.zeros(0, dtype=np.int64)
    roi_px_flat = roi2d_px_map.reshape(-1, 2)[sel_flat]
    xyz_gt_flat = xyz_gt_m.reshape(-1, 3)[sel_flat]
    return FixedPlan(
        target_id=target_id,
        scene_id=int(scene_id),
        im_id=int(im_id),
        obj_id=int(obj_id),
        instance_id=int(instance_id),
        map_h=int(map_h),
        map_w=int(map_w),
        flat_indices=np.ascontiguousarray(flat, dtype=np.int64),
        selected=np.ascontiguousarray(selected, dtype=np.int64),
        xy2d_px=np.ascontiguousarray(roi_px_flat, dtype=np.float64),
        xyz_gt_m=np.ascontiguousarray(xyz_gt_flat, dtype=np.float64),
        K=np.ascontiguousarray(K, dtype=np.float64),
        extent_m=np.ascontiguousarray(extent_m, dtype=np.float64),
        R_gt=np.ascontiguousarray(R_gt, dtype=np.float64),
        t_gt=np.ascontiguousarray(t_gt, dtype=np.float64),
    )


def _pose_json(pose: PoseResult) -> dict:
    pose.validate()
    return {
        "success": bool(pose.success),
        "solver": pose.solver,
        "message": pose.message,
        "num_points": int(pose.num_points),
        "R": np.asarray(pose.R, dtype=float).reshape(3, 3).tolist(),
        "t": np.asarray(pose.t, dtype=float).reshape(3).tolist(),
    }


def _decode_selected(plan: FixedPlan, xyz_norm: np.ndarray) -> np.ndarray:
    """Predicted metric XYZ on the frozen selected flat indices only."""
    if plan.num_selected == 0:
        return np.zeros((0, 3), dtype=np.float64)
    xyz_m = xyz_norm_to_metric(xyz_norm, plan.extent_m)  # [3,H,W]
    flat = plan.flat_indices[plan.selected]
    return np.ascontiguousarray(xyz_m.reshape(3, -1).T[flat], dtype=np.float64)


def solve_arm_fixed(plan: FixedPlan, xyz_norm: np.ndarray, seed: int) -> dict:
    """Matched RANSAC on the frozen support; A/B replace only their XYZ."""
    x3d = _decode_selected(plan, xyz_norm)
    corr_ok = bool(
        x3d.size and np.isfinite(x3d).all() and np.isfinite(plan.xy2d_px).all()
    )
    if plan.num_selected < MIN_FIXED_SUPPORT:
        pose = failed_pose(
            "ransac_epnp",
            "fewer_than_min_fixed_support",
            plan.num_selected,
        )
    elif not corr_ok:
        pose = failed_pose("ransac_epnp", "nonfinite_fixed_support_xyz", plan.num_selected)
    else:
        corr = Correspondences(
            x3d_m=x3d,
            x2d_px=plan.xy2d_px,
            flat_indices=np.ascontiguousarray(
                plan.flat_indices[plan.selected], dtype=np.int64
            ),
            K=plan.K,
        )
        pose = solve_ransac_pnp(
            corr,
            seed=int(seed),
            reprojection_error_px=RANSAC_REPROJ_ERR_PX,
            confidence=RANSAC_CONFIDENCE,
            iterations_count=RANSAC_ITERATIONS,
        )
    # Producer-geometry diagnostics on the fixed support (never change the
    # solver input set): metric correspondence error and GT-pose reprojection.
    if corr_ok and x3d.shape[0]:
        with np.errstate(invalid="ignore", over="ignore"):
            corr_err = np.linalg.norm(x3d - plan.xyz_gt_m, axis=1) * 1000.0  # mm
            reproj = project_points(x3d, plan.R_gt, plan.t_gt, plan.K)
            reproj_err = np.linalg.norm(reproj - plan.xy2d_px, axis=1)
        corr_err_mm = float(np.nanmean(corr_err)) if np.isfinite(corr_err).all() else None
        reproj_err_px = float(np.nanmean(reproj_err)) if np.isfinite(reproj_err).all() else None
    else:
        corr_err_mm, reproj_err_px = None, None
    return {
        "pose": _pose_json(pose),
        "corr_err_mm": corr_err_mm,
        "reproj_err_px": reproj_err_px,
    }


def _load_model(cfg: Config, checkpoint: Path, device: str) -> torch.nn.Module:
    model_cfg = copy.deepcopy(cfg)
    model_cfg.MODEL.WEIGHTS = str(checkpoint)
    model_cfg.MODEL.DEVICE = device
    model, _optimizer = build_model_optimizer(model_cfg, is_test=True)
    MyCheckpointer(
        model,
        save_dir=str(Path(checkpoint).parent),
        prefix_to_remove="_module.",
    ).resume_or_load(str(checkpoint), resume=False)
    model.eval()
    return model


def _forward(model: torch.nn.Module, cfg: Config, batch: dict, device: str):
    with inference_context(model), torch.no_grad():
        output = model(
            batch["roi_img"],
            roi_classes=batch["roi_cls"],
            roi_cams=batch["roi_cam"],
            roi_whs=batch["roi_wh"],
            roi_centers=batch["roi_center"],
            resize_ratios=batch["resize_ratio"],
            roi_coord_2d=batch.get("roi_coord_2d"),
            roi_coord_2d_rel=batch.get("roi_coord_2d_rel"),
            roi_extents=batch["roi_extent"],
        )
    xyz = get_out_coor(
        cfg,
        output["coor_x"].detach(),
        output["coor_y"].detach(),
        output["coor_z"].detach(),
    ).float().cpu().numpy()
    vis = get_out_mask(cfg, output["mask"].detach()).float().cpu().numpy()
    return xyz, vis  # xyz [K,3,H,W], vis [K,1,H,W]


def _git_state():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    return commit, branch, dirty


def run_evaluation(
    *,
    reference_checkpoint: Path,
    checkpoint_a: Path,
    checkpoint_b: Path,
    gdrn_config: Path,
    device: str,
    output: Path,
    limit: int | None,
    max_correspondences: int = 0,
) -> dict:
    """Stage 1 (reference -> S_fixed) then Stage 2 (A/B on the frozen plan)."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit, branch, dirty = _git_state()

    cfg = _exp019_configure(str(gdrn_config), str(reference_checkpoint), device)
    cfg.OUTPUT_DIR = str(output)
    register_datasets_in_cfg(cfg)
    metadata = MetadataCatalog.get("lmo_bop_test")
    data_ref = ref.__dict__[metadata.ref_key]
    object_ids = [int(data_ref.obj2id[name]) for name in metadata.objs]
    images, annotations = _dataset_lookups("lmo_bop_test")
    target_counts = _target_counts()

    models = {
        "reference": _load_model(cfg, reference_checkpoint, device),
        "a": _load_model(cfg, checkpoint_a, device),
        "b": _load_model(cfg, checkpoint_b, device),
    }
    loader = build_gdrn_test_loader(
        cfg, "lmo_bop_test", train_objs=metadata.objs, batch_size=1
    )

    threshold = float(cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST)
    metadata_dict = {
        "status": "RUNNING",
        "experiment_id": "EXP-20260909-020-geometry-aware-correspondence-loss",
        "diagnostic_only": True,
        "training": False,
        "evaluation": "matched_classical_pnp_ransac",
        "dataset": "lmo_bop_test",
        "bbox_source": "gt",
        "fixed_support_source": "reference_pred_visible INTERSECT gt_visible INTERSECT valid_depth",
        "fixed_support_used_by": ["a", "b"],
        "native_support": False,
        "epro_started": False,
        "alpha_sweep": False,
        "ransac_seed_base": SEED_BASE,
        "ransac_reprojection_error_px": RANSAC_REPROJ_ERR_PX,
        "ransac_confidence": RANSAC_CONFIDENCE,
        "ransac_iterations": RANSAC_ITERATIONS,
        "min_fixed_support": MIN_FIXED_SUPPORT,
        "max_correspondences": int(max_correspondences),
        "reference_checkpoint": str(reference_checkpoint),
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "gdrn_config": str(gdrn_config),
        "device": device,
        "limit": limit,
        "source_commit": commit,
        "source_branch": branch,
        "source_tree_dirty": dirty,
    }
    metadata_path = output / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    np.random.seed(SEED_BASE)
    torch.manual_seed(SEED_BASE)

    started = time.perf_counter()
    processed = 0
    max_gt_reproj = 0.0
    depth_cache: dict[str, np.ndarray] = {}
    occurrences: dict[tuple, int] = {}
    selected_per_object = {obj_id: 0 for obj_id in object_ids}
    quota = math.ceil(limit / len(object_ids)) if limit is not None and limit <= 32 else None
    rows = []

    try:
        for inputs in loader:
            if not isinstance(inputs, list):
                inputs = [inputs]
            batch = batch_data(cfg, inputs, device=device, phase="test")
            xyz_ref, vis_ref = _forward(models["reference"], cfg, batch, device)
            xyz_a, vis_a = _forward(models["a"], cfg, batch, device)
            xyz_b, vis_b = _forward(models["b"], cfg, batch, device)

            flat_index = -1
            for input_item in inputs:
                for local_index in range(len(input_item["roi_img"])):
                    flat_index += 1
                    class_index = int(input_item["roi_cls"][local_index])
                    obj_id = object_ids[class_index]
                    scene_im_id = input_item["scene_im_id"][local_index]
                    target_key = (scene_im_id, obj_id)
                    occurrence = occurrences.get(target_key, 0)
                    if occurrence >= target_counts.get(target_key, 0):
                        continue
                    occurrences[target_key] = occurrence + 1
                    if quota is not None and selected_per_object[obj_id] >= quota:
                        continue
                    instance_id = int(input_item["inst_id"][local_index])
                    gt = annotations[(scene_im_id, instance_id)]
                    image = images[scene_im_id]
                    scene_id, im_id = map(int, scene_im_id.split("/"))
                    K = input_item["cam"][local_index].cpu().numpy().astype(np.float64)
                    height = int(input_item["im_H"][local_index])
                    width = int(input_item["im_W"][local_index])
                    extent = input_item["roi_extent"][local_index].cpu().numpy()
                    roi2d_norm = input_item["roi_coord_2d"][local_index].cpu().numpy()
                    roi2d_px = roi2d_norm_to_pixels(
                        roi2d_norm, np.array([height, width], dtype=np.float64)
                    )  # [2,H,W]
                    roi2d_px_map = np.moveaxis(roi2d_px, 0, -1)  # [H,W,2]
                    if scene_im_id not in depth_cache:
                        depth_path = (ROOT / image["depth_file"]).resolve()
                        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                        if raw_depth is None:
                            raise FileNotFoundError(depth_path)
                        depth_cache[scene_im_id] = raw_depth.astype(np.float64) / float(
                            image["depth_factor"]
                        )
                    visible_full = cocosegm2mask(gt["segmentation"], height, width).astype(bool)
                    visible_sampled, _in_image = _sample_nearest(
                        visible_full, roi2d_px_map
                    )
                    R_gt = np.asarray(gt["pose"][:, :3], dtype=np.float64)
                    t_gt = np.asarray(gt["pose"][:, 3], dtype=np.float64)
                    gt_xyz_m, depth_valid = _depth_to_object(
                        depth_cache[scene_im_id], roi2d_px_map, K, R_gt, t_gt
                    )
                    gt_visible = visible_sampled & _in_image & depth_valid

                    pred_hwc = xyz_ref[flat_index].transpose(1, 2, 0)
                    mask_prob = vis_ref[flat_index, 0]
                    pred_visible_ref = _prediction_valid_mask(
                        pred_hwc, mask_prob, extent, threshold
                    )
                    support = _to_bool_mask(pred_visible_ref) & gt_visible

                    # GT reprojection sanity on the reference geometry (same
                    # gate as EXP019): GT XYZ must project back < 0.5 px.
                    reproj_error = historical_gt_reprojection_errors(
                        gt_xyz_m, roi2d_px_map, gt_visible, R_gt, t_gt, K
                    )
                    if len(reproj_error):
                        max_gt_reproj = max(max_gt_reproj, float(reproj_error.max()))
                        if float(reproj_error.max()) >= 0.5:
                            raise RuntimeError(
                                f"{scene_im_id}/{instance_id}: GT XYZ reprojection "
                                f"{float(reproj_error.max()):.4f}px >= 0.5px"
                            )

                    target_id = (
                        f"{scene_id:06d}/{im_id:06d}/{obj_id:06d}/{instance_id:03d}"
                    )
                    plan = build_fixed_plan(
                        target_id=target_id,
                        scene_id=scene_id,
                        im_id=im_id,
                        obj_id=obj_id,
                        instance_id=instance_id,
                        support_mask=support,
                        roi2d_px_map=roi2d_px_map,
                        xyz_gt_m=gt_xyz_m,
                        K=K,
                        extent_m=extent,
                        R_gt=R_gt,
                        t_gt=t_gt,
                        max_correspondences=max_correspondences,
                    )
                    seed = SEED_BASE + processed
                    arm_a = solve_arm_fixed(plan, xyz_a[flat_index], seed=seed)
                    arm_b = solve_arm_fixed(plan, xyz_b[flat_index], seed=seed)
                    row = {
                        "target_id": target_id,
                        "scene_id": int(scene_id),
                        "im_id": int(im_id),
                        "obj_id": int(obj_id),
                        "instance_id": int(instance_id),
                        "num_fixed_support": int(plan.num_fixed_support),
                        "num_selected": int(plan.num_selected),
                        "ref_pred_visible_points": int(np.count_nonzero(pred_visible_ref)),
                        "gt_visible_points": int(np.count_nonzero(gt_visible)),
                        "gt_xyz_max_reprojection_px": float(max_gt_reproj),
                        "a": arm_a,
                        "b": arm_b,
                    }
                    rows.append(row)
                    processed += 1
                    selected_per_object[obj_id] += 1
                    print(f"EXP020 matched-PnP targets: {processed}", flush=True)
                    if limit is not None and processed >= limit:
                        break
            if limit is not None and processed >= limit:
                break
    except Exception as exc:  # noqa: BLE001
        metadata_dict.update(
            {
                "status": "FAILED",
                "num_targets": processed,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        metadata_path.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")
        raise
    finally:
        for model in models.values():
            model.cpu()

    expected = limit if limit else _full_target_total(target_counts)
    if limit is None and processed != expected:
        raise RuntimeError(f"processed {processed} targets, expected {expected}")
    metadata_dict.update(
        {
            "status": "COMPLETE",
            "num_targets": processed,
            "max_gt_xyz_reprojection_px": max_gt_reproj,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    metadata_path.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")

    with (output / "poses.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = aggregate(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETE", "num_targets": processed, "summary": summary}, indent=2))
    return metadata_dict


def _full_target_total(target_counts: dict) -> int:
    return int(sum(target_counts.values()))


def aggregate(rows: list[dict]) -> dict:
    """Internal (BOP-free) aggregation of the matched-PnP rows."""
    total = len(rows)
    out = {"num_targets": total}
    for arm in ARMS:
        solved = [row for row in rows if row[arm]["pose"]["success"]]
        corr = [row[arm]["corr_err_mm"] for row in rows if row[arm]["corr_err_mm"] is not None]
        reproj = [row[arm]["reproj_err_px"] for row in rows if row[arm]["reproj_err_px"] is not None]
        out[arm] = {
            "solve_success_rate": float(len(solved) / total) if total else float("nan"),
            "solve_successes": len(solved),
            "solve_failures": total - len(solved),
            "mean_corr_err_mm": float(np.mean(corr)) if corr else None,
            "mean_reproj_err_px": float(np.mean(reproj)) if reproj else None,
            "mean_num_support": float(np.mean([row["num_fixed_support"] for row in rows]))
            if total
            else float("nan"),
        }
    # Consistency between the two arms (used by the A==B smoke).
    deltas_r = []
    deltas_t = []
    deltas_corr = []
    deltas_reproj = []
    for row in rows:
        pa, pb = row["a"]["pose"], row["b"]["pose"]
        if pa["success"] and pb["success"]:
            deltas_r.append(
                float(np.max(np.abs(np.asarray(pa["R"]) - np.asarray(pb["R"]))))
            )
            deltas_t.append(
                float(np.max(np.abs(np.asarray(pa["t"]) - np.asarray(pb["t"]))))
            )
        ca, cb = row["a"]["corr_err_mm"], row["b"]["corr_err_mm"]
        ra, rb = row["a"]["reproj_err_px"], row["b"]["reproj_err_px"]
        if ca is not None and cb is not None:
            deltas_corr.append(abs(ca - cb))
        if ra is not None and rb is not None:
            deltas_reproj.append(abs(ra - rb))
    out["arm_consistency"] = {
        "max_abs_R_delta": float(max(deltas_r)) if deltas_r else None,
        "max_abs_t_delta": float(max(deltas_t)) if deltas_t else None,
        "max_abs_corr_err_delta_mm": float(max(deltas_corr)) if deltas_corr else None,
        "max_abs_reproj_err_delta_px": float(max(deltas_reproj)) if deltas_reproj else None,
    }
    return out


def export_bop_csv(rows: list[dict], result_dir: Path) -> list[str]:
    result_dir.mkdir(parents=True, exist_ok=False)
    fields = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    names = []
    for arm in ARMS:
        name = f"arm{arm.upper()}_lmo-test.csv"
        names.append(name)
        with (result_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                pose = row[arm]["pose"]
                if not pose["success"]:
                    continue
                writer.writerow(
                    {
                        "scene_id": int(row["scene_id"]),
                        "im_id": int(row["im_id"]),
                        "obj_id": int(row["obj_id"]),
                        "score": 1.0,
                        "R": " ".join(f"{float(v):.9g}" for v in np.asarray(pose["R"]).reshape(-1)),
                        "t": " ".join(
                            f"{float(v):.9g}" for v in np.asarray(pose["t"]) * 1000.0
                        ),
                        "time": -1.0,
                    }
                )
    return names


def _maybe_bop_eval(run_dir: Path) -> dict:
    """Optional BOP-AR/ADD(-S)/reS/teS aggregation (full runs only)."""
    run_dir = Path(run_dir).resolve()
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_dir / "poses.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    toolkit = ROOT / ".local" / "bop_toolkit"
    renderer = ROOT / ".local" / "bop_renderer" / "build"
    result_dir, eval_dir = run_dir / "bop_results", run_dir / "bop_eval"
    names = export_bop_csv(rows, result_dir)
    environment = os.environ.copy()
    pythonpath = [str(ROOT), str(toolkit), str(renderer)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "BOP_PATH": str((ROOT / "datasets" / "BOP_DATASETS").resolve()),
            "BOP_RESULTS_PATH": str(result_dir),
            "BOP_EVAL_PATH": str(eval_dir),
            "BOP_RENDERER_PATH": str(renderer),
            "BOP_NUM_WORKERS": "1",
        }
    )
    command = [
        sys.executable,
        str(ROOT / "lib" / "pysixd" / "scripts" / "eval_pose_results_more.py"),
        f"--results_path={result_dir}",
        f"--eval_path={eval_dir}",
        f"--result_filenames={','.join(names)}",
        "--renderer_type=cpp",
        "--error_types=mspd,mssd,vsd,ad,reS,teS",
        "--targets_filename=test_targets_bop19.json",
        "--n_top=1",
        "--dataset=lmo",
    ]
    subprocess.run(command, check=True, cwd=ROOT, env=environment)
    result = {}
    for arm in ARMS:
        stem = (run_dir / "bop_results" / f"arm{arm.upper()}_lmo-test.csv").stem
        root = eval_dir / stem
        bop = json.loads((root / "scores_bop19.json").read_text(encoding="utf-8"))
        add_files = list(root.glob("error=ad_ntop=*/scores_th=0.100_min-visib=-1.000.json"))
        add_files += list(root.glob("error:ad_ntop:*/scores_th:0.100_min-visib:-1.000.json"))
        if len(add_files) != 1:
            raise RuntimeError(f"Expected one ADD(-S) score under {root}, got {add_files}")
        add = json.loads(add_files[0].read_text(encoding="utf-8"))
        result[arm] = {
            "bop": float(bop["bop19_average_recall"]),
            "add": float(add["recall"]),
            "reS": float(bop["bop19_average_recall_reS"]),
            "teS": float(bop["bop19_average_recall_teS"]),
        }
    (run_dir / "bop_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--gdrn-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-correspondences", type=int, default=0)
    parser.add_argument(
        "--bop-eval",
        action="store_true",
        help="After a complete (limit=None) run, aggregate BOP AR/ADD(-S)/reS/teS",
    )
    args = parser.parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} requested but CUDA is unavailable")
    run_evaluation(
        reference_checkpoint=args.reference_checkpoint,
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        gdrn_config=args.gdrn_config,
        device=args.device,
        output=args.output,
        limit=args.limit,
        max_correspondences=args.max_correspondences,
    )
    if args.bop_eval:
        if args.limit is not None:
            raise RuntimeError("--bop-eval requires a complete run (no --limit)")
        print(json.dumps({"bop_metrics": _maybe_bop_eval(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
