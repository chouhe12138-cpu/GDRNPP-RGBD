#!/usr/bin/env python3
"""Run the leakage-marked frozen GDRNPP calibration on LM-PBR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / ".local" / "bop_toolkit"))
sys.path.insert(0, str(PROJECT_ROOT / ".local" / "bop_renderer" / "build"))

import cv2
import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation.evaluator import inference_context
from torch.cuda.amp import autocast

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
from core.gdrn_modeling.engine.engine_utils import batch_data, get_out_coor, get_out_mask
from core.gdrn_modeling.models import GDRN_double_mask
from core.utils.my_checkpoint import MyCheckpointer
from research.oracle_diagnostic.oracle_utils import (
    build_correspondences_from_dense,
    depth_to_object_coordinates,
    mask_metrics,
    normalized_image_points,
    normalized_xyz_to_metric,
    paired_cluster_bootstrap,
    prediction_valid_mask,
    region_confidence,
    safe_spearman,
    sample_nearest,
    subset_top_fraction,
)
from research.oracle_diagnostic.run_oracle_diagnostic import (
    EXPECTED_WEIGHT_HASH,
    failed_solution,
    oracle_selectors,
    ransac,
    sha256,
    solution_metrics,
    timed_solution,
)
from research.pbr_validation.pbr_dataset import (
    build_validation_records,
    register_validation_records,
)
from research.pose_aggregation.run_diagnostic import (
    aggregate,
    configure,
    load_model_points,
    load_symmetry_rotations,
    write_csv,
)
from research.pose_aggregation.solvers import reliable_subset, solution_from_pose


EXPERIMENT_ID = "EXP-20260731-003-pbr-validation-calibration"
DATASET_NAME = "lmo_pbr_stage3_calibration"
METHODS = (
    "patch_pnp",
    "pred_vis_ransac",
    "pred_full_ransac",
    "pred_inter_gt_vis_ransac",
    "pred_gt_vis_ransac",
    "gt_gt_vis_ransac",
    "current_top50_ransac",
    "oracle_xyz_top50_ransac",
    "oracle_best_pose",
    "oracle_best_axis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--pbr-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bbox-source", choices=("gt",), default="gt")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--seed", default=20260730, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--images-per-scene", default=100, type=int)
    parser.add_argument("--smoke-per-object", default=0, type=int)
    parser.add_argument("--bootstrap-iterations", default=2000, type=int)
    return parser.parse_args()


def select_records(records: List[dict], images_per_scene: int) -> List[dict]:
    if images_per_scene <= 0:
        return records
    counts: Dict[int, int] = {}
    selected = []
    for record in records:
        scene_id = int(record["scene_im_id"].split("/")[0])
        if counts.get(scene_id, 0) >= images_per_scene:
            continue
        counts[scene_id] = counts.get(scene_id, 0) + 1
        selected.append(record)
    if set(counts) != {12, 13, 14} or any(value != images_per_scene for value in counts.values()):
        raise RuntimeError(f"incomplete scene selection: {counts}")
    return selected


def build_lookups(records: List[dict]) -> tuple:
    images = {record["scene_im_id"]: record for record in records}
    annotations = {}
    for record in records:
        for index, annotation in enumerate(record["annotations"]):
            annotations[(record["scene_im_id"], index)] = annotation
    return images, annotations


def visibility_bin(value: float) -> str:
    if value < 0.1:
        return "lt_0.1"
    if value < 0.3:
        return "0.1_to_0.3"
    if value < 0.5:
        return "0.3_to_0.5"
    return "ge_0.5"


def correspondence_statistics(correspondences, prefix: str) -> dict:
    result = {
        f"{prefix}_points": correspondences.size,
        f"{prefix}_image_eigen_ratio": float("nan"),
        f"{prefix}_model_eigen_ratio": float("nan"),
    }
    if correspondences.size >= 3:
        eigen = np.linalg.eigvalsh(np.cov(correspondences.image_points, rowvar=False))
        result[f"{prefix}_image_eigen_ratio"] = float(eigen[0] / max(eigen[-1], 1e-12))
    if correspondences.size >= 4:
        eigen = np.linalg.eigvalsh(np.cov(correspondences.model_points, rowvar=False))
        result[f"{prefix}_model_eigen_ratio"] = float(eigen[0] / max(eigen[-1], 1e-12))
    return result


def object_nonnegative(per_object: List[dict], method: str, baseline: str) -> int:
    method_rows = {item["obj_id"]: item for item in per_object if item["method"] == method}
    baseline_rows = {item["obj_id"]: item for item in per_object if item["method"] == baseline}
    return sum(
        method_rows[obj_id]["add_s_0.1d_recall"] >= baseline_rows[obj_id]["add_s_0.1d_recall"]
        for obj_id in method_rows
    )


def compare(
    rows: List[dict],
    per_method: List[dict],
    per_object: List[dict],
    method: str,
    baseline: str,
    iterations: int,
    seed: int,
) -> dict:
    summary = {item["method"]: item for item in per_method}
    method_rows = [row for row in rows if row["method"] == method]
    baseline_rows = [row for row in rows if row["method"] == baseline]
    bootstrap = paired_cluster_bootstrap(
        method_rows,
        baseline_rows,
        "add_s_0.1d",
        iterations=iterations,
        seed=seed,
    )
    return {
        "method": method,
        "baseline": baseline,
        "add_s_recall_delta": summary[method]["add_s_0.1d_recall"]
        - summary[baseline]["add_s_0.1d_recall"],
        "ci95_low": bootstrap["ci95_low"],
        "ci95_high": bootstrap["ci95_high"],
        "objects_nonnegative": object_nonnegative(per_object, method, baseline),
    }


def decide(rows: List[dict], per_method: List[dict], per_object: List[dict], iterations: int, seed: int) -> dict:
    summary = {item["method"]: item for item in per_method}
    better_aggregator = max(
        ("patch_pnp", "pred_vis_ransac"),
        key=lambda method: summary[method]["add_s_0.1d_recall"],
    )
    comparisons = {
        "mask": compare(rows, per_method, per_object, "pred_gt_vis_ransac", "pred_vis_ransac", iterations, seed),
        "xyz": compare(rows, per_method, per_object, "gt_gt_vis_ransac", "pred_gt_vis_ransac", iterations, seed + 1),
        "reliability": compare(
            rows,
            per_method,
            per_object,
            "oracle_xyz_top50_ransac",
            "pred_inter_gt_vis_ransac",
            iterations,
            seed + 2,
        ),
        "aggregation": compare(
            rows,
            per_method,
            per_object,
            "oracle_best_pose",
            better_aggregator,
            iterations,
            seed + 3,
        ),
        "axis": compare(
            rows,
            per_method,
            per_object,
            "oracle_best_axis",
            better_aggregator,
            iterations,
            seed + 4,
        ),
    }
    xyz_pattern = (
        comparisons["xyz"]["add_s_recall_delta"] >= 0.20
        and comparisons["xyz"]["ci95_low"] > 0
        and comparisons["xyz"]["objects_nonnegative"] == 8
    )
    mask_not_primary = comparisons["mask"]["add_s_recall_delta"] < 0.05
    reliability_not_sufficient = comparisons["reliability"]["add_s_recall_delta"] <= 0.0
    axis_complementary = (
        comparisons["axis"]["add_s_recall_delta"] >= 0.05
        and comparisons["axis"]["ci95_low"] > 0
        and comparisons["axis"]["objects_nonnegative"] >= 6
    )
    return {
        "status": "CALIBRATION_MATCH" if xyz_pattern and mask_not_primary else "CALIBRATION_MISMATCH",
        "not_formal_validation": True,
        "reason": "official checkpoint may have trained on validation scenes",
        "xyz_primary_pattern": xyz_pattern,
        "mask_not_primary_pattern": mask_not_primary,
        "scalar_reliability_not_sufficient_pattern": reliability_not_sufficient,
        "axis_complementary_pattern": axis_complementary,
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    for field in ("config_file", "weights", "pbr_root", "manifest", "models_root", "output_dir"):
        path = getattr(args, field)
        setattr(args, field, (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path)
    if sha256(args.weights) != EXPECTED_WEIGHT_HASH:
        raise RuntimeError("unexpected checkpoint SHA-256")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records, source_summary = build_validation_records(args.pbr_root, args.manifest, args.models_root)
    records = select_records(all_records, args.images_per_scene)
    register_validation_records(args.dataset, records)
    images, gt_lookup = build_lookups(records)
    expected_targets = sum(len(record["annotations"]) for record in records)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    cfg = configure(args)
    metadata = MetadataCatalog.get(args.dataset)
    data_ref = ref.__dict__[metadata.ref_key]
    object_ids = [data_ref.obj2id[name] for name in metadata.objs]
    models = load_model_points(data_ref)
    symmetries = load_symmetry_rotations(data_ref)
    diameters = {obj_id: float(data_ref.diameters[index]) for index, obj_id in enumerate(object_ids)}
    model, _ = GDRN_double_mask.build_model_optimizer(cfg, is_test=True)
    MyCheckpointer(model, save_dir=str(args.output_dir), prefix_to_remove="_module.").resume_or_load(
        str(args.weights), resume=False
    )
    model.eval()
    loader = build_gdrn_test_loader(cfg, args.dataset, train_objs=metadata.objs, batch_size=1)

    rows: List[dict] = []
    dense_rows: List[dict] = []
    smoke_counts = {obj_id: 0 for obj_id in object_ids}
    processed_targets = 0
    inference_seconds = 0.0
    threshold = float(cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST)
    stop = False
    cached_scene_im_id = None
    cached_depth = None

    with inference_context(model), torch.no_grad():
        for inputs in loader:
            if not isinstance(inputs, list):
                inputs = [inputs]
            batch = batch_data(cfg, inputs, device=args.device, phase="test")
            start = time.perf_counter()
            with autocast(enabled=cfg.TEST.AMP_TEST):
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
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start
            xyz_batch = get_out_coor(
                cfg, output["coor_x"].detach(), output["coor_y"].detach(), output["coor_z"].detach()
            ).cpu().numpy()
            visible_batch = get_out_mask(cfg, output["mask"].detach()).cpu().numpy()
            full_batch = get_out_mask(cfg, output["full_mask"].detach()).cpu().numpy()
            region_batch = output["region"].detach().cpu().numpy()
            rotations = output["rot"].detach().cpu().numpy()
            translations = output["trans"].detach().cpu().numpy()

            flat_index = -1
            for input_item in inputs:
                for local_index in range(len(input_item["roi_img"])):
                    flat_index += 1
                    class_index = int(input_item["roi_cls"][local_index])
                    obj_id = int(object_ids[class_index])
                    if args.smoke_per_object and smoke_counts[obj_id] >= args.smoke_per_object:
                        continue
                    scene_im_id = input_item["scene_im_id"][local_index]
                    scene_id, im_id = map(int, scene_im_id.split("/"))
                    instance_id = int(input_item["inst_id"][local_index])
                    gt = gt_lookup[(scene_im_id, instance_id)]
                    image_record = images[scene_im_id]
                    rotation_gt = np.asarray(gt["pose"][:, :3], dtype=np.float64)
                    translation_gt = np.asarray(gt["pose"][:, 3], dtype=np.float64)
                    visibility = float(gt["visib_fract"])
                    camera = input_item["cam"][local_index].cpu().numpy()
                    height = int(input_item["im_H"][local_index])
                    width = int(input_item["im_W"][local_index])
                    extent = input_item["roi_extent"][local_index].cpu().numpy()
                    coord_norm = input_item["roi_coord_2d"][local_index].cpu().numpy().transpose(1, 2, 0)
                    image_points = normalized_image_points(coord_norm, height, width)

                    if cached_scene_im_id != scene_im_id:
                        raw_depth = cv2.imread(image_record["depth_file"], cv2.IMREAD_UNCHANGED)
                        if raw_depth is None:
                            raise FileNotFoundError(image_record["depth_file"])
                        cached_depth = raw_depth.astype(np.float64) / float(image_record["depth_factor"])
                        cached_scene_im_id = scene_im_id
                    gt_visible_full = cv2.imread(gt["mask_visib_file"], cv2.IMREAD_GRAYSCALE)
                    gt_full_full = cv2.imread(gt["mask_full_file"], cv2.IMREAD_GRAYSCALE)
                    if gt_visible_full is None or gt_full_full is None:
                        raise FileNotFoundError("missing PBR mask")
                    gt_visible_sampled, in_image = sample_nearest(gt_visible_full > 0, image_points)
                    gt_full_sampled, _ = sample_nearest(gt_full_full > 0, image_points)
                    gt_xyz_m, depth_valid = depth_to_object_coordinates(
                        cached_depth, image_points, camera, rotation_gt, translation_gt
                    )
                    gt_visible = gt_visible_sampled.astype(bool) & in_image & depth_valid

                    pred_xyz_norm = xyz_batch[flat_index].transpose(1, 2, 0)
                    pred_xyz_m = normalized_xyz_to_metric(pred_xyz_norm, extent)
                    pred_visible_probability = visible_batch[flat_index, 0]
                    pred_full_probability = full_batch[flat_index, 0]
                    pred_visible_support = prediction_valid_mask(
                        pred_xyz_norm, pred_visible_probability, extent, threshold
                    )
                    pred_full_support = prediction_valid_mask(
                        pred_xyz_norm, pred_full_probability, extent, threshold
                    )
                    shared_support = pred_visible_support & gt_visible
                    reliability_map = np.clip(pred_visible_probability, 0, 1) * np.clip(
                        region_confidence(region_batch[flat_index]), 0, 1
                    )
                    xyz_error_map = np.linalg.norm(pred_xyz_m - gt_xyz_m, axis=2)

                    pred_vis_corr = build_correspondences_from_dense(
                        image_points, pred_xyz_m, pred_visible_support, reliability_map
                    )
                    pred_full_corr = build_correspondences_from_dense(
                        image_points, pred_xyz_m, pred_full_support, reliability_map
                    )
                    pred_shared_corr = build_correspondences_from_dense(
                        image_points, pred_xyz_m, shared_support, reliability_map
                    )
                    pred_gt_corr = build_correspondences_from_dense(
                        image_points, pred_xyz_m, gt_visible, reliability_map
                    )
                    gt_corr = build_correspondences_from_dense(
                        image_points, gt_xyz_m, gt_visible, reliability_map
                    )
                    current_top50 = reliable_subset(pred_vis_corr)
                    oracle_top50 = (
                        pred_shared_corr
                        if pred_shared_corr.size < 32
                        else subset_top_fraction(
                            pred_shared_corr, xyz_error_map[shared_support], 0.5, largest=False
                        )
                    )
                    seed = args.seed + processed_targets
                    solutions = {
                        "patch_pnp": timed_solution(
                            lambda: solution_from_pose(
                                rotations[flat_index], translations[flat_index], pred_vis_corr, camera
                            )
                        ),
                        "pred_vis_ransac": ransac(pred_vis_corr, camera, seed),
                        "pred_full_ransac": ransac(pred_full_corr, camera, seed),
                        "pred_inter_gt_vis_ransac": ransac(pred_shared_corr, camera, seed),
                        "pred_gt_vis_ransac": ransac(pred_gt_corr, camera, seed),
                        "gt_gt_vis_ransac": ransac(gt_corr, camera, seed),
                        "current_top50_ransac": ransac(current_top50, camera, seed),
                        "oracle_xyz_top50_ransac": ransac(oracle_top50, camera, seed),
                    }
                    patch_metrics = solution_metrics(
                        solutions["patch_pnp"], rotation_gt, translation_gt, models[obj_id], diameters[obj_id], obj_id, symmetries[obj_id]
                    )
                    geometric_metrics = solution_metrics(
                        solutions["pred_vis_ransac"], rotation_gt, translation_gt, models[obj_id], diameters[obj_id], obj_id, symmetries[obj_id]
                    )
                    best_pose, best_axis = oracle_selectors(
                        solutions["patch_pnp"], solutions["pred_vis_ransac"], patch_metrics, geometric_metrics, pred_vis_corr, camera
                    )
                    solutions["oracle_best_pose"] = best_pose
                    solutions["oracle_best_axis"] = best_axis

                    for method in METHODS:
                        solution = solutions.get(method, failed_solution("missing_method"))
                        metrics = solution_metrics(
                            solution, rotation_gt, translation_gt, models[obj_id], diameters[obj_id], obj_id, symmetries[obj_id]
                        )
                        rows.append(
                            {
                                "method": method,
                                "scene_id": scene_id,
                                "im_id": im_id,
                                "instance_id": instance_id,
                                "obj_id": obj_id,
                                "obj_name": data_ref.id2obj[obj_id],
                                "visibility": visibility,
                                "visibility_bin": visibility_bin(visibility),
                                "success": solution.success,
                                "failure_reason": solution.failure_reason,
                                "num_points": solution.num_points,
                                "num_inliers": solution.num_inliers,
                                "inlier_ratio": solution.num_inliers / solution.num_points if solution.num_points else 0.0,
                                "median_reprojection_error_px": solution.median_reprojection_error,
                                "solver_time_ms": solution.solver_time_ms,
                                **metrics,
                            }
                        )

                    visible_metrics = mask_metrics(pred_visible_probability > threshold, gt_visible_sampled > 0)
                    full_metrics = mask_metrics(pred_full_probability > threshold, gt_full_sampled > 0)
                    xyz_errors = xyz_error_map[shared_support]
                    axis_errors = np.abs(pred_xyz_m - gt_xyz_m)[shared_support]
                    eroded = cv2.erode(gt_visible.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                    boundary = gt_visible & ~eroded & pred_visible_support
                    interior = eroded & pred_visible_support
                    dense_rows.append(
                        {
                            "scene_id": scene_id,
                            "im_id": im_id,
                            "instance_id": instance_id,
                            "obj_id": obj_id,
                            "obj_name": data_ref.id2obj[obj_id],
                            "visibility": visibility,
                            "visible_mask_iou": visible_metrics["iou"],
                            "full_mask_iou": full_metrics["iou"],
                            "xyz_median_error_m": float(np.median(xyz_errors)) if len(xyz_errors) else float("nan"),
                            "xyz_p90_error_m": float(np.quantile(xyz_errors, 0.9)) if len(xyz_errors) else float("nan"),
                            "xyz_median_error_over_diameter": float(np.median(xyz_errors) / diameters[obj_id]) if len(xyz_errors) else float("nan"),
                            "x_abs_median_error_m": float(np.median(axis_errors[:, 0])) if len(axis_errors) else float("nan"),
                            "y_abs_median_error_m": float(np.median(axis_errors[:, 1])) if len(axis_errors) else float("nan"),
                            "z_abs_median_error_m": float(np.median(axis_errors[:, 2])) if len(axis_errors) else float("nan"),
                            "boundary_xyz_median_error_m": float(np.median(xyz_error_map[boundary])) if np.any(boundary) else float("nan"),
                            "interior_xyz_median_error_m": float(np.median(xyz_error_map[interior])) if np.any(interior) else float("nan"),
                            "reliability_vs_xyz_error_spearman": safe_spearman(reliability_map[shared_support], xyz_errors),
                            **correspondence_statistics(pred_vis_corr, "all"),
                            **correspondence_statistics(current_top50, "current_top50"),
                            **correspondence_statistics(oracle_top50, "oracle_top50"),
                        }
                    )
                    processed_targets += 1
                    smoke_counts[obj_id] += 1
                    if processed_targets % 250 == 0:
                        print(f"processed {processed_targets}/{expected_targets}", flush=True)
                    if args.smoke_per_object and all(value >= args.smoke_per_object for value in smoke_counts.values()):
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break

    if not args.smoke_per_object and processed_targets != expected_targets:
        raise RuntimeError(f"expected {expected_targets} targets, processed {processed_targets}")
    per_method = aggregate(rows, ("method",))
    per_object = aggregate(rows, ("method", "obj_id", "obj_name"))
    per_visibility = aggregate(rows, ("method", "visibility_bin"))
    conclusion = (
        decide(rows, per_method, per_object, args.bootstrap_iterations, args.seed)
        if not args.smoke_per_object
        else {"status": "SMOKE_ONLY", "not_formal_validation": True}
    )
    write_csv(args.output_dir / "per_instance.csv", rows)
    write_csv(args.output_dir / "per_object.csv", per_object)
    write_csv(args.output_dir / "visibility_bins.csv", per_visibility)
    write_csv(args.output_dir / "dense_quality.csv", dense_rows)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "official_checkpoint_training_leakage_warning": True,
        "formal_model_selection_result": False,
        "source_dataset": source_summary,
        "selected_images": len(records),
        "expected_targets": expected_targets,
        "processed_targets": processed_targets,
        "inference_seconds": inference_seconds,
        "methods": per_method,
        "conclusion": conclusion,
    }
    with (args.output_dir / "calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)
    print(json.dumps({"processed_targets": processed_targets, "conclusion": conclusion}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
