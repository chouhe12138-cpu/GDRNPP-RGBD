#!/usr/bin/env python3
"""Run the frozen GDRNPP causal-oracle diagnostic on LM-O."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Dict, List, Tuple

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
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, get_out_coor, get_out_mask
from core.gdrn_modeling.models import GDRN_double_mask
from core.utils.my_checkpoint import MyCheckpointer
from lib.pysixd import inout
from lib.utils.mask_utils import cocosegm2mask
from research.oracle_diagnostic.oracle_utils import (
    build_correspondences_from_dense,
    dense_geometry_statistics,
    depth_to_object_coordinates,
    mask_metrics,
    normalized_image_points,
    normalized_xyz_to_metric,
    paired_cluster_bootstrap,
    prediction_valid_mask,
    region_confidence,
    reprojection_error_for_gt,
    safe_spearman,
    sample_nearest,
    subset_top_fraction,
)
from research.pose_aggregation.metrics import pose_metrics
from research.pose_aggregation.run_diagnostic import (
    EXPECTED_LMO_TARGETS,
    aggregate,
    configure,
    load_bop_target_counts,
    load_model_points,
    load_symmetry_rotations,
    write_csv,
)
from research.pose_aggregation.solvers import (
    Correspondences,
    PoseSolution,
    reliable_subset,
    solution_from_pose,
    solve_ransac_epnp,
)


EXPERIMENT_ID = "EXP-20260731-002-gdrnpp-causal-oracle"
EXPECTED_WEIGHT_HASH = "bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c"
METHODS = (
    "patch_pnp",
    "pred_vis_ransac",
    "pred_full_ransac",
    "pred_inter_gt_vis_ransac",
    "pred_gt_vis_ransac",
    "gt_inter_vis_ransac",
    "gt_gt_vis_ransac",
    "current_top50_ransac",
    "current_shared_top50_ransac",
    "oracle_xyz_top50_ransac",
    "oracle_best_pose",
    "oracle_best_axis",
)
BOP_METHOD_TOKENS = {
    "patch_pnp": "patchpnp",
    "pred_vis_ransac": "predvis",
    "pred_full_ransac": "predfull",
    "pred_inter_gt_vis_ransac": "predintergtvis",
    "pred_gt_vis_ransac": "predgtvis",
    "gt_inter_vis_ransac": "gtintervis",
    "gt_gt_vis_ransac": "gtgtvis",
    "current_top50_ransac": "currenttop50",
    "current_shared_top50_ransac": "currentsharedtop50",
    "oracle_xyz_top50_ransac": "oraclexyztop50",
    "oracle_best_pose": "oraclebestpose",
    "oracle_best_axis": "oraclebestaxis",
}
BOP_NAMES = {method: f"{BOP_METHOD_TOKENS[method]}_lmo-test.csv" for method in METHODS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dataset", default="lmo_bop_test")
    parser.add_argument("--bbox-source", choices=("gt",), default="gt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=20260730, type=int)
    parser.add_argument("--smoke-per-object", default=0, type=int)
    parser.add_argument("--max-targets", default=0, type=int)
    parser.add_argument("--bop-eval", action="store_true")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--bootstrap-iterations", default=10000, type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(args: argparse.Namespace) -> str:
    required = (
        args.config_file,
        args.weights,
        PROJECT_ROOT / "datasets" / "BOP_DATASETS" / "lmo" / "test_targets_bop19.json",
        PROJECT_ROOT / "datasets" / "BOP_DATASETS" / "lmo" / "models_eval",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} requested, but CUDA is unavailable")
    if args.smoke_per_object < 0 or args.max_targets < 0:
        raise ValueError("smoke-per-object and max-targets must be non-negative")
    if args.bop_eval and (args.smoke_per_object or args.max_targets):
        raise ValueError("BOP evaluation requires the complete 1445-target run")
    weight_hash = sha256(args.weights)
    if weight_hash != EXPECTED_WEIGHT_HASH:
        raise RuntimeError(f"Unexpected checkpoint SHA-256: {weight_hash}")
    return weight_hash


def build_dataset_lookups(dataset_name: str) -> Tuple[Dict[str, dict], Dict[tuple, dict]]:
    images: Dict[str, dict] = {}
    annotations: Dict[tuple, dict] = {}
    for image in DatasetCatalog.get(dataset_name):
        images[image["scene_im_id"]] = image
        for instance_index, annotation in enumerate(image.get("annotations", [])):
            annotations[(image["scene_im_id"], instance_index)] = annotation
    return images, annotations


def visibility_bin(value: float) -> str:
    if value < 0.1:
        return "lt_0.1"
    if value < 0.3:
        return "0.1_to_0.3"
    if value < 0.5:
        return "0.3_to_0.5"
    return "ge_0.5"


def timed_solution(function) -> PoseSolution:
    start = time.perf_counter()
    solution = function()
    return replace(solution, solver_time_ms=(time.perf_counter() - start) * 1000.0)


def ransac(correspondences: Correspondences, camera: np.ndarray, seed: int) -> PoseSolution:
    return timed_solution(lambda: solve_ransac_epnp(correspondences, camera, seed=seed))


def failed_solution(reason: str, num_points: int = 0) -> PoseSolution:
    return PoseSolution(
        success=False,
        rotation=np.full((3, 3), np.nan),
        translation=np.full(3, np.nan),
        num_points=num_points,
        num_inliers=0,
        median_reprojection_error=float("nan"),
        failure_reason=reason,
    )


def solution_metrics(
    solution: PoseSolution,
    rotation_gt: np.ndarray,
    translation_gt: np.ndarray,
    model_points: np.ndarray,
    diameter: float,
    obj_id: int,
    symmetries: np.ndarray,
) -> dict:
    if not solution.success:
        return {
            "rotation_error_deg": float("nan"),
            "translation_error_mm": float("nan"),
            "add_s_m": float("nan"),
            "add_s_0.1d": 0.0,
        }
    return pose_metrics(
        solution.rotation,
        solution.translation,
        rotation_gt,
        translation_gt,
        model_points,
        diameter,
        obj_id,
        symmetries,
    )


def oracle_selectors(
    patch: PoseSolution,
    geometric: PoseSolution,
    patch_metrics: dict,
    geometric_metrics: dict,
    correspondences: Correspondences,
    camera: np.ndarray,
) -> Tuple[PoseSolution, PoseSolution]:
    candidates = [
        (patch, patch_metrics),
        (geometric, geometric_metrics),
    ]
    successful = [(solution, metrics) for solution, metrics in candidates if solution.success]
    if not successful:
        failure = failed_solution("both_sources_failed", correspondences.size)
        return failure, failure
    best_pose = min(successful, key=lambda item: item[1]["add_s_m"])[0]
    best_rotation = min(successful, key=lambda item: item[1]["rotation_error_deg"])[0]
    best_translation = min(successful, key=lambda item: item[1]["translation_error_mm"])[0]
    best_axis = timed_solution(
        lambda: solution_from_pose(
            best_rotation.rotation,
            best_translation.translation,
            correspondences,
            camera,
        )
    )
    return replace(best_pose, solver_time_ms=0.0), best_axis


def save_bop_files(output_dir: Path, bop_rows: Dict[str, List[dict]]) -> None:
    result_dir = output_dir / "bop_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        inout.save_bop_results(result_dir / BOP_NAMES[method], bop_rows[method], version="bop19")


def run_bop_evaluation(output_dir: Path) -> Dict[str, dict]:
    toolkit = PROJECT_ROOT / ".local" / "bop_toolkit"
    result_dir = output_dir / "bop_results"
    eval_dir = output_dir / "bop_eval"
    environment = os.environ.copy()
    environment.update(
        {
            "BOP_PATH": str((PROJECT_ROOT / "datasets" / "BOP_DATASETS").resolve()),
            "BOP_RESULTS_PATH": str(result_dir.resolve()),
            "BOP_EVAL_PATH": str(eval_dir.resolve()),
            "BOP_RENDERER_PATH": str((PROJECT_ROOT / ".local" / "bop_renderer" / "build").resolve()),
            "BOP_NUM_WORKERS": "1",
        }
    )
    command = [
        sys.executable,
        str(toolkit / "scripts" / "eval_bop19_pose.py"),
        "--renderer_type=cpp",
        "--num_workers=1",
        "--results_path=" + str(result_dir.resolve()),
        "--eval_path=" + str(eval_dir.resolve()),
        "--result_filenames=" + ",".join(BOP_NAMES[method] for method in METHODS),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)
    scores = {}
    hashes = {}
    for method, filename in BOP_NAMES.items():
        score_path = eval_dir / Path(filename).stem / "scores_bop19.json"
        with score_path.open(encoding="utf-8") as handle:
            scores[method] = json.load(handle)
        hashes[filename] = sha256(result_dir / filename)
    with (eval_dir / "bop_result_sha256.json").open("w", encoding="utf-8") as handle:
        json.dump(hashes, handle, indent=2)
    return scores


def stage1_reproduction(per_method: List[dict]) -> dict:
    stage1 = PROJECT_ROOT / "output" / "EXP-20260730-001" / "full" / "summary.json"
    if not stage1.exists():
        return {"status": "NOT_AVAILABLE"}
    with stage1.open(encoding="utf-8") as handle:
        old = {item["method"]: item for item in json.load(handle)["methods"]}
    current = {item["method"]: item for item in per_method}
    mapping = {"patch_pnp": "patch_pnp", "pred_vis_ransac": "ransac_epnp"}
    comparisons = {}
    passed = True
    for current_name, old_name in mapping.items():
        add_delta = current[current_name]["add_s_0.1d_recall"] - old[old_name]["add_s_0.1d_recall"]
        bop_delta = current[current_name].get("bop_ar", float("nan")) - old[old_name].get(
            "bop_ar", float("nan")
        )
        comparisons[current_name] = {"add_delta": add_delta, "bop_ar_delta": bop_delta}
        passed &= abs(add_delta) <= 1e-12 and (not np.isfinite(bop_delta) or abs(bop_delta) <= 1e-12)
    return {"status": "PASS" if passed else "FAIL", "comparisons": comparisons}


def compare_factor(
    factor: str,
    method: str,
    baseline: str,
    rows: List[dict],
    per_method: List[dict],
    per_object: List[dict],
    full_oracle_gap: float,
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
    method_objects = {item["obj_id"]: item for item in per_object if item["method"] == method}
    baseline_objects = {item["obj_id"]: item for item in per_object if item["method"] == baseline}
    objects_nonnegative = sum(
        method_objects[obj_id]["add_s_0.1d_recall"]
        >= baseline_objects[obj_id]["add_s_0.1d_recall"]
        for obj_id in method_objects
    )
    add_delta = summary[method]["add_s_0.1d_recall"] - summary[baseline]["add_s_0.1d_recall"]
    bop_delta = summary[method]["bop_ar"] - summary[baseline]["bop_ar"]
    gap_closure = add_delta / full_oracle_gap if full_oracle_gap > 0 else float("nan")
    passed = (
        bop_delta >= 0.01
        and add_delta >= 0.05
        and bootstrap["ci95_low"] > 0.0
        and objects_nonnegative >= 6
        and gap_closure >= 0.30
    )
    return {
        "factor": factor,
        "method": method,
        "baseline": baseline,
        "bop_ar_delta": bop_delta,
        "add_s_recall_delta": add_delta,
        "add_s_ci95_low": bootstrap["ci95_low"],
        "add_s_ci95_high": bootstrap["ci95_high"],
        "objects_nonnegative": objects_nonnegative,
        "full_oracle_gap_closure": gap_closure,
        "pass": passed,
    }


def decide(
    rows: List[dict],
    per_method: List[dict],
    per_object: List[dict],
    iterations: int,
    seed: int,
) -> dict:
    summary = {item["method"]: item for item in per_method}
    full_gap = (
        summary["gt_gt_vis_ransac"]["add_s_0.1d_recall"]
        - summary["pred_vis_ransac"]["add_s_0.1d_recall"]
    )
    better_aggregator = max(
        ("patch_pnp", "pred_vis_ransac"),
        key=lambda method: summary[method]["add_s_0.1d_recall"],
    )
    candidates = [
        ("mask_support", "pred_inter_gt_vis_ransac", "pred_vis_ransac"),
        ("mask_support", "pred_gt_vis_ransac", "pred_vis_ransac"),
        ("xyz_geometry", "gt_inter_vis_ransac", "pred_inter_gt_vis_ransac"),
        ("xyz_geometry", "gt_gt_vis_ransac", "pred_gt_vis_ransac"),
        ("pixel_reliability", "oracle_xyz_top50_ransac", "pred_inter_gt_vis_ransac"),
        ("double_mask", "pred_full_ransac", "pred_vis_ransac"),
        ("aggregation", "oracle_best_pose", better_aggregator),
        ("axis_decoupling", "oracle_best_axis", better_aggregator),
    ]
    comparisons = [
        compare_factor(
            factor,
            method,
            baseline,
            rows,
            per_method,
            per_object,
            full_gap,
            iterations,
            seed + index,
        )
        for index, (factor, method, baseline) in enumerate(candidates)
    ]
    passed = [item for item in comparisons if item["pass"]]
    passed.sort(key=lambda item: item["full_oracle_gap_closure"], reverse=True)
    return {
        "status": "PASS" if passed else "FAIL",
        "primary": passed[0] if passed else None,
        "secondary": passed[1:] if len(passed) > 1 else [],
        "full_oracle_add_s_gap": full_gap,
        "comparisons": comparisons,
        "rule": {
            "bop_ar_delta_min": 0.01,
            "add_s_recall_delta_min": 0.05,
            "bootstrap_ci95_low_strictly_positive": True,
            "objects_nonnegative_min": 6,
            "full_oracle_gap_closure_min": 0.30,
        },
    }


def main() -> int:
    args = parse_args()
    args.config_file = (
        (PROJECT_ROOT / args.config_file).resolve() if not args.config_file.is_absolute() else args.config_file
    )
    args.weights = (PROJECT_ROOT / args.weights).resolve() if not args.weights.is_absolute() else args.weights
    args.output_dir = (
        (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    )
    weight_hash = preflight(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    cfg = configure(args)
    register_datasets_in_cfg(cfg)
    metadata = MetadataCatalog.get(args.dataset)
    data_ref = ref.__dict__[metadata.ref_key]
    object_ids = [data_ref.obj2id[name] for name in metadata.objs]
    images, gt_lookup = build_dataset_lookups(args.dataset)
    target_counts = load_bop_target_counts()
    target_occurrences: Dict[tuple, int] = {}
    model_points = load_model_points(data_ref)
    symmetry_rotations = load_symmetry_rotations(data_ref)
    diameters = {obj_id: float(data_ref.diameters[index]) for index, obj_id in enumerate(object_ids)}

    model, _ = GDRN_double_mask.build_model_optimizer(cfg, is_test=True)
    MyCheckpointer(model, save_dir=str(args.output_dir), prefix_to_remove="_module.").resume_or_load(
        str(args.weights), resume=False
    )
    model.eval()
    loader = build_gdrn_test_loader(cfg, args.dataset, train_objs=metadata.objs, batch_size=1)

    rows: List[dict] = []
    dense_rows: List[dict] = []
    bop_rows: Dict[str, List[dict]] = {method: [] for method in METHODS}
    smoke_counts = {obj_id: 0 for obj_id in object_ids}
    processed_targets = 0
    inference_seconds = 0.0
    stop = False
    depth_cache: Dict[str, np.ndarray] = {}
    threshold = float(cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST)

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
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start

            xyz_batch = get_out_coor(
                cfg,
                output["coor_x"].detach(),
                output["coor_y"].detach(),
                output["coor_z"].detach(),
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
                    scene_im_id = input_item["scene_im_id"][local_index]
                    target_key = (scene_im_id, obj_id)
                    occurrence = target_occurrences.get(target_key, 0)
                    if occurrence >= target_counts.get(target_key, 0):
                        continue
                    target_occurrences[target_key] = occurrence + 1
                    if args.smoke_per_object and smoke_counts[obj_id] >= args.smoke_per_object:
                        continue

                    scene_id, im_id = map(int, scene_im_id.split("/"))
                    instance_id = int(input_item["inst_id"][local_index])
                    gt = gt_lookup[(scene_im_id, instance_id)]
                    image_record = images[scene_im_id]
                    rotation_gt = np.asarray(gt["pose"][:, :3], dtype=np.float64)
                    translation_gt = np.asarray(gt["pose"][:, 3], dtype=np.float64)
                    visibility = float(gt.get("visib_fract", 1.0))
                    camera = input_item["cam"][local_index].cpu().numpy()
                    height = int(input_item["im_H"][local_index])
                    width = int(input_item["im_W"][local_index])
                    extent = input_item["roi_extent"][local_index].cpu().numpy()
                    coord_norm = input_item["roi_coord_2d"][local_index].cpu().numpy().transpose(1, 2, 0)
                    image_points = normalized_image_points(coord_norm, height, width)

                    if scene_im_id not in depth_cache:
                        depth_path = (PROJECT_ROOT / image_record["depth_file"]).resolve()
                        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                        if raw_depth is None:
                            raise FileNotFoundError(depth_path)
                        depth_cache[scene_im_id] = raw_depth.astype(np.float64) / float(
                            image_record["depth_factor"]
                        )
                    depth_m = depth_cache[scene_im_id]
                    gt_visible_full = cocosegm2mask(gt["segmentation"], height, width).astype(bool)
                    gt_full_full = cocosegm2mask(gt["mask_full"], height, width).astype(bool)
                    gt_visible_sampled, in_image = sample_nearest(gt_visible_full, image_points)
                    gt_full_sampled, _ = sample_nearest(gt_full_full, image_points)
                    gt_xyz_m, depth_valid = depth_to_object_coordinates(
                        depth_m,
                        image_points,
                        camera,
                        rotation_gt,
                        translation_gt,
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
                    reliability_map = np.clip(pred_visible_probability, 0.0, 1.0) * np.clip(
                        region_confidence(region_batch[flat_index]), 0.0, 1.0
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
                    pred_gt_vis_corr = build_correspondences_from_dense(
                        image_points, pred_xyz_m, gt_visible, reliability_map
                    )
                    gt_shared_corr = build_correspondences_from_dense(
                        image_points, gt_xyz_m, shared_support, reliability_map
                    )
                    gt_visible_corr = build_correspondences_from_dense(
                        image_points, gt_xyz_m, gt_visible, reliability_map
                    )
                    current_top50_corr = reliable_subset(pred_vis_corr)
                    current_shared_corr = (
                        pred_shared_corr
                        if pred_shared_corr.size < 32
                        else subset_top_fraction(
                            pred_shared_corr,
                            pred_shared_corr.reliability,
                            keep_fraction=0.5,
                            largest=True,
                        )
                    )
                    oracle_top50_corr = (
                        pred_shared_corr
                        if pred_shared_corr.size < 32
                        else subset_top_fraction(
                            pred_shared_corr,
                            xyz_error_map[shared_support],
                            keep_fraction=0.5,
                            largest=False,
                        )
                    )

                    seed = args.seed + processed_targets
                    solutions = {
                        "patch_pnp": timed_solution(
                            lambda: solution_from_pose(
                                rotations[flat_index],
                                translations[flat_index],
                                pred_vis_corr,
                                camera,
                                num_inliers=pred_vis_corr.size,
                            )
                        ),
                        "pred_vis_ransac": ransac(pred_vis_corr, camera, seed),
                        "pred_full_ransac": ransac(pred_full_corr, camera, seed),
                        "pred_inter_gt_vis_ransac": ransac(pred_shared_corr, camera, seed),
                        "pred_gt_vis_ransac": ransac(pred_gt_vis_corr, camera, seed),
                        "gt_inter_vis_ransac": ransac(gt_shared_corr, camera, seed),
                        "gt_gt_vis_ransac": ransac(gt_visible_corr, camera, seed),
                        "current_top50_ransac": ransac(current_top50_corr, camera, seed),
                        "current_shared_top50_ransac": ransac(current_shared_corr, camera, seed),
                        "oracle_xyz_top50_ransac": ransac(oracle_top50_corr, camera, seed),
                    }
                    patch_metrics = solution_metrics(
                        solutions["patch_pnp"],
                        rotation_gt,
                        translation_gt,
                        model_points[obj_id],
                        diameters[obj_id],
                        obj_id,
                        symmetry_rotations[obj_id],
                    )
                    geometric_metrics = solution_metrics(
                        solutions["pred_vis_ransac"],
                        rotation_gt,
                        translation_gt,
                        model_points[obj_id],
                        diameters[obj_id],
                        obj_id,
                        symmetry_rotations[obj_id],
                    )
                    best_pose, best_axis = oracle_selectors(
                        solutions["patch_pnp"],
                        solutions["pred_vis_ransac"],
                        patch_metrics,
                        geometric_metrics,
                        pred_vis_corr,
                        camera,
                    )
                    solutions["oracle_best_pose"] = best_pose
                    solutions["oracle_best_axis"] = best_axis

                    for method in METHODS:
                        solution = solutions[method]
                        metrics = solution_metrics(
                            solution,
                            rotation_gt,
                            translation_gt,
                            model_points[obj_id],
                            diameters[obj_id],
                            obj_id,
                            symmetry_rotations[obj_id],
                        )
                        if solution.success:
                            bop_rows[method].append(
                                {
                                    "scene_id": scene_id,
                                    "im_id": im_id,
                                    "obj_id": obj_id,
                                    "score": 1.0,
                                    "R": solution.rotation,
                                    "t": solution.translation * 1000.0,
                                    "time": -1,
                                }
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
                                "inlier_ratio": (
                                    solution.num_inliers / solution.num_points
                                    if solution.num_points
                                    else 0.0
                                ),
                                "median_reprojection_error_px": solution.median_reprojection_error,
                                "solver_time_ms": solution.solver_time_ms,
                                **metrics,
                            }
                        )

                    visible_metrics = mask_metrics(
                        pred_visible_probability > threshold, gt_visible_sampled.astype(bool)
                    )
                    full_metrics = mask_metrics(
                        pred_full_probability > threshold, gt_full_sampled.astype(bool)
                    )
                    xyz_errors = xyz_error_map[shared_support]
                    gt_reprojection = reprojection_error_for_gt(
                        gt_xyz_m,
                        image_points,
                        gt_visible,
                        camera,
                        rotation_gt,
                        translation_gt,
                    )
                    geometry = dense_geometry_statistics(
                        image_points, pred_xyz_m, pred_visible_support
                    )
                    dense_rows.append(
                        {
                            "scene_id": scene_id,
                            "im_id": im_id,
                            "instance_id": instance_id,
                            "obj_id": obj_id,
                            "obj_name": data_ref.id2obj[obj_id],
                            "visibility": visibility,
                            "visible_mask_iou": visible_metrics["iou"],
                            "visible_mask_precision": visible_metrics["precision"],
                            "visible_mask_recall": visible_metrics["recall"],
                            "full_mask_iou": full_metrics["iou"],
                            "full_mask_precision": full_metrics["precision"],
                            "full_mask_recall": full_metrics["recall"],
                            "pred_visible_points": int(np.count_nonzero(pred_visible_support)),
                            "pred_full_points": int(np.count_nonzero(pred_full_support)),
                            "gt_visible_points": int(np.count_nonzero(gt_visible)),
                            "shared_points": int(np.count_nonzero(shared_support)),
                            "xyz_median_error_m": float(np.median(xyz_errors)) if len(xyz_errors) else float("nan"),
                            "xyz_p90_error_m": float(np.quantile(xyz_errors, 0.9)) if len(xyz_errors) else float("nan"),
                            "xyz_median_error_over_diameter": (
                                float(np.median(xyz_errors) / diameters[obj_id])
                                if len(xyz_errors)
                                else float("nan")
                            ),
                            "reliability_vs_xyz_error_spearman": safe_spearman(
                                reliability_map[shared_support], xyz_errors
                            ),
                            "gt_reprojection_median_px": (
                                float(np.median(gt_reprojection)) if len(gt_reprojection) else float("nan")
                            ),
                            "gt_reprojection_max_px": (
                                float(np.max(gt_reprojection)) if len(gt_reprojection) else float("nan")
                            ),
                            **geometry,
                        }
                    )

                    processed_targets += 1
                    smoke_counts[obj_id] += 1
                    if processed_targets % 100 == 0:
                        print(f"processed {processed_targets} targets", flush=True)
                    if args.max_targets and processed_targets >= args.max_targets:
                        stop = True
                        break
                    if args.smoke_per_object and all(
                        count >= args.smoke_per_object for count in smoke_counts.values()
                    ):
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break

    if not args.smoke_per_object and not args.max_targets and processed_targets != EXPECTED_LMO_TARGETS:
        raise RuntimeError(f"Expected {EXPECTED_LMO_TARGETS} targets, processed {processed_targets}")

    per_method = aggregate(rows, ("method",))
    per_object = aggregate(rows, ("method", "obj_id", "obj_name"))
    per_visibility = aggregate(rows, ("method", "visibility_bin"))
    save_bop_files(args.output_dir, bop_rows)
    bop_scores = run_bop_evaluation(args.output_dir) if args.bop_eval else {}
    for item in per_method:
        score = bop_scores.get(item["method"], {})
        item["bop_ar"] = score.get("bop19_average_recall", float("nan"))
        item["vsd_ar"] = score.get("bop19_average_recall_vsd", float("nan"))
        item["mssd_ar"] = score.get("bop19_average_recall_mssd", float("nan"))
        item["mspd_ar"] = score.get("bop19_average_recall_mspd", float("nan"))

    if processed_targets == EXPECTED_LMO_TARGETS and args.bop_eval:
        conclusion = decide(
            rows,
            per_method,
            per_object,
            args.bootstrap_iterations,
            args.seed,
        )
        reproduction = stage1_reproduction(per_method)
    else:
        conclusion = {"status": "SMOKE_ONLY", "primary": None, "secondary": []}
        reproduction = {"status": "NOT_CHECKED_ON_SMOKE"}

    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_only": True,
        "test_gt_and_depth_are_oracle_only": True,
        "seed": args.seed,
        "dataset": args.dataset,
        "bbox_source": args.bbox_source,
        "expected_full_targets": EXPECTED_LMO_TARGETS,
        "processed_targets": processed_targets,
        "methods": list(METHODS),
        "mask_threshold": threshold,
        "top_fraction": 0.5,
        "minimum_points_before_top_fraction": 32,
        "ransac_reprojection_px": 3.0,
        "ransac_iterations": 100,
        "bootstrap_iterations": args.bootstrap_iterations,
        "weights_sha256": weight_hash,
        "inference_seconds": inference_seconds,
        "device": args.device,
        "bop_evaluation": "computed" if args.bop_eval else "not_requested",
    }
    write_csv(args.output_dir / "per_instance.csv", rows)
    write_csv(args.output_dir / "per_object.csv", per_object)
    write_csv(args.output_dir / "visibility_bins.csv", per_visibility)
    write_csv(args.output_dir / "dense_quality.csv", dense_rows)
    with (args.output_dir / "oracle_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "methods": per_method,
                "conclusion": conclusion,
                "stage1_reproduction": reproduction,
            },
            handle,
            indent=2,
            allow_nan=True,
        )
    with (args.output_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    print(
        json.dumps(
            {
                "processed_targets": processed_targets,
                "conclusion": conclusion,
                "stage1_reproduction": reproduction,
            },
            indent=2,
            allow_nan=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
