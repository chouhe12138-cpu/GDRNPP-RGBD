#!/usr/bin/env python3
"""Measure whether frozen GDRNPP Patch-PnP uses progressively improved XYZ."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
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
import torch.nn.functional as F
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation.evaluator import inference_context
from torch.cuda.amp import autocast

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, get_out_coor, get_out_mask
from core.gdrn_modeling.models import GDRN_double_mask
from core.gdrn_modeling.models.model_utils import get_mask_prob, get_rot_mat
from core.gdrn_modeling.models.pose_from_pred import pose_from_pred
from core.gdrn_modeling.models.pose_from_pred_centroid_z import pose_from_pred_centroid_z
from core.gdrn_modeling.models.pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
from core.utils.my_checkpoint import MyCheckpointer
from lib.pysixd import inout
from lib.utils.mask_utils import cocosegm2mask
from research.oracle_diagnostic.oracle_utils import (
    build_correspondences_from_dense,
    depth_to_object_coordinates,
    normalized_image_points,
    normalized_xyz_to_metric,
    prediction_valid_mask,
    region_confidence,
    sample_nearest,
)
from research.oracle_diagnostic.run_oracle_diagnostic import (
    build_dataset_lookups,
    solution_metrics,
    visibility_bin,
)
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
    PoseSolution,
    solution_from_pose,
    solve_ransac_epnp,
)
from research.pose_head_utilization.utilization_utils import (
    DEFAULT_ALPHAS,
    alpha_token,
    interpolate_xyz,
    utilization_decision,
)


EXPERIMENT_ID = "EXP-20260731-004-gdrnpp-pose-head-utilization"
EXPECTED_WEIGHT_HASH = "bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c"
METHODS = tuple(
    f"{family}_{alpha_token(alpha)}"
    for alpha in DEFAULT_ALPHAS
    for family in ("patch", "ransac")
)
BOP_NAMES = {
    method: f"{method.replace('_', '')}_lmo-test.csv"
    for method in METHODS
}


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


def _slice(batch: dict, key: str, index: int):
    value = batch.get(key)
    return None if value is None else value[index : index + 1]


def patch_pose_from_xyz(
    model,
    cfg,
    xyz_normalized: np.ndarray,
    output: dict,
    batch: dict,
    index: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rerun only the frozen official Patch-PnP head for one XYZ intervention."""

    pnp_cfg = cfg.MODEL.POSE_NET.PNP_NET
    net_cfg = cfg.MODEL.POSE_NET
    device = output["coor_x"].device
    dtype = output["coor_x"].dtype
    xyz = torch.as_tensor(
        np.ascontiguousarray(xyz_normalized.transpose(2, 0, 1)),
        device=device,
        dtype=dtype,
    ).unsqueeze(0)
    coor_feat = xyz
    if pnp_cfg.WITH_2D_COORD:
        coord_key = "roi_coord_2d_rel" if pnp_cfg.COORD_2D_TYPE == "rel" else "roi_coord_2d"
        coord = _slice(batch, coord_key, index)
        if coord is None:
            raise RuntimeError(f"Missing {coord_key} required by Patch-PnP")
        coor_feat = torch.cat((coor_feat, coord.to(dtype=dtype)), dim=1)

    region_logits = output["region"][index : index + 1]
    region_attention = (
        F.softmax(region_logits[:, 1:, :, :], dim=1)
        if pnp_cfg.REGION_ATTENTION
        else None
    )
    mask_attention = None
    if pnp_cfg.MASK_ATTENTION != "none":
        mask_attention = get_mask_prob(
            output["mask"][index : index + 1],
            mask_loss_type=net_cfg.LOSS_CFG.MASK_LOSS_TYPE,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with autocast(enabled=bool(cfg.TEST.AMP_TEST)):
        pred_rot_raw, pred_t_raw = model.pnp_net(
            coor_feat,
            region=region_attention,
            extents=_slice(batch, "roi_extent", index),
            mask_attention=mask_attention,
        )
        pred_rot_m = get_rot_mat(pred_rot_raw, pnp_cfg.ROT_TYPE)
        if pnp_cfg.TRANS_TYPE == "centroid_z":
            pred_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_raw[:, :2],
                pred_z_vals=pred_t_raw[:, 2:3],
                roi_cams=_slice(batch, "roi_cam", index),
                roi_centers=_slice(batch, "roi_center", index),
                resize_ratios=_slice(batch, "resize_ratio", index),
                roi_whs=_slice(batch, "roi_wh", index),
                eps=1e-4,
                is_allo="allo" in pnp_cfg.ROT_TYPE,
                z_type=pnp_cfg.Z_TYPE,
                is_train=False,
            )
        elif pnp_cfg.TRANS_TYPE == "centroid_z_abs":
            pred_rot, pred_trans = pose_from_pred_centroid_z_abs(
                pred_rot_m,
                pred_centroids=pred_t_raw[:, :2],
                pred_z_vals=pred_t_raw[:, 2:3],
                roi_cams=_slice(batch, "roi_cam", index),
                eps=1e-4,
                is_allo="allo" in pnp_cfg.ROT_TYPE,
                is_train=False,
            )
        elif pnp_cfg.TRANS_TYPE == "trans":
            pred_rot, pred_trans = pose_from_pred(
                pred_rot_m,
                pred_t_raw,
                eps=1e-4,
                is_allo="allo" in pnp_cfg.ROT_TYPE,
                is_train=False,
            )
        else:
            raise ValueError(f"Unsupported translation type: {pnp_cfg.TRANS_TYPE}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return (
        pred_rot[0].detach().float().cpu().numpy(),
        pred_trans[0].detach().float().cpu().numpy(),
        elapsed_ms,
    )


def timed_ransac(correspondences, camera: np.ndarray, seed: int) -> PoseSolution:
    start = time.perf_counter()
    solution = solve_ransac_epnp(correspondences, camera, seed=seed)
    return replace(solution, solver_time_ms=(time.perf_counter() - start) * 1000.0)


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
            "BOP_RENDERER_PATH": str(
                (PROJECT_ROOT / ".local" / "bop_renderer" / "build").resolve()
            ),
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


def baseline_reproduction(per_method: List[dict]) -> dict:
    old_path = PROJECT_ROOT / "output" / "EXP-20260731-002" / "full" / "oracle_summary.json"
    if not old_path.exists():
        return {"status": "NOT_AVAILABLE"}
    with old_path.open(encoding="utf-8") as handle:
        old = {row["method"]: row for row in json.load(handle)["methods"]}
    new = {row["method"]: row for row in per_method}
    comparisons = {
        "patch": (
            new["patch_a000"]["add_s_0.1d_recall"]
            - old["patch_pnp"]["add_s_0.1d_recall"]
        ),
        "ransac": (
            new["ransac_a000"]["add_s_0.1d_recall"]
            - old["pred_inter_gt_vis_ransac"]["add_s_0.1d_recall"]
        ),
    }
    passed = all(abs(value) <= 1e-12 for value in comparisons.values())
    return {"status": "PASS" if passed else "FAIL", "add_recall_deltas": comparisons}


def main() -> int:
    args = parse_args()
    args.config_file = (
        (PROJECT_ROOT / args.config_file).resolve()
        if not args.config_file.is_absolute()
        else args.config_file
    )
    args.weights = (
        (PROJECT_ROOT / args.weights).resolve()
        if not args.weights.is_absolute()
        else args.weights
    )
    args.output_dir = (
        (PROJECT_ROOT / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir
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
    diameters = {
        obj_id: float(data_ref.diameters[index])
        for index, obj_id in enumerate(object_ids)
    }

    model, _ = GDRN_double_mask.build_model_optimizer(cfg, is_test=True)
    MyCheckpointer(
        model,
        save_dir=str(args.output_dir),
        prefix_to_remove="_module.",
    ).resume_or_load(str(args.weights), resume=False)
    model.eval()
    loader = build_gdrn_test_loader(
        cfg,
        args.dataset,
        train_objs=metadata.objs,
        batch_size=1,
    )

    rows: List[dict] = []
    dense_rows: List[dict] = []
    bop_rows: Dict[str, List[dict]] = {method: [] for method in METHODS}
    smoke_counts = {obj_id: 0 for obj_id in object_ids}
    depth_cache: Dict[str, np.ndarray] = {}
    processed_targets = 0
    inference_seconds = 0.0
    max_alpha0_rotation_difference = 0.0
    max_alpha0_translation_difference = 0.0
    threshold = float(cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST)
    stop = False

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
            ).float().cpu().numpy()
            visible_batch = get_out_mask(cfg, output["mask"].detach()).float().cpu().numpy()
            region_batch = output["region"].detach().float().cpu().numpy()
            original_rotations = output["rot"].detach().float().cpu().numpy()
            original_translations = output["trans"].detach().float().cpu().numpy()

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
                    coord_norm = (
                        input_item["roi_coord_2d"][local_index]
                        .cpu()
                        .numpy()
                        .transpose(1, 2, 0)
                    )
                    image_points = normalized_image_points(coord_norm, height, width)

                    if scene_im_id not in depth_cache:
                        depth_path = (PROJECT_ROOT / image_record["depth_file"]).resolve()
                        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                        if raw_depth is None:
                            raise FileNotFoundError(depth_path)
                        depth_cache[scene_im_id] = raw_depth.astype(np.float64) / float(
                            image_record["depth_factor"]
                        )
                    gt_visible_full = cocosegm2mask(
                        gt["segmentation"],
                        height,
                        width,
                    ).astype(bool)
                    gt_visible_sampled, in_image = sample_nearest(
                        gt_visible_full,
                        image_points,
                    )
                    gt_xyz_m, depth_valid = depth_to_object_coordinates(
                        depth_cache[scene_im_id],
                        image_points,
                        camera,
                        rotation_gt,
                        translation_gt,
                    )
                    gt_visible = gt_visible_sampled.astype(bool) & in_image & depth_valid

                    pred_xyz_norm = xyz_batch[flat_index].transpose(1, 2, 0)
                    pred_visible_probability = visible_batch[flat_index, 0]
                    pred_visible_support = prediction_valid_mask(
                        pred_xyz_norm,
                        pred_visible_probability,
                        extent,
                        threshold,
                    )
                    correction_support = pred_visible_support & gt_visible
                    reliability = np.clip(pred_visible_probability, 0.0, 1.0) * np.clip(
                        region_confidence(region_batch[flat_index]),
                        0.0,
                        1.0,
                    )

                    solutions: Dict[str, PoseSolution] = {}
                    for alpha in DEFAULT_ALPHAS:
                        token = alpha_token(alpha)
                        xyz_interpolated = interpolate_xyz(
                            pred_xyz_norm,
                            gt_xyz_m,
                            extent,
                            correction_support,
                            alpha,
                        )
                        xyz_metric = normalized_xyz_to_metric(xyz_interpolated, extent)
                        correspondences = build_correspondences_from_dense(
                            image_points,
                            xyz_metric,
                            correction_support,
                            reliability,
                        )
                        patch_rotation, patch_translation, patch_time_ms = patch_pose_from_xyz(
                            model,
                            cfg,
                            xyz_interpolated,
                            output,
                            batch,
                            flat_index,
                        )
                        if alpha == 0.0:
                            rerun_rotation = patch_rotation
                            rerun_translation = patch_translation
                            patch_rotation = original_rotations[flat_index]
                            patch_translation = original_translations[flat_index]
                            max_alpha0_rotation_difference = max(
                                max_alpha0_rotation_difference,
                                float(
                                    np.max(
                                        np.abs(
                                            rerun_rotation
                                            - original_rotations[flat_index]
                                        )
                                    )
                                ),
                            )
                            max_alpha0_translation_difference = max(
                                max_alpha0_translation_difference,
                                float(
                                    np.max(
                                        np.abs(
                                            rerun_translation
                                            - original_translations[flat_index]
                                        )
                                    )
                                ),
                            )
                        patch_solution = solution_from_pose(
                            patch_rotation,
                            patch_translation,
                            correspondences,
                            camera,
                            num_inliers=correspondences.size,
                        )
                        solutions[f"patch_{token}"] = replace(
                            patch_solution,
                            solver_time_ms=patch_time_ms,
                        )
                        solutions[f"ransac_{token}"] = timed_ransac(
                            correspondences,
                            camera,
                            args.seed + processed_targets,
                        )

                        errors = np.linalg.norm(xyz_metric - gt_xyz_m, axis=2)[
                            correction_support
                        ]
                        dense_rows.append(
                            {
                                "alpha": alpha,
                                "alpha_token": token,
                                "scene_id": scene_id,
                                "im_id": im_id,
                                "instance_id": instance_id,
                                "obj_id": obj_id,
                                "obj_name": data_ref.id2obj[obj_id],
                                "visibility": visibility,
                                "correction_points": int(
                                    np.count_nonzero(correction_support)
                                ),
                                "pred_visible_points": int(
                                    np.count_nonzero(pred_visible_support)
                                ),
                                "xyz_mean_error_m": (
                                    float(np.mean(errors)) if len(errors) else float("nan")
                                ),
                                "xyz_median_error_m": (
                                    float(np.median(errors)) if len(errors) else float("nan")
                                ),
                                "xyz_mean_error_over_diameter": (
                                    float(np.mean(errors) / diameters[obj_id])
                                    if len(errors)
                                    else float("nan")
                                ),
                            }
                        )

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
                                "median_reprojection_error_px": (
                                    solution.median_reprojection_error
                                ),
                                "solver_time_ms": solution.solver_time_ms,
                                **metrics,
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

    if not args.smoke_per_object and not args.max_targets:
        if processed_targets != EXPECTED_LMO_TARGETS:
            raise RuntimeError(
                f"Expected {EXPECTED_LMO_TARGETS} targets, processed {processed_targets}"
            )
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

    reproduction = (
        baseline_reproduction(per_method)
        if processed_targets == EXPECTED_LMO_TARGETS
        else {"status": "NOT_CHECKED_ON_PARTIAL_RUN"}
    )
    if processed_targets == EXPECTED_LMO_TARGETS and args.bop_eval:
        conclusion = utilization_decision(per_method, per_object)
    else:
        conclusion = {
            "status": "SMOKE_ONLY",
            "next_action": "RUN_FULL_BOP_EVALUATION",
        }

    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_only": True,
        "gt_xyz_is_oracle_only": True,
        "deployable_method": False,
        "seed": args.seed,
        "dataset": args.dataset,
        "bbox_source": args.bbox_source,
        "expected_full_targets": EXPECTED_LMO_TARGETS,
        "processed_targets": processed_targets,
        "alphas": list(DEFAULT_ALPHAS),
        "intervention": "pred + alpha * (gt - pred) on pred_visible intersect gt_visible",
        "ransac_support": "fixed pred_visible intersect gt_visible for every alpha",
        "fixed_components": [
            "visible_mask",
            "region_logits",
            "2d_coordinates",
            "patch_pnp_weights",
            "correspondence_support",
        ],
        "methods": list(METHODS),
        "mask_threshold": threshold,
        "ransac_reprojection_px": 3.0,
        "ransac_iterations": 100,
        "weights_sha256": weight_hash,
        "network_inference_seconds": inference_seconds,
        "max_alpha0_rotation_abs_difference": max_alpha0_rotation_difference,
        "max_alpha0_translation_abs_difference_m": max_alpha0_translation_difference,
        "alpha0_reentry_is_audit_only": True,
        "device": args.device,
        "bop_evaluation": "computed" if args.bop_eval else "not_requested",
    }
    write_csv(args.output_dir / "per_instance.csv", rows)
    write_csv(args.output_dir / "per_object.csv", per_object)
    write_csv(args.output_dir / "visibility_bins.csv", per_visibility)
    write_csv(args.output_dir / "dense_interpolation.csv", dense_rows)
    with (args.output_dir / "utilization_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "methods": per_method,
                "conclusion": conclusion,
                "baseline_reproduction": reproduction,
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
                "baseline_reproduction": reproduction,
            },
            indent=2,
            allow_nan=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
