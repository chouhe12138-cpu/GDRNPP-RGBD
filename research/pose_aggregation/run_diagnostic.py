#!/usr/bin/env python3
"""Run the pre-registered GDRNPP pose-aggregation diagnostic on LM-O."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / ".local" / "bop_toolkit"))
sys.path.insert(0, str(PROJECT_ROOT / ".local" / "bop_renderer" / "build"))

import cv2
import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation.evaluator import inference_context
from mmcv import Config
from torch.cuda.amp import autocast

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, get_out_coor, get_out_mask
from core.gdrn_modeling.models import GDRN_double_mask
from core.utils.my_checkpoint import MyCheckpointer
from lib.pysixd import inout, misc
from research.pose_aggregation.metrics import pose_metrics
from research.pose_aggregation.solvers import METHODS, build_correspondences, solve_methods


EXPERIMENT_ID = "EXP-20260730-001-gdrnpp-pose-aggregation-diagnostic"
EXPECTED_LMO_TARGETS = 1445
BOP_NAMES = {
    "patch_pnp": "patchpnp_lmo-test.csv",
    "epnp_all": "epnpall_lmo-test.csv",
    "ransac_epnp": "ransacepnp_lmo-test.csv",
    "reliable_ransac": "reliableransac_lmo-test.csv",
    "geom_R_net_t": "geomrnett_lmo-test.csv",
    "net_R_geom_t": "netrgeomt_lmo-test.csv",
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
    parser.add_argument(
        "--reuse-bop-eval",
        action="store_true",
        help="Reuse existing BOP score files after a metrics-only rerun; poses must be unchanged.",
    )
    parser.add_argument("--num-workers", default=0, type=int)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(str(args.config_file))
    cfg.MODEL.WEIGHTS = str(args.weights.resolve())
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.LOAD_DETS_TEST = False
    cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.pretrained = False
    # The diagnostic never trains.  Point TRAIN at the test metadata so the
    # upstream registrar does not require the absent LM-O PBR training set.
    cfg.DATASETS.TRAIN = (args.dataset,)
    cfg.DATASETS.TEST = (args.dataset,)
    cfg.DATASETS.DET_FILES_TEST = ()
    cfg.TEST.TEST_BBOX_TYPE = args.bbox_source
    cfg.TEST.USE_PNP = True
    cfg.TEST.SAVE_RESULTS_ONLY = False
    cfg.TEST.AMP_TEST = args.device.startswith("cuda")
    cfg.DATALOADER.NUM_WORKERS = int(args.num_workers)
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG["lr"])
    cfg.OUTPUT_DIR = str(args.output_dir)
    return cfg


def preflight(args: argparse.Namespace) -> None:
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
        raise RuntimeError(f"{args.device} requested, but torch.cuda.is_available() is False")
    if args.smoke_per_object < 0 or args.max_targets < 0:
        raise ValueError("smoke-per-object and max-targets must be non-negative")
    if (args.bop_eval or args.reuse_bop_eval) and (args.smoke_per_object or args.max_targets):
        raise ValueError("BOP evaluation is only valid for the complete 1445-target run")


def annotation_lookup(dataset_name: str) -> Dict[tuple, dict]:
    lookup: Dict[tuple, dict] = {}
    for image in DatasetCatalog.get(dataset_name):
        for instance_index, annotation in enumerate(image.get("annotations", [])):
            key = (image["scene_im_id"], instance_index)
            lookup[key] = annotation
    return lookup


def load_bop_target_counts() -> Dict[tuple, int]:
    target_path = PROJECT_ROOT / "datasets" / "BOP_DATASETS" / "lmo" / "test_targets_bop19.json"
    with target_path.open(encoding="utf-8") as handle:
        targets = json.load(handle)
    counts = {
        (f"{int(item['scene_id'])}/{int(item['im_id'])}", int(item["obj_id"])): int(item["inst_count"])
        for item in targets
    }
    if sum(counts.values()) != EXPECTED_LMO_TARGETS:
        raise RuntimeError(
            f"LM-O target file contains {sum(counts.values())} instances, expected {EXPECTED_LMO_TARGETS}"
        )
    return counts


def load_model_points(data_ref) -> Dict[int, np.ndarray]:
    points = {}
    for obj_id, model_path in zip(data_ref.id2obj, data_ref.model_paths):
        vertices = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)["pts"]
        if len(vertices) > 2000:
            indices = np.linspace(0, len(vertices) - 1, 2000, dtype=np.int64)
            vertices = vertices[indices]
        points[int(obj_id)] = np.asarray(vertices, dtype=np.float64)
    return points


def load_symmetry_rotations(data_ref) -> Dict[int, np.ndarray]:
    rotations = {}
    for obj_id in data_ref.id2obj:
        model_info = data_ref.get_models_info()[str(obj_id)]
        if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
            transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
            rotations[int(obj_id)] = np.asarray([item["R"] for item in transforms], dtype=np.float64)
        else:
            rotations[int(obj_id)] = np.eye(3, dtype=np.float64)[None]
    return rotations


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else float("nan")


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else float("nan")


def aggregate(rows: List[dict], key_fields: tuple) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, []).append(row)

    results = []
    for key, group in sorted(groups.items()):
        successful = [row for row in group if row["success"]]
        item = {field: value for field, value in zip(key_fields, key)}
        item.update(
            {
                "instances": len(group),
                "successes": len(successful),
                "failure_rate": 1.0 - len(successful) / len(group),
                "mean_rotation_error_deg": finite_mean(row["rotation_error_deg"] for row in successful),
                "median_rotation_error_deg": finite_median(row["rotation_error_deg"] for row in successful),
                "mean_translation_error_mm": finite_mean(row["translation_error_mm"] for row in successful),
                "median_translation_error_mm": finite_median(row["translation_error_mm"] for row in successful),
                "mean_add_s_m": finite_mean(row["add_s_m"] for row in successful),
                "median_add_s_m": finite_median(row["add_s_m"] for row in successful),
                "add_s_0.1d_recall": float(
                    np.mean(
                        [
                            float(row["add_s_0.1d"]) if row["success"] and np.isfinite(row["add_s_0.1d"]) else 0.0
                            for row in group
                        ]
                    )
                ),
                "mean_correspondences": finite_mean(row["num_points"] for row in group),
                "mean_inlier_ratio": finite_mean(row["inlier_ratio"] for row in successful),
                "median_reprojection_px": finite_mean(
                    row["median_reprojection_error_px"] for row in successful
                ),
                "mean_solver_time_ms": finite_mean(row["solver_time_ms"] for row in group),
            }
        )
        results.append(item)
    return results


def visibility_bin(value: float) -> str:
    if value < 0.25:
        return "lt_0.25"
    if value < 0.5:
        return "0.25_to_0.5"
    return "ge_0.5"


def save_bop_files(output_dir: Path, bop_rows: Dict[str, List[dict]]) -> None:
    results_dir = output_dir / "bop_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for method, rows in bop_rows.items():
        inout.save_bop_results(results_dir / BOP_NAMES[method], rows, version="bop19")


def bop_result_hashes(output_dir: Path) -> Dict[str, str]:
    hashes = {}
    for filename in BOP_NAMES.values():
        digest = hashlib.sha256()
        with (output_dir / "bop_results" / filename).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[filename] = digest.hexdigest()
    return hashes


def run_bop_evaluation(output_dir: Path) -> Dict[str, float]:
    toolkit = PROJECT_ROOT / ".local" / "bop_toolkit"
    script = toolkit / "scripts" / "eval_bop19_pose.py"
    results_dir = output_dir / "bop_results"
    eval_dir = output_dir / "bop_eval"
    environment = os.environ.copy()
    environment.update(
        {
            "BOP_PATH": str((PROJECT_ROOT / "datasets" / "BOP_DATASETS").resolve()),
            "BOP_RESULTS_PATH": str(results_dir.resolve()),
            "BOP_EVAL_PATH": str(eval_dir.resolve()),
            "BOP_RENDERER_PATH": str((PROJECT_ROOT / ".local" / "bop_renderer" / "build").resolve()),
            "BOP_NUM_WORKERS": "1",
        }
    )
    command = [
        sys.executable,
        str(script),
        "--renderer_type=cpp",
        "--num_workers=1",
        "--results_path=" + str(results_dir.resolve()),
        "--eval_path=" + str(eval_dir.resolve()),
        "--result_filenames=" + ",".join(BOP_NAMES[method] for method in METHODS),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)
    with (eval_dir / "bop_result_sha256.json").open("w", encoding="utf-8") as handle:
        json.dump(bop_result_hashes(output_dir), handle, indent=2)

    scores = {}
    for method, filename in BOP_NAMES.items():
        result_name = Path(filename).stem
        score_path = eval_dir / result_name / "scores_bop19.json"
        with score_path.open(encoding="utf-8") as handle:
            scores[method] = float(json.load(handle)["bop19_average_recall"])
    return scores


def reuse_bop_evaluation(output_dir: Path) -> Dict[str, float]:
    eval_dir = output_dir / "bop_eval"
    manifest_path = eval_dir / "bop_result_sha256.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing BOP pose hash manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        recorded_hashes = json.load(handle)
    current_hashes = bop_result_hashes(output_dir)
    if current_hashes != recorded_hashes:
        raise RuntimeError("BOP pose files changed; refusing to reuse stale evaluation scores")

    scores = {}
    for method, filename in BOP_NAMES.items():
        score_path = eval_dir / Path(filename).stem / "scores_bop19.json"
        if not score_path.exists():
            raise FileNotFoundError(f"Missing reusable BOP score: {score_path}")
        with score_path.open(encoding="utf-8") as handle:
            scores[method] = float(json.load(handle)["bop19_average_recall"])
    return scores


def decide(summary_by_method: List[dict]) -> dict:
    by_method = {item["method"]: item for item in summary_by_method}
    baseline = by_method["patch_pnp"]
    strong = []
    axis = []
    for method, item in by_method.items():
        if method == "patch_pnp":
            continue
        object_nonnegative = item.get("objects_nonnegative_add_recall", 0)
        bop_delta = item.get("bop_ar", float("nan")) - baseline.get("bop_ar", float("nan"))
        add_delta = item["add_s_0.1d_recall"] - baseline["add_s_0.1d_recall"]
        if np.isfinite(bop_delta) and bop_delta >= 0.005 and add_delta >= 0.005 and object_nonnegative >= 6:
            strong.append(method)

        rot_gain = (
            (baseline["mean_rotation_error_deg"] - item["mean_rotation_error_deg"])
            / baseline["mean_rotation_error_deg"]
        )
        trans_gain = (
            (baseline["mean_translation_error_mm"] - item["mean_translation_error_mm"])
            / baseline["mean_translation_error_mm"]
        )
        if max(rot_gain, trans_gain) >= 0.10 and min(rot_gain, trans_gain) >= -0.05:
            if not np.isfinite(bop_delta) or bop_delta >= -0.002:
                axis.append(method)

    if strong:
        return {"status": "PASS", "methods": strong}
    if axis:
        return {"status": "AXIS_PASS", "methods": axis}
    return {"status": "FAIL", "methods": []}


def main() -> int:
    args = parse_args()
    args.config_file = (PROJECT_ROOT / args.config_file).resolve() if not args.config_file.is_absolute() else args.config_file
    args.weights = (PROJECT_ROOT / args.weights).resolve() if not args.weights.is_absolute() else args.weights
    args.output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    preflight(args)
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
    gt_lookup = annotation_lookup(args.dataset)
    target_counts = load_bop_target_counts()
    target_occurrences: Dict[tuple, int] = {}
    models = load_model_points(data_ref)
    symmetry_rotations = load_symmetry_rotations(data_ref)
    diameters = {obj_id: float(data_ref.diameters[index]) for index, obj_id in enumerate(object_ids)}

    model, _ = GDRN_double_mask.build_model_optimizer(cfg, is_test=True)
    MyCheckpointer(model, save_dir=str(args.output_dir), prefix_to_remove="_module.").resume_or_load(
        str(args.weights), resume=False
    )
    model.eval()

    loader = build_gdrn_test_loader(cfg, args.dataset, train_objs=metadata.objs, batch_size=1)
    rows: List[dict] = []
    bop_rows: Dict[str, List[dict]] = {method: [] for method in METHODS}
    smoke_counts = {obj_id: 0 for obj_id in object_ids}
    processed_targets = 0
    inference_seconds = 0.0
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
            ).cpu().numpy()
            mask_batch = get_out_mask(cfg, output["mask"].detach()).cpu().numpy()
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
                    rotation_gt = np.asarray(gt["pose"][:, :3], dtype=np.float64)
                    translation_gt = np.asarray(gt["pose"][:, 3], dtype=np.float64)
                    visibility = float(gt.get("visib_fract", 1.0))

                    correspondences = build_correspondences(
                        mask_batch[flat_index, 0],
                        xyz_batch[flat_index].transpose(1, 2, 0),
                        input_item["roi_coord_2d"][local_index].cpu().numpy().transpose(1, 2, 0),
                        int(input_item["im_H"][local_index]),
                        int(input_item["im_W"][local_index]),
                        input_item["roi_extent"][local_index].cpu().numpy(),
                        region_logits=region_batch[flat_index],
                        mask_threshold=cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST,
                    )
                    camera_matrix = input_item["cam"][local_index].cpu().numpy()

                    solutions = solve_methods(
                        correspondences,
                        camera_matrix,
                        rotations[flat_index],
                        translations[flat_index],
                        seed=args.seed + processed_targets,
                    )

                    for method, solution in solutions.items():
                        metric_values = {
                            "rotation_error_deg": float("nan"),
                            "translation_error_mm": float("nan"),
                            "add_s_m": float("nan"),
                            "add_s_0.1d": float("nan"),
                        }
                        if solution.success:
                            metric_values = pose_metrics(
                                solution.rotation,
                                solution.translation,
                                rotation_gt,
                                translation_gt,
                                models[obj_id],
                                diameters[obj_id],
                                obj_id,
                                symmetry_rotations[obj_id],
                            )
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
                                    solution.num_inliers / solution.num_points if solution.num_points else 0.0
                                ),
                                "median_reprojection_error_px": solution.median_reprojection_error,
                                "solver_time_ms": solution.solver_time_ms,
                                **metric_values,
                            }
                        )

                    processed_targets += 1
                    smoke_counts[obj_id] += 1
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
    baseline_objects = {
        item["obj_id"]: item for item in per_object if item["method"] == "patch_pnp"
    }
    for method_item in per_method:
        method_objects = [item for item in per_object if item["method"] == method_item["method"]]
        method_item["objects_nonnegative_add_recall"] = sum(
            item["add_s_0.1d_recall"] >= baseline_objects[item["obj_id"]]["add_s_0.1d_recall"]
            for item in method_objects
        )

    save_bop_files(args.output_dir, bop_rows)
    if args.reuse_bop_eval:
        bop_scores = reuse_bop_evaluation(args.output_dir)
    elif args.bop_eval:
        bop_scores = run_bop_evaluation(args.output_dir)
    else:
        bop_scores = {}
    for item in per_method:
        item["bop_ar"] = bop_scores.get(item["method"], float("nan"))

    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "seed": args.seed,
        "dataset": args.dataset,
        "bbox_source": args.bbox_source,
        "expected_full_targets": EXPECTED_LMO_TARGETS,
        "processed_targets": processed_targets,
        "methods": list(METHODS),
        "mask_threshold": float(cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST),
        "reliable_fraction": 0.5,
        "ransac_reprojection_px": 3.0,
        "ransac_iterations": 100,
        "weights_sha256": "bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c",
        "inference_seconds": inference_seconds,
        "device": args.device,
        "bop_evaluation": (
            "reused_hash_verified_scores"
            if args.reuse_bop_eval
            else "computed"
            if args.bop_eval
            else "not_requested"
        ),
    }
    conclusion = decide(per_method) if processed_targets == EXPECTED_LMO_TARGETS else {
        "status": "SMOKE_ONLY",
        "methods": [],
    }

    write_csv(args.output_dir / "per_instance.csv", rows)
    write_csv(args.output_dir / "per_object.csv", per_object)
    write_csv(args.output_dir / "visibility_bins.csv", per_visibility)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"methods": per_method, "conclusion": conclusion}, handle, indent=2, allow_nan=True)
    with (args.output_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)

    print(json.dumps({"processed_targets": processed_targets, "conclusion": conclusion}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
