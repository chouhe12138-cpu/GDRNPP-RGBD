#!/usr/bin/env python3
"""Run an aggregate-only frozen Patch-PnP information-flow diagnostic."""

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
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / ".local" / "bop_toolkit"))
sys.path.insert(0, str(PROJECT_ROOT / ".local" / "bop_renderer" / "build"))

import cv2
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import inference_context
from torch.cuda.amp import autocast

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, get_out_mask
from core.gdrn_modeling.models import GDRN_double_mask
from core.gdrn_modeling.models.heads.cpm_pnp_net import (
    CorrespondenceAwareMomentPnPNet,
)
from core.utils.my_checkpoint import MyCheckpointer
from lib.pysixd import inout
from lib.utils.mask_utils import cocosegm2mask
from research.oracle_diagnostic.oracle_utils import (
    depth_to_object_coordinates,
    normalized_image_points,
    prediction_valid_mask,
    sample_nearest,
)
from research.oracle_diagnostic.run_oracle_diagnostic import build_dataset_lookups
from research.pose_aggregation.metrics import pose_metrics
from research.pose_aggregation.run_diagnostic import (
    configure,
    load_bop_target_counts,
    load_model_points,
    load_symmetry_rotations,
)
from research.pose_head_diagnostic.diagnostic_utils import (
    apply_intervention,
    matched_spatial_masks,
    response_metrics,
    rotation_geodesic_deg,
    tensor_state_sha256,
)
from research.pose_head_diagnostic.run_information_flow import (
    CPM_MOMENT_CONDITIONS,
    apply_cpm_diagnostic_intervention,
    conditions_for_model,
    layers_for_model,
    run_head_with_hooks,
    sha256,
)
from research.pose_head_diagnostic.statistical_utils import (
    aggregate_scalar_records,
    assign_quartile_labels,
)
from research.pose_head_utilization.utilization_utils import metric_xyz_to_normalized


EXPECTED_OFFICIAL_HASH = "bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c"
# The formal comparison protocol is FP32, matching the C1/B/C2 evaluator.
# ADD(-S) here is the diagnostic's 1,445-target micro recall (730 successes),
# not the eight-object macro average printed by the training evaluator.
EXPECTED_OFFICIAL_ADD_RECALL = 0.5051903114186851
EXPECTED_OFFICIAL_BOP_AR = 0.6904152249134947
OFFICIAL_BOP_TOLERANCE = 5e-5
RAW_REENTRY_TOLERANCE_CUDA = 5e-5
POSE_ROTATION_REENTRY_TOLERANCE_CUDA = 3e-4
POSE_TRANSLATION_REENTRY_TOLERANCE_CUDA = 5e-5
REENTRY_TOLERANCE_CPU = 1e-6
EXPERIMENT_ID = "EXP-20260804-007-pose-head-information-flow"
MODEL_ROLES = ("official", "c1", "pnp_adapted", "joint", "cpm")
MODES = ("smoke", "audit80", "full")
POSE_METRICS = (
    "rotation_delta_deg",
    "translation_delta_mm",
    "rotation_error_deg",
    "translation_error_mm",
    "rotation_error_improvement_deg",
    "translation_error_improvement_mm",
    "add_s_m",
    "add_s_0.1d",
    "add_s_improvement",
)
COMPONENT_METRICS = (
    "centroid_xy_delta",
    "depth_raw_delta",
    "raw_rotation_relative_l2",
    "raw_translation_relative_l2",
    "final_pose_relative_l2",
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model-role", choices=MODEL_ROLES, required=True)
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dataset", default="lmo_bop_test")
    parser.add_argument("--bbox-source", choices=("gt",), default="gt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=20260804, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--bootstrap-samples", default=1000, type=int)
    parser.add_argument("--bop-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    # Compatibility attributes consumed by the shared configure() helper.
    parser.set_defaults(smoke_per_object=0, max_targets=0, bop_eval=False)
    return parser.parse_args()


def visibility_bin(value: float) -> str:
    if value < 0.1:
        return "lt_0.1"
    if value < 0.3:
        return "0.1_to_0.3"
    if value < 0.5:
        return "0.3_to_0.5"
    return "ge_0.5"


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def deterministic_subset(
    images: Dict[str, dict],
    object_ids: List[int],
    target_counts: Dict[tuple, int],
    per_object: int,
) -> set[tuple[str, int]]:
    """Choose fixed visibility-spanning targets for smoke/audit modes."""

    candidates: Dict[int, list[tuple[float, str, int]]] = {
        obj_id: [] for obj_id in object_ids
    }
    for scene_im_id, image in sorted(images.items()):
        used: Dict[int, int] = {}
        for instance_id, annotation in enumerate(image.get("annotations", [])):
            obj_id = object_ids[int(annotation["category_id"])]
            occurrence = used.get(obj_id, 0)
            if occurrence >= target_counts.get((scene_im_id, obj_id), 0):
                continue
            used[obj_id] = occurrence + 1
            candidates[obj_id].append(
                (
                    float(annotation.get("visib_fract", 1.0)),
                    scene_im_id,
                    instance_id,
                )
            )
    selected = set()
    for obj_id in object_ids:
        ordered = sorted(candidates[obj_id])
        if len(ordered) < per_object:
            raise RuntimeError(f"Object {obj_id} has fewer than {per_object} targets")
        indices = np.linspace(0, len(ordered) - 1, per_object, dtype=np.int64)
        selected.update((ordered[index][1], ordered[index][2]) for index in indices)
    return selected


def save_bop_results(output_dir: Path, rows: Dict[str, List[dict]]) -> Dict[str, str]:
    result_dir = output_dir / "bop_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for condition, condition_rows in rows.items():
        filename = f"{condition.replace('_', '')}_lmo-test.csv"
        path = result_dir / filename
        inout.save_bop_results(path, condition_rows, version="bop19")
        hashes[filename] = sha256(path)
    return hashes


def run_bop_evaluation(output_dir: Path, conditions: tuple[str, ...]) -> dict:
    toolkit = PROJECT_ROOT / ".local" / "bop_toolkit"
    result_dir = output_dir / "bop_results"
    eval_dir = output_dir / "bop_eval"
    filenames = [f"{condition.replace('_', '')}_lmo-test.csv" for condition in conditions]
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
        "--result_filenames=" + ",".join(filenames),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)
    scores = {}
    for condition, filename in zip(conditions, filenames):
        with (eval_dir / Path(filename).stem / "scores_bop19.json").open(
            encoding="utf-8"
        ) as handle:
            scores[condition] = float(json.load(handle)["bop19_average_recall"])
    return scores


def output_hashes(output_dir: Path) -> dict:
    hashes = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "hashes.sha256":
            hashes[str(path.relative_to(output_dir))] = sha256(path)
    return hashes


def maybe_resume(args: argparse.Namespace, checkpoint_hash: str) -> bool:
    state_path = args.output_dir / "run_state.json"
    protocol_path = args.output_dir / "protocol.json"
    if not args.resume or not state_path.exists() or not protocol_path.exists():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = {
        "mode": args.mode,
        "model_role": args.model_role,
        "seed": args.seed,
        "checkpoint_sha256": checkpoint_hash,
    }
    if state.get("status") != "COMPLETE" or any(
        protocol.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("Existing output is incomplete or its frozen protocol differs")
    hash_path = args.output_dir / "hashes.sha256"
    if not hash_path.exists():
        raise RuntimeError("Completed output is missing hashes.sha256")
    recorded_hashes = json.loads(hash_path.read_text(encoding="utf-8"))
    for relative_path, expected_hash in recorded_hashes.items():
        path = args.output_dir / relative_path
        if not path.exists() or sha256(path) != expected_hash:
            raise RuntimeError(f"Completed output hash mismatch: {relative_path}")
    print(json.dumps({"resume": "COMPLETE_OUTPUT_REUSED", **expected}, indent=2))
    return True


def main() -> int:
    args = parse_args()
    args.config_file = (
        args.config_file if args.config_file.is_absolute() else PROJECT_ROOT / args.config_file
    ).resolve()
    args.weights = (
        args.weights if args.weights.is_absolute() else PROJECT_ROOT / args.weights
    ).resolve()
    args.output_dir = (
        args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    ).resolve()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    if args.mode == "full" and not args.bop_eval:
        raise ValueError("Full mode requires --bop-eval")
    checkpoint_hash = sha256(args.weights)
    if args.model_role == "official" and checkpoint_hash != EXPECTED_OFFICIAL_HASH:
        raise RuntimeError(f"Unexpected official checkpoint SHA-256: {checkpoint_hash}")
    if maybe_resume(args, checkpoint_hash):
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(
            "Output directory is not empty; use a fresh directory or a valid --resume"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    cfg = configure(args)
    # Freeze numerical precision across CPU/GPU and model roles. The shared
    # diagnostic configure() helper enables AMP for CUDA, but repeated FP16
    # Patch-PnP calls differ by one or two quantization steps and cannot satisfy
    # the exact baseline re-entry gate.
    cfg.TEST.AMP_TEST = False
    register_datasets_in_cfg(cfg)
    metadata = MetadataCatalog.get(args.dataset)
    data_ref = ref.__dict__[metadata.ref_key]
    object_ids = [int(data_ref.obj2id[name]) for name in metadata.objs]
    images, gt_lookup = build_dataset_lookups(args.dataset)
    target_counts = load_bop_target_counts()
    per_object = {"smoke": 1, "audit80": 10}.get(args.mode)
    selected_targets = (
        deterministic_subset(images, object_ids, target_counts, per_object)
        if per_object is not None
        else None
    )
    expected_targets = {"smoke": 8, "audit80": 80, "full": 1445}[args.mode]

    model_points = load_model_points(data_ref)
    symmetry_rotations = load_symmetry_rotations(data_ref)
    diameters = {
        obj_id: float(data_ref.diameters[index])
        for index, obj_id in enumerate(object_ids)
    }
    model, optimizer = GDRN_double_mask.build_model_optimizer(cfg, is_test=True)
    if optimizer is not None:
        raise RuntimeError("Frozen diagnostic unexpectedly constructed an optimizer")
    MyCheckpointer(
        model, save_dir=str(args.output_dir), prefix_to_remove="_module."
    ).resume_or_load(str(args.weights), resume=False)
    model.eval()
    layers = layers_for_model(model)
    conditions = conditions_for_model(model)
    is_cpm = isinstance(model.pnp_net, CorrespondenceAwareMomentPnPNet)
    if (args.model_role == "cpm") != is_cpm:
        raise RuntimeError(
            "--model-role cpm must agree with the pose-head type built by the config"
        )
    mask_attention_type = str(cfg.MODEL.POSE_NET.PNP_NET.MASK_ATTENTION)
    if is_cpm and mask_attention_type != "mul":
        raise RuntimeError("CPM diagnostic requires MASK_ATTENTION='mul'")
    if not is_cpm and mask_attention_type != "none":
        raise RuntimeError("The frozen ConvPnP protocol requires MASK_ATTENTION='none'")
    layer_metrics = tuple(
        f"{layer}_{metric}"
        for layer in layers
        for metric in ("relative_l2", "cosine_distance", "mean_absolute")
    )
    state_before = tensor_state_sha256(model.state_dict())
    loader = build_gdrn_test_loader(
        cfg, args.dataset, train_objs=metadata.objs, batch_size=1
    )

    scalar_records: List[dict] = []
    bop_rows: Dict[str, List[dict]] = {condition: [] for condition in conditions}
    shapes: Dict[str, list[int]] = {}
    occurrences: Dict[tuple, int] = {}
    depth_cache: Dict[str, np.ndarray] = {}
    threshold = float(cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST)
    if args.device.startswith("cuda"):
        raw_reentry_tolerance = RAW_REENTRY_TOLERANCE_CUDA
        rotation_reentry_tolerance = POSE_ROTATION_REENTRY_TOLERANCE_CUDA
        translation_reentry_tolerance = POSE_TRANSLATION_REENTRY_TOLERANCE_CUDA
    else:
        raw_reentry_tolerance = REENTRY_TOLERANCE_CPU
        rotation_reentry_tolerance = REENTRY_TOLERANCE_CPU
        translation_reentry_tolerance = REENTRY_TOLERANCE_CPU
    processed_targets = 0
    empty_support_targets = 0
    max_raw_rotation_reentry = 0.0
    max_raw_translation_reentry = 0.0
    max_rotation_reentry = 0.0
    max_translation_reentry = 0.0

    with inference_context(model), torch.no_grad():
        for inputs in loader:
            if processed_targets >= expected_targets:
                break
            if not isinstance(inputs, list):
                inputs = [inputs]
            batch = batch_data(cfg, inputs, device=args.device, phase="test")
            captured: Dict[str, torch.Tensor] = {}

            def capture_pnp_input(_module, hook_inputs, hook_kwargs):
                captured["coor"] = (
                    hook_inputs[0].detach().float().cpu().clone()
                )
                captured["region"] = (
                    hook_kwargs["region"].detach().float().cpu().clone()
                )
                mask_attention = hook_kwargs.get("mask_attention")
                if mask_attention is not None:
                    captured["mask_attention"] = (
                        mask_attention.detach().float().cpu().clone()
                    )

            def capture_pnp_output(_module, _hook_inputs, hook_output):
                captured["raw_rotation"] = (
                    hook_output[0].detach().float().cpu().clone()
                )
                captured["raw_translation"] = (
                    hook_output[1].detach().float().cpu().clone()
                )

            input_handle = model.pnp_net.register_forward_pre_hook(
                capture_pnp_input, with_kwargs=True
            )
            output_handle = model.pnp_net.register_forward_hook(capture_pnp_output)
            try:
                with autocast(enabled=bool(cfg.TEST.AMP_TEST)):
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
            finally:
                input_handle.remove()
                output_handle.remove()
            if not all(
                key in captured
                for key in ("coor", "region", "raw_rotation", "raw_translation")
            ):
                raise RuntimeError("Failed to capture the effective Patch-PnP I/O")
            if captured["coor"].shape[1] != 5 or captured["region"].shape[1] != 64:
                raise RuntimeError(
                    "Unexpected effective Patch-PnP inputs: "
                    f"coor={tuple(captured['coor'].shape)}, "
                    f"region={tuple(captured['region'].shape)}"
                )
            visible_batch = get_out_mask(cfg, output["mask"]).detach().float().cpu().numpy()
            original_rotations = output["rot"].detach().float().cpu().numpy()
            original_translations = output["trans"].detach().float().cpu().numpy()

            flat_index = -1
            for input_item in inputs:
                for local_index in range(len(input_item["roi_img"])):
                    if processed_targets >= expected_targets:
                        break
                    flat_index += 1
                    class_index = int(input_item["roi_cls"][local_index])
                    obj_id = object_ids[class_index]
                    scene_im_id = input_item["scene_im_id"][local_index]
                    target_key = (scene_im_id, obj_id)
                    occurrence = occurrences.get(target_key, 0)
                    if occurrence >= target_counts.get(target_key, 0):
                        continue
                    occurrences[target_key] = occurrence + 1
                    instance_id = int(input_item["inst_id"][local_index])
                    if selected_targets is not None and (
                        scene_im_id,
                        instance_id,
                    ) not in selected_targets:
                        continue

                    gt = gt_lookup[(scene_im_id, instance_id)]
                    image_record = images[scene_im_id]
                    height = int(input_item["im_H"][local_index])
                    width = int(input_item["im_W"][local_index])
                    camera = input_item["cam"][local_index].cpu().numpy()
                    extent = input_item["roi_extent"][local_index].cpu().numpy()
                    effective_coor = captured["coor"][flat_index].numpy().transpose(1, 2, 0)
                    xyz = effective_coor[..., :3]
                    roi_2d = effective_coor[..., 3:5]
                    region = captured["region"][flat_index].numpy().transpose(1, 2, 0)
                    effective_visible_mask = (
                        captured["mask_attention"][flat_index, 0].numpy()
                        if "mask_attention" in captured
                        else visible_batch[flat_index, 0]
                    )
                    image_points = normalized_image_points(roi_2d, height, width)
                    if scene_im_id not in depth_cache:
                        depth_path = (PROJECT_ROOT / image_record["depth_file"]).resolve()
                        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                        if raw_depth is None:
                            raise FileNotFoundError(depth_path)
                        depth_cache[scene_im_id] = raw_depth.astype(np.float64) / float(
                            image_record["depth_factor"]
                        )
                    visible_full = cocosegm2mask(
                        gt["segmentation"], height, width
                    ).astype(bool)
                    sampled_visible, in_image = sample_nearest(visible_full, image_points)
                    rotation_gt = np.asarray(gt["pose"][:, :3], dtype=np.float64)
                    translation_gt = np.asarray(gt["pose"][:, 3], dtype=np.float64)
                    gt_xyz_m, depth_valid = depth_to_object_coordinates(
                        depth_cache[scene_im_id],
                        image_points,
                        camera,
                        rotation_gt,
                        translation_gt,
                    )
                    gt_visible = sampled_visible.astype(bool) & in_image & depth_valid
                    predicted_support = prediction_valid_mask(
                        xyz, visible_batch[flat_index, 0], extent, threshold
                    )
                    support = predicted_support & gt_visible
                    if not np.any(support):
                        empty_support_targets += 1
                    gt_xyz = metric_xyz_to_normalized(gt_xyz_m, extent)
                    xyz_errors = np.linalg.norm(xyz - gt_xyz, axis=2)[support]
                    xyz_error_mean = (
                        float(np.mean(xyz_errors)) if len(xyz_errors) else float("nan")
                    )
                    instance_seed = args.seed + processed_targets * 1009
                    spatial_masks = matched_spatial_masks(
                        xyz, gt_xyz, support, instance_seed
                    )
                    baseline_activations = None
                    baseline_rotation = None
                    baseline_translation = None
                    baseline_pose_metrics = None

                    for condition in conditions:
                        if is_cpm:
                            xyz_i, roi_i, region_i, moment_condition = (
                                apply_cpm_diagnostic_intervention(
                                    xyz,
                                    gt_xyz,
                                    roi_2d,
                                    region,
                                    support,
                                    condition,
                                    instance_seed,
                                )
                            )
                        else:
                            xyz_i, roi_i, region_i = apply_intervention(
                                xyz,
                                gt_xyz,
                                roi_2d,
                                region,
                                support,
                                condition,
                                instance_seed,
                            )
                            moment_condition = None
                        activations, rotation, translation = run_head_with_hooks(
                            model,
                            cfg,
                            xyz_i,
                            roi_i,
                            region_i,
                            effective_visible_mask,
                            batch,
                            flat_index,
                            moment_condition=moment_condition,
                        )
                        metrics = pose_metrics(
                            rotation,
                            translation,
                            rotation_gt,
                            translation_gt,
                            model_points[obj_id],
                            diameters[obj_id],
                            obj_id,
                            symmetry_rotations[obj_id],
                        )
                        if condition == "baseline":
                            baseline_activations = activations
                            baseline_rotation = rotation
                            baseline_translation = translation
                            baseline_pose_metrics = metrics
                            max_raw_rotation_reentry = max(
                                max_raw_rotation_reentry,
                                float(
                                    torch.max(
                                        torch.abs(
                                            activations["raw_rotation"].float()
                                            - captured["raw_rotation"][flat_index : flat_index + 1]
                                        )
                                    ).item()
                                ),
                            )
                            max_raw_translation_reentry = max(
                                max_raw_translation_reentry,
                                float(
                                    torch.max(
                                        torch.abs(
                                            activations["raw_translation"].float()
                                            - captured["raw_translation"][flat_index : flat_index + 1]
                                        )
                                    ).item()
                                ),
                            )
                            max_rotation_reentry = max(
                                max_rotation_reentry,
                                float(
                                    np.max(
                                        np.abs(rotation - original_rotations[flat_index])
                                    )
                                ),
                            )
                            max_translation_reentry = max(
                                max_translation_reentry,
                                float(
                                    np.max(
                                        np.abs(
                                            translation
                                            - original_translations[flat_index]
                                        )
                                    )
                                ),
                            )
                        layer_values = {}
                        for layer in layers:
                            shapes.setdefault(layer, list(activations[layer].shape))
                            for metric_name, value in response_metrics(
                                baseline_activations[layer], activations[layer]
                            ).items():
                                layer_values[f"{layer}_{metric_name}"] = value
                        condition_support = (
                            np.zeros_like(support)
                            if condition == "baseline" or condition in CPM_MOMENT_CONDITIONS
                            else spatial_masks.get(condition, support)
                        )
                        record = {
                            "target_index": processed_targets,
                            "condition": condition,
                            "intervention_domain": (
                                "none"
                                if condition == "baseline"
                                else (
                                    "moment_descriptor"
                                    if condition in CPM_MOMENT_CONDITIONS
                                    else "pixel_input"
                                )
                            ),
                            "obj_name": data_ref.id2obj[obj_id],
                            "visibility_bin": visibility_bin(
                                float(gt.get("visib_fract", 1.0))
                            ),
                            "symmetry_group": (
                                "symmetric" if obj_id in {10, 11} else "non_symmetric"
                            ),
                            "support_points": int(np.count_nonzero(support)),
                            "changed_points": int(np.count_nonzero(condition_support)),
                            "xyz_error_mean": xyz_error_mean,
                            "rotation_delta_deg": rotation_geodesic_deg(
                                rotation, baseline_rotation
                            ),
                            "translation_delta_mm": float(
                                np.linalg.norm(translation - baseline_translation)
                                * 1000.0
                            ),
                            "centroid_xy_delta": float(
                                np.linalg.norm(
                                    activations["raw_translation"][0, :2].numpy()
                                    - baseline_activations["raw_translation"][0, :2].numpy()
                                )
                            ),
                            "depth_raw_delta": float(
                                abs(
                                    activations["raw_translation"][0, 2].item()
                                    - baseline_activations["raw_translation"][0, 2].item()
                                )
                            ),
                            **metrics,
                            "rotation_error_improvement_deg": float(
                                baseline_pose_metrics["rotation_error_deg"]
                                - metrics["rotation_error_deg"]
                            ),
                            "translation_error_improvement_mm": float(
                                baseline_pose_metrics["translation_error_mm"]
                                - metrics["translation_error_mm"]
                            ),
                            "add_s_improvement": float(
                                metrics["add_s_0.1d"]
                                - baseline_pose_metrics["add_s_0.1d"]
                            ),
                            **layer_values,
                        }
                        scalar_records.append(record)
                        bop_rows[condition].append(
                            {
                                "scene_id": int(scene_im_id.split("/")[0]),
                                "im_id": int(scene_im_id.split("/")[1]),
                                "obj_id": obj_id,
                                "score": 1.0,
                                "R": rotation,
                                "t": translation * 1000.0,
                                "time": -1,
                            }
                        )
                    processed_targets += 1
                if processed_targets >= expected_targets:
                    break

    state_after = tensor_state_sha256(model.state_dict())
    baseline_records = [
        record for record in scalar_records if record["condition"] == "baseline"
    ]
    quartiles = assign_quartile_labels(baseline_records, "support_points")
    quartile_map = {
        record["target_index"]: label
        for record, label in zip(baseline_records, quartiles)
    }
    for record in scalar_records:
        record["support_quartile"] = quartile_map[record["target_index"]]

    aggregate_args = {
        "records": scalar_records,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
    }
    overall = aggregate_scalar_records(
        group_fields=("condition",), metric_fields=POSE_METRICS, **aggregate_args
    )
    layer_summary = aggregate_scalar_records(
        group_fields=("condition",), metric_fields=layer_metrics, **aggregate_args
    )
    component_summary = aggregate_scalar_records(
        group_fields=("condition",), metric_fields=COMPONENT_METRICS, **aggregate_args
    )
    object_summary = aggregate_scalar_records(
        group_fields=("condition", "obj_name"),
        metric_fields=POSE_METRICS,
        **aggregate_args,
    )
    visibility_summary = aggregate_scalar_records(
        group_fields=("condition", "visibility_bin"),
        metric_fields=POSE_METRICS,
        **aggregate_args,
    )
    symmetry_summary = aggregate_scalar_records(
        group_fields=("condition", "symmetry_group"),
        metric_fields=POSE_METRICS,
        **aggregate_args,
    )
    support_summary = aggregate_scalar_records(
        group_fields=("condition", "support_quartile"),
        metric_fields=POSE_METRICS,
        **aggregate_args,
    )
    condition_counts = {
        condition: sum(
            record["condition"] == condition for record in scalar_records
        )
        for condition in conditions
    }
    baseline_add_recall = float(
        np.mean(
            [
                record["add_s_0.1d"]
                for record in scalar_records
                if record["condition"] == "baseline"
            ]
        )
    )

    bop_scores = {}
    bop_hashes = {}
    if args.mode == "full":
        bop_hashes = save_bop_results(args.output_dir, bop_rows)
        bop_scores = run_bop_evaluation(args.output_dir, conditions)
    official_add_reproduced = (
        args.model_role != "official"
        or args.mode != "full"
        or abs(baseline_add_recall - EXPECTED_OFFICIAL_ADD_RECALL) <= 1e-12
    )
    official_bop_reproduced = (
        args.model_role != "official"
        or args.mode != "full"
        or abs(bop_scores["baseline"] - EXPECTED_OFFICIAL_BOP_AR)
        <= OFFICIAL_BOP_TOLERANCE
    )
    nonfinite_scalar_count = int(
        sum(
            not np.isfinite(value)
            for record in scalar_records
            for value in record.values()
            if isinstance(value, (int, float))
        )
    )
    expected_nonfinite_scalar_count = empty_support_targets * len(conditions)
    unexpected_nonfinite_scalar_count = (
        nonfinite_scalar_count - expected_nonfinite_scalar_count
    )

    protocol = {
        "experiment_id": EXPERIMENT_ID,
        "mode": args.mode,
        "model_role": args.model_role,
        "seed": args.seed,
        "dataset": args.dataset,
        "checkpoint": str(args.weights),
        "checkpoint_sha256": checkpoint_hash,
        "config": str(args.config_file),
        "config_sha256": sha256(args.config_file),
        "conditions": conditions,
        "layers": layers,
        "fixed_support": "predicted-visible AND GT-visible AND valid-depth",
        "intervention_points": {
            "pixel_input": (
                "effective Patch-PnP input after any quality/coverage module; "
                "all non-target inputs remain fixed"
            ),
            "moment_descriptor": (
                "CPM raw low-order descriptor before deterministic scaling and MLP; "
                "used only by coverage_only and cxu_null"
                if is_cpm
                else None
            ),
        },
        "instance_level_features_persisted": False,
        "parameter_updates": 0,
        "optimizer_created": optimizer is not None,
        "precision": "FP32",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    architecture = {
        "components": {
            "xyz": 3,
            "roi_2d": 2,
            "effective_region": 64,
            "visible_mask": 1 if is_cpm else 0,
        },
        "layers": shapes,
        "rotation_representation": str(cfg.MODEL.POSE_NET.PNP_NET.ROT_TYPE),
        "translation_representation": str(cfg.MODEL.POSE_NET.PNP_NET.TRANS_TYPE),
    }
    quality_control = {
        "expected_targets": expected_targets,
        "processed_targets": processed_targets,
        "conditions": len(conditions),
        "condition_counts": condition_counts,
        "empty_support_targets": empty_support_targets,
        "nonfinite_scalar_count": nonfinite_scalar_count,
        "expected_nonfinite_scalar_count": expected_nonfinite_scalar_count,
        "unexpected_nonfinite_scalar_count": unexpected_nonfinite_scalar_count,
        "max_baseline_raw_rotation_abs_error": max_raw_rotation_reentry,
        "max_baseline_raw_translation_abs_error": max_raw_translation_reentry,
        "max_baseline_rotation_abs_error": max_rotation_reentry,
        "max_baseline_translation_abs_error": max_translation_reentry,
        "raw_reentry_tolerance": raw_reentry_tolerance,
        "rotation_reentry_tolerance": rotation_reentry_tolerance,
        "translation_reentry_tolerance": translation_reentry_tolerance,
        "state_unchanged": state_before == state_after,
        "baseline_add_s_0.1d_recall": baseline_add_recall,
        "official_add_reproduced": official_add_reproduced,
        "official_bop_reproduced": official_bop_reproduced,
        "instance_level_feature_file_written": False,
    }
    quality_control["passed"] = bool(
        processed_targets == expected_targets
        and all(count == expected_targets for count in condition_counts.values())
        and state_before == state_after
        and unexpected_nonfinite_scalar_count == 0
        and max_raw_rotation_reentry <= raw_reentry_tolerance
        and max_raw_translation_reentry <= raw_reentry_tolerance
        and max_rotation_reentry <= rotation_reentry_tolerance
        and max_translation_reentry <= translation_reentry_tolerance
        and official_add_reproduced
        and official_bop_reproduced
    )

    write_json(args.output_dir / "protocol.json", protocol)
    write_json(args.output_dir / "architecture.json", architecture)
    write_json(args.output_dir / "quality_control.json", quality_control)
    write_csv(args.output_dir / "overall_condition_summary.csv", overall)
    write_csv(args.output_dir / "layer_response_summary.csv", layer_summary)
    write_csv(args.output_dir / "pose_component_summary.csv", component_summary)
    write_csv(args.output_dir / "per_object_summary.csv", object_summary)
    write_csv(args.output_dir / "visibility_summary.csv", visibility_summary)
    write_csv(args.output_dir / "symmetry_summary.csv", symmetry_summary)
    write_csv(args.output_dir / "support_quartile_summary.csv", support_summary)
    write_json(
        args.output_dir / "condition_summary.json",
        {"overall": overall, "quality_control": quality_control},
    )

    write_json(
        args.output_dir / "bop_score_summary.json",
        {"scores": bop_scores, "pose_file_sha256": bop_hashes},
    )
    write_json(
        args.output_dir / "run_state.json",
        {
            "status": "COMPLETE" if quality_control["passed"] else "FAILED",
            "model_role": args.model_role,
            "mode": args.mode,
        },
    )
    write_json(args.output_dir / "hashes.sha256", output_hashes(args.output_dir))
    print(json.dumps(quality_control, indent=2))
    return 0 if quality_control["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
