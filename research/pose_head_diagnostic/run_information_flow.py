#!/usr/bin/env python3
"""Run a frozen, one-target-per-object Patch-PnP information-flow smoke test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / ".local" / "bop_toolkit"))
sys.path.insert(0, str(PROJECT_ROOT / ".local" / "bop_renderer" / "build"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import inference_context
from torch.cuda.amp import autocast

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, get_out_coor, get_out_mask
from core.gdrn_modeling.models import GDRN_double_mask
from core.gdrn_modeling.models.heads.cpm_pnp_net import (
    CorrespondenceAwareMomentPnPNet,
)
from core.gdrn_modeling.models.model_utils import get_rot_mat
from core.gdrn_modeling.models.pose_from_pred import pose_from_pred
from core.gdrn_modeling.models.pose_from_pred_centroid_z import pose_from_pred_centroid_z
from core.gdrn_modeling.models.pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
from core.utils.my_checkpoint import MyCheckpointer
from lib.utils.mask_utils import cocosegm2mask
from research.oracle_diagnostic.oracle_utils import (
    depth_to_object_coordinates,
    normalized_image_points,
    prediction_valid_mask,
    sample_nearest,
)
from research.oracle_diagnostic.run_oracle_diagnostic import build_dataset_lookups
from research.pose_aggregation.run_diagnostic import configure, load_bop_target_counts
from research.pose_head_diagnostic.diagnostic_utils import (
    CONDITIONS,
    apply_intervention,
    response_metrics,
    rotation_geodesic_deg,
    tensor_state_sha256,
)
from research.pose_head_utilization.utilization_utils import metric_xyz_to_normalized


EXPECTED_WEIGHT_HASH = "bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c"
LAYERS = (
    "pnp_input",
    "conv_stage1",
    "conv_stage2",
    "conv_stage3",
    "flat_conv",
    "fc1",
    "fc2",
    "raw_rotation",
    "raw_translation",
    "final_pose",
)
CPM_LAYERS = (
    "pnp_input",
    "raw_moments",
    "scaled_moments",
    "fc1",
    "fc2",
    "raw_rotation",
    "raw_translation",
    "final_pose",
)
CPM_XYZ_ALPHA_CONDITIONS = {
    "gt_xyz_alpha_025": 0.25,
    "gt_xyz_alpha_050": 0.50,
    "gt_xyz_alpha_075": 0.75,
}
CPM_MOMENT_CONDITIONS = ("coverage_only", "cxu_null")
CPM_EXTRA_CONDITIONS = tuple(CPM_XYZ_ALPHA_CONDITIONS) + CPM_MOMENT_CONDITIONS
CPM_XYZ_REGION_2X2_CONDITIONS = (
    "pred_xyz_pred_region",
    "gt_xyz_pred_region",
    "pred_xyz_gt_region",
    "gt_xyz_gt_region",
)
CPM_XYZ_REGION_ALPHA_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)
CPM_XYZ_REGION_ALPHA_CONDITIONS = tuple(
    f"xyz_alpha_{int(round(alpha * 100)):03d}_{region_source}_region"
    for region_source in ("pred", "gt")
    for alpha in CPM_XYZ_REGION_ALPHA_VALUES
)
CPM_CONDITION_SETS = (
    "legacy",
    "cpm_xyz_region_2x2",
    "cpm_xyz_region_alpha_sweep",
)


def layers_for_model(model) -> tuple[str, ...]:
    """Return architecture-compatible diagnostic checkpoints."""

    return (
        CPM_LAYERS
        if isinstance(model.pnp_net, CorrespondenceAwareMomentPnPNet)
        else LAYERS
    )


def conditions_for_model(model, condition_set: str = "legacy") -> tuple[str, ...]:
    """Keep the frozen ConvPnP protocol unchanged and extend only CPM."""

    if condition_set not in CPM_CONDITION_SETS:
        raise ValueError(f"Unknown diagnostic condition set: {condition_set}")
    if condition_set != "legacy":
        if not isinstance(model.pnp_net, CorrespondenceAwareMomentPnPNet):
            raise ValueError(f"{condition_set} requires a CPM pose head")
        return (
            CPM_XYZ_REGION_2X2_CONDITIONS
            if condition_set == "cpm_xyz_region_2x2"
            else CPM_XYZ_REGION_ALPHA_CONDITIONS
        )
    if isinstance(model.pnp_net, CorrespondenceAwareMomentPnPNet):
        return CONDITIONS + CPM_EXTRA_CONDITIONS
    return CONDITIONS


def cpm_xyz_region_condition(condition: str) -> tuple[float, str]:
    """Resolve a preregistered XYZ alpha and Region source."""

    endpoints = {
        "pred_xyz_pred_region": (0.0, "pred"),
        "gt_xyz_pred_region": (1.0, "pred"),
        "pred_xyz_gt_region": (0.0, "gt"),
        "gt_xyz_gt_region": (1.0, "gt"),
    }
    if condition in endpoints:
        return endpoints[condition]
    for alpha in CPM_XYZ_REGION_ALPHA_VALUES:
        token = int(round(alpha * 100))
        for region_source in ("pred", "gt"):
            if condition == f"xyz_alpha_{token:03d}_{region_source}_region":
                return alpha, region_source
    raise ValueError(f"Unknown CPM XYZ/Region condition: {condition}")


def apply_cpm_xyz_region_intervention(
    xyz: np.ndarray,
    gt_xyz: np.ndarray,
    roi_2d: np.ndarray,
    pred_region: np.ndarray,
    gt_region: np.ndarray,
    support: np.ndarray,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Change only XYZ and/or Region on the frozen diagnostic support."""

    alpha, region_source = cpm_xyz_region_condition(condition)
    xyz_out = np.asarray(xyz).copy()
    gt = np.asarray(gt_xyz)
    mask = np.asarray(support, dtype=bool)
    xyz_out[mask] = (1.0 - alpha) * xyz_out[mask] + alpha * gt[mask]
    region_out = np.asarray(pred_region).copy()
    if region_source == "gt":
        region_out[mask] = np.asarray(gt_region)[mask]
    return xyz_out, np.asarray(roi_2d).copy(), region_out


def apply_cpm_diagnostic_intervention(
    xyz: np.ndarray,
    gt_xyz: np.ndarray,
    roi_2d: np.ndarray,
    region: np.ndarray,
    support: np.ndarray,
    condition: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    """Apply a standard input intervention or a CPM-only diagnostic condition.

    ``coverage_only`` and ``cxu_null`` deliberately leave the pixel inputs
    unchanged.  They are applied to the deterministic moment descriptor inside
    :func:`run_head_with_hooks` and are diagnostics, not trainable variants.
    """

    if condition in CPM_XYZ_ALPHA_CONDITIONS:
        xyz_out = np.asarray(xyz).copy()
        gt = np.asarray(gt_xyz)
        mask = np.asarray(support, dtype=bool)
        alpha = CPM_XYZ_ALPHA_CONDITIONS[condition]
        xyz_out[mask] = (1.0 - alpha) * xyz_out[mask] + alpha * gt[mask]
        return xyz_out, np.asarray(roi_2d).copy(), np.asarray(region).copy(), None
    if condition in CPM_MOMENT_CONDITIONS:
        return (
            np.asarray(xyz).copy(),
            np.asarray(roi_2d).copy(),
            np.asarray(region).copy(),
            condition,
        )
    xyz_out, roi_out, region_out = apply_intervention(
        xyz, gt_xyz, roi_2d, region, support, condition, seed
    )
    return xyz_out, roi_out, region_out, None


def apply_cpm_moment_intervention(
    pnp: CorrespondenceAwareMomentPnPNet,
    raw_descriptor: torch.Tensor,
    scaled_descriptor: torch.Tensor,
    condition: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a fixed descriptor-level CPM diagnostic without changing weights."""

    if condition is None:
        return raw_descriptor, scaled_descriptor
    if condition not in CPM_MOMENT_CONDITIONS:
        raise ValueError(f"Unknown CPM moment condition: {condition}")
    changed = raw_descriptor.clone()
    if condition == "coverage_only":
        changed[..., 1:21] = 0
    else:
        changed[..., 15:21] = 0
    return changed, pnp._apply_moment_scales(changed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dataset", default="lmo_bop_test")
    parser.add_argument("--bbox-source", choices=("gt",), default="gt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=20260730, type=int)
    parser.add_argument("--smoke-per-object", default=1, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--max-targets", default=0, type=int)
    parser.add_argument("--bop-eval", action="store_false", default=False)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sliced(batch: dict, key: str, index: int):
    value = batch.get(key)
    return None if value is None else value[index : index + 1]


def to_chw(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(
        np.ascontiguousarray(array.transpose(2, 0, 1)), device=device, dtype=dtype
    ).unsqueeze(0)


def pose_from_raw(model, cfg, raw_rotation, raw_translation, batch, index):
    pnp_cfg = cfg.MODEL.POSE_NET.PNP_NET
    rotation_matrix = get_rot_mat(raw_rotation, pnp_cfg.ROT_TYPE)
    common = {
        "eps": 1e-4,
        "is_allo": "allo" in pnp_cfg.ROT_TYPE,
        "is_train": False,
    }
    if pnp_cfg.TRANS_TYPE == "centroid_z":
        return pose_from_pred_centroid_z(
            rotation_matrix,
            pred_centroids=raw_translation[:, :2],
            pred_z_vals=raw_translation[:, 2:3],
            roi_cams=sliced(batch, "roi_cam", index),
            roi_centers=sliced(batch, "roi_center", index),
            resize_ratios=sliced(batch, "resize_ratio", index),
            roi_whs=sliced(batch, "roi_wh", index),
            z_type=pnp_cfg.Z_TYPE,
            **common,
        )
    if pnp_cfg.TRANS_TYPE == "centroid_z_abs":
        return pose_from_pred_centroid_z_abs(
            rotation_matrix,
            pred_centroids=raw_translation[:, :2],
            pred_z_vals=raw_translation[:, 2:3],
            roi_cams=sliced(batch, "roi_cam", index),
            **common,
        )
    if pnp_cfg.TRANS_TYPE == "trans":
        return pose_from_pred(rotation_matrix, raw_translation, **common)
    raise ValueError(f"Unsupported translation type: {pnp_cfg.TRANS_TYPE}")


def run_head_with_hooks(
    model,
    cfg,
    xyz: np.ndarray,
    roi_2d: np.ndarray,
    region: np.ndarray,
    visible_mask: np.ndarray,
    batch: dict,
    index: int,
    moment_condition: str | None = None,
) -> tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    pnp = model.pnp_net
    device = next(pnp.parameters()).device
    dtype = next(pnp.parameters()).dtype
    xyz_tensor = to_chw(xyz, device, dtype)
    roi_tensor = to_chw(roi_2d, device, dtype)
    region_tensor = to_chw(region, device, dtype)
    mask_array = np.asarray(visible_mask)
    if mask_array.ndim == 2:
        mask_array = mask_array[..., None]
    if mask_array.ndim != 3 or mask_array.shape[-1] != 1:
        raise ValueError(
            f"visible_mask must have shape HxW or HxWx1, got {mask_array.shape}"
        )
    mask_tensor = to_chw(mask_array, device, dtype)
    coor = torch.cat((xyz_tensor, roi_tensor), dim=1)
    extent = sliced(batch, "roi_extent", index).to(device=device, dtype=dtype)
    metric_xyz = (xyz_tensor - 0.5) * extent.view(1, 3, 1, 1)
    actual_input_parts = [metric_xyz, roi_tensor, region_tensor]
    if isinstance(pnp, CorrespondenceAwareMomentPnPNet):
        actual_input_parts.append(mask_tensor)
    actual_input = torch.cat(actual_input_parts, dim=1)

    if moment_condition is not None and moment_condition not in CPM_MOMENT_CONDITIONS:
        raise ValueError(f"Unknown CPM moment condition: {moment_condition}")
    if moment_condition is not None and not isinstance(
        pnp, CorrespondenceAwareMomentPnPNet
    ):
        raise ValueError("Moment-level interventions require a CPM pose head")

    if isinstance(pnp, CorrespondenceAwareMomentPnPNet):
        with autocast(enabled=bool(cfg.TEST.AMP_TEST)):
            encoding = pnp.encode_moments(
                coor.clone(),
                region=region_tensor,
                extents=extent,
                mask_attention=mask_tensor,
            )
            raw_descriptor, scaled_descriptor = apply_cpm_moment_intervention(
                pnp,
                encoding.raw_descriptor,
                encoding.scaled_descriptor,
                moment_condition,
            )
            fc1 = pnp.act(pnp.moment_fc1(scaled_descriptor.flatten(1)))
            fc2 = pnp.act(pnp.moment_fc2(fc1))
            raw_rotation = pnp.rotation_head(fc2)
            raw_translation = pnp.translation_head(fc2)
            final_rotation, final_translation = pose_from_raw(
                model, cfg, raw_rotation, raw_translation, batch, index
            )
        activations = {
            "pnp_input": actual_input.detach().cpu(),
            "raw_moments": raw_descriptor.detach().cpu(),
            "scaled_moments": scaled_descriptor.detach().cpu(),
            "fc1": fc1.detach().cpu(),
            "fc2": fc2.detach().cpu(),
            "raw_rotation": raw_rotation.detach().cpu(),
            "raw_translation": raw_translation.detach().cpu(),
            "final_pose": torch.cat(
                (
                    final_rotation.detach().cpu().flatten(1),
                    final_translation.detach().cpu(),
                ),
                dim=1,
            ),
        }
        return (
            activations,
            final_rotation[0].detach().float().cpu().numpy(),
            final_translation[0].detach().float().cpu().numpy(),
        )

    conv_outputs: List[torch.Tensor] = []
    fc_outputs: List[torch.Tensor] = []
    direct: Dict[str, torch.Tensor] = {}

    def append_conv(_module, _inputs, output):
        conv_outputs.append(output.detach().cpu())

    def append_fc(_module, _inputs, output):
        fc_outputs.append(output.detach().cpu())

    def save_direct(name):
        def hook(_module, _inputs, output):
            direct[name] = output.detach().cpu()
        return hook

    handles = [
        # The same activation module is shared by all three convolution stages.
        pnp.features[2].register_forward_hook(append_conv),
        pnp.act.register_forward_hook(append_fc),
        pnp.fc_r.register_forward_hook(save_direct("raw_rotation")),
        pnp.fc_t.register_forward_hook(save_direct("raw_translation")),
    ]
    try:
        with autocast(enabled=bool(cfg.TEST.AMP_TEST)):
            raw_rotation, raw_translation = pnp(
                coor.clone(), region=region_tensor, extents=extent, mask_attention=None
            )
            final_rotation, final_translation = pose_from_raw(
                model, cfg, raw_rotation, raw_translation, batch, index
            )
    finally:
        for handle in handles:
            handle.remove()
    if len(conv_outputs) != 3 or len(fc_outputs) != 2:
        raise RuntimeError(
            f"Unexpected hook calls: conv={len(conv_outputs)}, fc={len(fc_outputs)}"
        )
    activations = {
        "pnp_input": actual_input.detach().cpu(),
        "conv_stage1": conv_outputs[0],
        "conv_stage2": conv_outputs[1],
        "conv_stage3": conv_outputs[2],
        "flat_conv": conv_outputs[2].flatten(1),
        "fc1": fc_outputs[0],
        "fc2": fc_outputs[1],
        "raw_rotation": direct["raw_rotation"],
        "raw_translation": direct["raw_translation"],
        "final_pose": torch.cat(
            (final_rotation.detach().cpu().flatten(1), final_translation.detach().cpu()), dim=1
        ),
    }
    return (
        activations,
        final_rotation[0].detach().float().cpu().numpy(),
        final_translation[0].detach().float().cpu().numpy(),
    )


def write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    if args.device != "cpu":
        raise ValueError("This preliminary smoke protocol is intentionally CPU-only")
    if args.smoke_per_object != 1 or args.max_targets:
        raise ValueError("This protocol requires --smoke-per-object 1 and no max-targets")
    weight_hash = sha256(args.weights)
    if weight_hash != EXPECTED_WEIGHT_HASH:
        raise RuntimeError(f"Unexpected checkpoint SHA-256: {weight_hash}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    cfg = configure(args)
    register_datasets_in_cfg(cfg)
    metadata = MetadataCatalog.get(args.dataset)
    data_ref = ref.__dict__[metadata.ref_key]
    object_ids = [int(data_ref.obj2id[name]) for name in metadata.objs]
    images, gt_lookup = build_dataset_lookups(args.dataset)
    target_counts = load_bop_target_counts()
    occurrences: Dict[tuple, int] = {}
    model, optimizer = GDRN_double_mask.build_model_optimizer(cfg, is_test=True)
    if optimizer is not None:
        raise RuntimeError("Frozen smoke protocol unexpectedly constructed an optimizer")
    MyCheckpointer(model, save_dir=str(args.output_dir), prefix_to_remove="_module.").resume_or_load(
        str(args.weights), resume=False
    )
    model.eval()
    layers = layers_for_model(model)
    before_hash = tensor_state_sha256(model.state_dict())
    loader = build_gdrn_test_loader(
        cfg, args.dataset, train_objs=metadata.objs, batch_size=1
    )

    rows: List[dict] = []
    selected: List[dict] = []
    shapes: Dict[str, list[int]] = {}
    counts = {obj_id: 0 for obj_id in object_ids}
    depth_cache: Dict[str, np.ndarray] = {}
    threshold = float(cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST)
    max_rotation_reentry = 0.0
    max_translation_reentry = 0.0

    with inference_context(model), torch.no_grad():
        for inputs in loader:
            if all(counts[obj_id] == 1 for obj_id in object_ids):
                break
            if not isinstance(inputs, list):
                inputs = [inputs]
            batch = batch_data(cfg, inputs, device=args.device, phase="test")
            with autocast(enabled=False):
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
            xyz_batch = get_out_coor(
                cfg, output["coor_x"], output["coor_y"], output["coor_z"]
            ).detach().float().cpu().numpy()
            visible_batch = get_out_mask(cfg, output["mask"]).detach().float().cpu().numpy()
            region_batch = F.softmax(output["region"][:, 1:], dim=1).detach().float().cpu().numpy()
            original_rotations = output["rot"].detach().float().cpu().numpy()
            original_translations = output["trans"].detach().float().cpu().numpy()

            flat_index = -1
            for input_item in inputs:
                for local_index in range(len(input_item["roi_img"])):
                    flat_index += 1
                    class_index = int(input_item["roi_cls"][local_index])
                    obj_id = object_ids[class_index]
                    scene_im_id = input_item["scene_im_id"][local_index]
                    key = (scene_im_id, obj_id)
                    occurrence = occurrences.get(key, 0)
                    if occurrence >= target_counts.get(key, 0):
                        continue
                    occurrences[key] = occurrence + 1
                    if counts[obj_id]:
                        continue
                    instance_id = int(input_item["inst_id"][local_index])
                    gt = gt_lookup[(scene_im_id, instance_id)]
                    image_record = images[scene_im_id]
                    height = int(input_item["im_H"][local_index])
                    width = int(input_item["im_W"][local_index])
                    camera = input_item["cam"][local_index].cpu().numpy()
                    extent = input_item["roi_extent"][local_index].cpu().numpy()
                    roi_2d = (
                        input_item["roi_coord_2d"][local_index].cpu().numpy().transpose(1, 2, 0)
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
                    xyz = xyz_batch[flat_index].transpose(1, 2, 0)
                    predicted_support = prediction_valid_mask(
                        xyz, visible_batch[flat_index, 0], extent, threshold
                    )
                    support = predicted_support & gt_visible
                    gt_xyz = metric_xyz_to_normalized(gt_xyz_m, extent)
                    region = region_batch[flat_index].transpose(1, 2, 0)
                    baseline_activations = None
                    baseline_rotation = None
                    baseline_translation = None
                    instance_seed = args.seed + len(selected) * 100
                    for condition_index, condition in enumerate(CONDITIONS):
                        xyz_i, roi_i, region_i = apply_intervention(
                            xyz, gt_xyz, roi_2d, region, support, condition,
                            instance_seed + condition_index,
                        )
                        activations, rotation, translation = run_head_with_hooks(
                            model,
                            cfg,
                            xyz_i,
                            roi_i,
                            region_i,
                            visible_batch[flat_index, 0],
                            batch,
                            flat_index,
                        )
                        if condition == "baseline":
                            baseline_activations = activations
                            baseline_rotation = rotation
                            baseline_translation = translation
                            max_rotation_reentry = max(
                                max_rotation_reentry,
                                float(np.max(np.abs(rotation - original_rotations[flat_index]))),
                            )
                            max_translation_reentry = max(
                                max_translation_reentry,
                                float(
                                    np.max(
                                        np.abs(translation - original_translations[flat_index])
                                    )
                                ),
                            )
                        record = {
                            "scene_id": int(scene_im_id.split("/")[0]),
                            "im_id": int(scene_im_id.split("/")[1]),
                            "instance_id": instance_id,
                            "obj_id": obj_id,
                            "obj_name": data_ref.id2obj[obj_id],
                            "condition": condition,
                            "support_points": int(np.count_nonzero(support)),
                            "rotation_delta_deg": rotation_geodesic_deg(
                                rotation, baseline_rotation
                            ),
                            "translation_delta_mm": float(
                                np.linalg.norm(translation - baseline_translation) * 1000.0
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
                            "rotation_error_gt_deg": rotation_geodesic_deg(
                                rotation, rotation_gt
                            ),
                            "translation_error_gt_mm": float(
                                np.linalg.norm(translation - translation_gt) * 1000.0
                            ),
                        }
                        for layer in layers:
                            shapes.setdefault(layer, list(activations[layer].shape))
                            metrics = response_metrics(
                                baseline_activations[layer], activations[layer]
                            )
                            for metric, value in metrics.items():
                                record[f"{layer}_{metric}"] = value
                        rows.append(record)
                    selected.append(
                        {
                            "scene_im_id": scene_im_id,
                            "instance_id": instance_id,
                            "obj_id": obj_id,
                            "obj_name": data_ref.id2obj[obj_id],
                            "support_points": int(np.count_nonzero(support)),
                        }
                    )
                    counts[obj_id] += 1

    after_hash = tensor_state_sha256(model.state_dict())
    all_numeric_finite = all(
        np.isfinite(value)
        for row in rows
        for key, value in row.items()
        if key not in {"condition", "obj_name"} and isinstance(value, (int, float))
    )
    acceptance = {
        "checkpoint_hash_matches": weight_hash == EXPECTED_WEIGHT_HASH,
        "objects": len(selected),
        "records": len(rows),
        "expected_objects": len(object_ids),
        "expected_records": len(object_ids) * len(CONDITIONS),
        "all_numeric_finite": bool(all_numeric_finite),
        "state_unchanged": before_hash == after_hash,
        "max_baseline_rotation_abs_error": max_rotation_reentry,
        "max_baseline_translation_abs_error": max_translation_reentry,
        "baseline_reentry_tolerance": 1e-6,
    }
    acceptance["passed"] = bool(
        acceptance["checkpoint_hash_matches"]
        and len(selected) == len(object_ids)
        and len(rows) == len(object_ids) * len(CONDITIONS)
        and all_numeric_finite
        and before_hash == after_hash
        and max_rotation_reentry <= 1e-6
        and max_translation_reentry <= 1e-6
    )
    condition_summary = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        condition_summary[condition] = {
            "instances": len(condition_rows),
            "mean_rotation_delta_deg": float(
                np.mean([row["rotation_delta_deg"] for row in condition_rows])
            ),
            "mean_translation_delta_mm": float(
                np.mean([row["translation_delta_mm"] for row in condition_rows])
            ),
            "mean_relative_l2_by_layer": {
                layer: float(
                    np.mean(
                        [row[f"{layer}_relative_l2"] for row in condition_rows]
                    )
                )
                for layer in layers
            },
        }
    architecture = {
        "pnp_input_channels": 69,
        "components": {"xyz": 3, "roi_2d_absolute": 2, "region_probabilities": 64},
        "layers": shapes,
        "rotation_representation": str(cfg.MODEL.POSE_NET.PNP_NET.ROT_TYPE),
        "translation_representation": str(cfg.MODEL.POSE_NET.PNP_NET.TRANS_TYPE),
    }
    protocol = {
        "status": "preliminary_cpu_smoke",
        "formal_experiment": False,
        "seed": args.seed,
        "conditions": CONDITIONS,
        "layers": layers,
        "checkpoint": str(args.weights),
        "checkpoint_sha256": weight_hash,
        "device": args.device,
        "selected_targets": selected,
        "fixed_support": "predicted_visible_support AND GT-visible/depth-valid support",
        "parameter_updates": 0,
        "optimizer_created": optimizer is not None,
    }
    for name, value in (
        ("architecture.json", architecture),
        ("protocol.json", protocol),
        (
            "information_flow_smoke.json",
            {"acceptance": acceptance, "condition_summary": condition_summary},
        ),
    ):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
    print(json.dumps(acceptance, indent=2))
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
