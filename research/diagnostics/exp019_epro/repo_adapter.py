"""Current-checkout adapter for the frozen official GDRNPP producer.

The fixed support and intervention semantics are restored from commit d702030's
EXP002/EXP004 implementation.  This module intentionally does not modify core
model, mapper, evaluator, or checkpoint behavior.
"""

from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation.evaluator import inference_context
from mmcv import Config

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
from lib.utils.mask_utils import cocosegm2mask

from .config import EXPECTED_LMO_TARGETS, EXPECTED_OFFICIAL_WEIGHT_SHA256
from .correspondence import project_points
from .types import DiagnosticSample, PoseResult


ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Context:
    cfg: Config
    model: torch.nn.Module
    loader: object
    dataset: str
    device: str
    checkpoint: Path
    limit: Optional[int]
    object_ids: list
    images: Dict[str, dict]
    annotations: Dict[tuple, dict]
    target_counts: Dict[tuple, int]
    current: Dict[str, dict] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure(cfg_path: Path, checkpoint: Path, device: str) -> Config:
    cfg = Config.fromfile(str(cfg_path))
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.LOAD_DETS_TEST = False
    cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.pretrained = False
    cfg.DATASETS.TRAIN = ("lmo_bop_test",)
    cfg.DATASETS.TEST = ("lmo_bop_test",)
    cfg.DATASETS.DET_FILES_TEST = ()
    cfg.TEST.TEST_BBOX_TYPE = "gt"
    cfg.TEST.USE_PNP = False
    # Current GDRN test forward returns dense producer tensors only for an
    # evaluator/serialization consumer.  This flag exposes them without
    # enabling evaluator-side PnP or changing the computed network pose.
    cfg.TEST.SAVE_RESULTS_ONLY = True
    cfg.TEST.AMP_TEST = False
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.PERSISTENT_WORKERS = False
    cfg.OUTPUT_DIR = str(ROOT / "output" / "experiments" / "EXP019-config-only")
    optimizer_cfg = cfg.SOLVER.OPTIMIZER_CFG
    if isinstance(optimizer_cfg, str):
        optimizer_cfg = ast.literal_eval(optimizer_cfg)
        cfg.SOLVER.OPTIMIZER_CFG = optimizer_cfg
    cfg.SOLVER.BASE_LR = float(optimizer_cfg.get("lr", 1e-4))
    cfg.SOLVER.MOMENTUM = float(optimizer_cfg.get("momentum", 0.9))
    cfg.SOLVER.WEIGHT_DECAY = float(optimizer_cfg.get("weight_decay", 1e-4))
    corrector = cfg.MODEL.POSE_NET.get("POSE_CORRECTOR", {})
    if corrector:
        corrector["ENABLED"] = False
    pnp_cfg = cfg.MODEL.POSE_NET.PNP_NET
    if pnp_cfg.INIT_CFG.type != "ConvPnPNet":
        raise RuntimeError("EXP019 requires the official ConvPnPNet Patch-PnP head")
    if not pnp_cfg.WITH_2D_COORD or pnp_cfg.COORD_2D_TYPE != "abs":
        raise RuntimeError("EXP019 requires official absolute ROI2D Patch-PnP input")
    if not pnp_cfg.REGION_ATTENTION:
        raise RuntimeError("EXP019 must preserve official Region attention")
    if cfg.TEST.USE_PNP or not cfg.TEST.SAVE_RESULTS_ONLY:
        raise RuntimeError("EXP019 must expose dense tensors without evaluator-side PnP")
    return cfg


def _dataset_lookups(dataset: str):
    images, annotations = {}, {}
    for image in DatasetCatalog.get(dataset):
        images[image["scene_im_id"]] = image
        for instance_index, annotation in enumerate(image.get("annotations", [])):
            annotations[(image["scene_im_id"], instance_index)] = annotation
    return images, annotations


def _target_counts() -> Dict[tuple, int]:
    import json

    path = ROOT / "datasets" / "BOP_DATASETS" / "lmo" / "test_targets_bop19.json"
    with path.open(encoding="utf-8") as stream:
        targets = json.load(stream)
    counts = {
        (f"{int(item['scene_id'])}/{int(item['im_id'])}", int(item["obj_id"])): int(item["inst_count"])
        for item in targets
    }
    if sum(counts.values()) != EXPECTED_LMO_TARGETS:
        raise RuntimeError(f"LM-O targets total {sum(counts.values())}, expected {EXPECTED_LMO_TARGETS}")
    return counts


def build_context(cfg_path: str, checkpoint: str, device: str, limit: Optional[int] = None):
    cfg_path = (ROOT / cfg_path).resolve() if not Path(cfg_path).is_absolute() else Path(cfg_path)
    checkpoint = (ROOT / checkpoint).resolve() if not Path(checkpoint).is_absolute() else Path(checkpoint)
    required = [
        cfg_path,
        checkpoint,
        ROOT / "datasets" / "BOP_DATASETS" / "lmo" / "test_targets_bop19.json",
        ROOT / "datasets" / "BOP_DATASETS" / "lmo" / "models_eval",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing EXP019 assets:\n" + "\n".join(missing))
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{device} requested but CUDA is unavailable")
    if limit is not None and int(limit) <= 0:
        limit = None
    weight_hash = _sha256(checkpoint)
    if weight_hash != EXPECTED_OFFICIAL_WEIGHT_SHA256:
        raise RuntimeError(f"Unexpected official checkpoint SHA-256: {weight_hash}")
    cfg = _configure(cfg_path, checkpoint, device)
    register_datasets_in_cfg(cfg)
    metadata = MetadataCatalog.get("lmo_bop_test")
    data_ref = ref.__dict__[metadata.ref_key]
    object_ids = [int(data_ref.obj2id[name]) for name in metadata.objs]
    images, annotations = _dataset_lookups("lmo_bop_test")
    model, _ = GDRN_double_mask.build_model_optimizer(cfg, is_test=True)
    MyCheckpointer(
        model, save_dir=str(checkpoint.parent), prefix_to_remove="_module."
    ).resume_or_load(str(checkpoint), resume=False)
    model.eval()
    if getattr(model, "pose_corrector", None) is not None:
        raise RuntimeError("EXP018 pose correction must be absent in EXP019")
    loader = build_gdrn_test_loader(
        cfg, "lmo_bop_test", train_objs=metadata.objs, batch_size=1
    )
    return Context(
        cfg=cfg,
        model=model,
        loader=loader,
        dataset="lmo_bop_test",
        device=device,
        checkpoint=checkpoint,
        limit=limit,
        object_ids=object_ids,
        images=images,
        annotations=annotations,
        target_counts=_target_counts(),
    )


def _sample_nearest(image: np.ndarray, points: np.ndarray):
    height, width = image.shape[:2]
    finite = np.isfinite(points).all(axis=2)
    safe = np.where(np.isfinite(points), points, 0.0)
    x = np.rint(safe[..., 0]).astype(np.int64)
    y = np.rint(safe[..., 1]).astype(np.int64)
    valid = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    sampled = np.zeros(points.shape[:2] + image.shape[2:], dtype=image.dtype)
    sampled[valid] = image[y[valid], x[valid]]
    return sampled, valid


def _depth_to_object(depth_m, points, K, R, t):
    depth, in_image = _sample_nearest(depth_m, points)
    z = np.asarray(depth, dtype=np.float64)
    x = (points[..., 0] - K[0, 2]) * z / K[0, 0]
    y = (points[..., 1] - K[1, 2]) * z / K[1, 1]
    camera = np.stack((x, y, z), axis=-1)
    obj = (camera - np.asarray(t).reshape(1, 1, 3)) @ np.asarray(R).reshape(3, 3)
    valid = in_image & np.isfinite(obj).all(axis=2) & np.isfinite(z) & (z > 0)
    obj[~valid] = 0.0
    return obj, valid


def _prediction_valid_mask(xyz_norm, mask_probability, extent, threshold):
    xyz_m = (np.asarray(xyz_norm, dtype=np.float64) - 0.5) * np.asarray(extent).reshape(1, 1, 3)
    mask = np.asarray(mask_probability, dtype=np.float64).squeeze()
    epsilon = 1e-4 * np.asarray(extent).reshape(1, 1, 3)
    return (
        np.isfinite(xyz_m).all(axis=2)
        & np.isfinite(mask)
        & (np.abs(xyz_m) > epsilon).all(axis=2)
        & (mask > float(threshold))
    )


def iter_samples(context: Context) -> Iterator[DiagnosticSample]:
    threshold = float(context.cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST)
    occurrences: Dict[tuple, int] = {}
    selected_per_object = {obj_id: 0 for obj_id in context.object_ids}
    quota = math.ceil(context.limit / len(context.object_ids)) if context.limit and context.limit <= 32 else None
    depth_cache: Dict[str, np.ndarray] = {}
    produced = 0

    with inference_context(context.model), torch.no_grad():
        for inputs in context.loader:
            if not isinstance(inputs, list):
                inputs = [inputs]
            batch = batch_data(context.cfg, inputs, device=context.device, phase="test")
            output = context.model(
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
                context.cfg,
                output["coor_x"].detach(),
                output["coor_y"].detach(),
                output["coor_z"].detach(),
            ).float().cpu().numpy()
            visible_batch = get_out_mask(context.cfg, output["mask"].detach()).float().cpu().numpy()
            region_batch = output["region"].detach().float().cpu().numpy()
            flat_index = -1
            for input_item in inputs:
                for local_index in range(len(input_item["roi_img"])):
                    flat_index += 1
                    class_index = int(input_item["roi_cls"][local_index])
                    obj_id = context.object_ids[class_index]
                    scene_im_id = input_item["scene_im_id"][local_index]
                    target_key = (scene_im_id, obj_id)
                    occurrence = occurrences.get(target_key, 0)
                    if occurrence >= context.target_counts.get(target_key, 0):
                        continue
                    occurrences[target_key] = occurrence + 1
                    if quota is not None and selected_per_object[obj_id] >= quota:
                        continue
                    instance_id = int(input_item["inst_id"][local_index])
                    gt = context.annotations[(scene_im_id, instance_id)]
                    image = context.images[scene_im_id]
                    scene_id, im_id = map(int, scene_im_id.split("/"))
                    K = input_item["cam"][local_index].cpu().numpy().astype(np.float64)
                    height = int(input_item["im_H"][local_index])
                    width = int(input_item["im_W"][local_index])
                    extent = input_item["roi_extent"][local_index].cpu().numpy()
                    roi2d = input_item["roi_coord_2d"][local_index].cpu().numpy()
                    image_points = roi2d.transpose(1, 2, 0).astype(np.float64)
                    image_points *= np.array([width, height], dtype=np.float64)
                    if scene_im_id not in depth_cache:
                        depth_path = (ROOT / image["depth_file"]).resolve()
                        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                        if raw_depth is None:
                            raise FileNotFoundError(depth_path)
                        depth_cache[scene_im_id] = raw_depth.astype(np.float64) / float(image["depth_factor"])
                    visible_full = cocosegm2mask(gt["segmentation"], height, width).astype(bool)
                    visible_sampled, in_image = _sample_nearest(visible_full, image_points)
                    R_gt = np.asarray(gt["pose"][:, :3], dtype=np.float64)
                    t_gt = np.asarray(gt["pose"][:, 3], dtype=np.float64)
                    gt_xyz_m, depth_valid = _depth_to_object(
                        depth_cache[scene_im_id], image_points, K, R_gt, t_gt
                    )
                    gt_visible = visible_sampled.astype(bool) & in_image & depth_valid
                    pred_xyz_hwc = xyz_batch[flat_index].transpose(1, 2, 0)
                    mask_prob = visible_batch[flat_index, 0]
                    pred_visible = _prediction_valid_mask(
                        pred_xyz_hwc, mask_prob, extent, threshold
                    )
                    support = pred_visible & gt_visible
                    gt_xyz_norm = (gt_xyz_m / extent.reshape(1, 1, 3) + 0.5).transpose(2, 0, 1)
                    reproj = project_points(gt_xyz_m[support], R_gt, t_gt, K)
                    reproj_error = (
                        np.linalg.norm(reproj - image_points[support], axis=1)
                        if np.count_nonzero(support)
                        else np.empty(0)
                    )
                    max_reprojection = float(reproj_error.max()) if len(reproj_error) else float("nan")
                    if len(reproj_error) and max_reprojection > 0.75:
                        raise RuntimeError(
                            f"{scene_im_id}/{instance_id}: GT XYZ reprojection {max_reprojection:.4f}px"
                        )
                    target_id = f"{scene_id:06d}/{im_id:06d}/{obj_id:06d}/{instance_id:03d}"
                    context.current[target_id] = {
                        "batch": batch,
                        "output": output,
                        "index": flat_index,
                    }
                    sample = DiagnosticSample(
                        target_id=target_id,
                        scene_id=scene_id,
                        im_id=im_id,
                        instance_id=instance_id,
                        obj_id=obj_id,
                        pred_xyz_norm=np.ascontiguousarray(xyz_batch[flat_index]),
                        gt_xyz_norm=np.ascontiguousarray(gt_xyz_norm, dtype=np.float64),
                        roi2d_norm=np.ascontiguousarray(roi2d),
                        extent_m=np.ascontiguousarray(extent),
                        K=np.ascontiguousarray(K),
                        image_hw=np.array([height, width], dtype=np.float32),
                        support_mask=np.ascontiguousarray(support),
                        mask_prob=np.ascontiguousarray(mask_prob),
                        region=np.ascontiguousarray(region_batch[flat_index]),
                        metadata={
                            "gt_xyz_max_reprojection_px": max_reprojection,
                            "pred_visible_points": int(np.count_nonzero(pred_visible)),
                            "gt_visible_points": int(np.count_nonzero(gt_visible)),
                        },
                    )
                    yield sample
                    context.current.pop(target_id, None)
                    produced += 1
                    selected_per_object[obj_id] += 1
                    if context.limit and produced >= context.limit:
                        return


def _slice(batch, key, index):
    value = batch.get(key)
    return None if value is None else value[index : index + 1]


def solve_patch_pnp(context: Context, sample: DiagnosticSample, xyz_alpha_norm: np.ndarray):
    state = context.current.get(sample.target_id)
    if state is None:
        raise RuntimeError("solve_patch_pnp must be called before advancing iter_samples")
    output, batch, index = state["output"], state["batch"], state["index"]
    pnp_cfg = context.cfg.MODEL.POSE_NET.PNP_NET
    net_cfg = context.cfg.MODEL.POSE_NET
    device, dtype = output["coor_x"].device, output["coor_x"].dtype
    xyz = torch.as_tensor(
        np.ascontiguousarray(xyz_alpha_norm), device=device, dtype=dtype
    ).unsqueeze(0)
    coor_feat = torch.cat((xyz, _slice(batch, "roi_coord_2d", index).to(dtype=dtype)), dim=1)
    region = F.softmax(output["region"][index : index + 1, 1:], dim=1)
    mask_attention = None
    if pnp_cfg.MASK_ATTENTION != "none":
        mask_attention = get_mask_prob(
            output["mask"][index : index + 1], net_cfg.LOSS_CFG.MASK_LOSS_TYPE
        )
    with torch.no_grad():
        raw_R, raw_t = context.model.pnp_net(
            coor_feat,
            region=region,
            extents=_slice(batch, "roi_extent", index),
            mask_attention=mask_attention,
        )
        rot_m = get_rot_mat(raw_R, pnp_cfg.ROT_TYPE)
        if pnp_cfg.TRANS_TYPE == "centroid_z":
            R, t = pose_from_pred_centroid_z(
                rot_m,
                pred_centroids=raw_t[:, :2],
                pred_z_vals=raw_t[:, 2:3],
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
            R, t = pose_from_pred_centroid_z_abs(
                rot_m,
                pred_centroids=raw_t[:, :2],
                pred_z_vals=raw_t[:, 2:3],
                roi_cams=_slice(batch, "roi_cam", index),
                eps=1e-4,
                is_allo="allo" in pnp_cfg.ROT_TYPE,
                is_train=False,
            )
        elif pnp_cfg.TRANS_TYPE == "trans":
            R, t = pose_from_pred(
                rot_m, raw_t, eps=1e-4, is_allo="allo" in pnp_cfg.ROT_TYPE, is_train=False
            )
        else:
            raise ValueError(f"Unsupported TRANS_TYPE={pnp_cfg.TRANS_TYPE}")
    rerun_R = R[0].detach().float().cpu().numpy()
    rerun_t = t[0].detach().float().cpu().numpy()
    message = ""
    if np.array_equal(np.asarray(xyz_alpha_norm), sample.pred_xyz_norm):
        original_R = output["rot"][index].detach().float().cpu().numpy()
        original_t = output["trans"][index].detach().float().cpu().numpy()
        message = (
            f"alpha0_reentry_R={np.max(np.abs(rerun_R-original_R)):.9g};"
            f"t={np.max(np.abs(rerun_t-original_t)):.9g}"
        )
        rerun_R, rerun_t = original_R, original_t
    return PoseResult(
        R=np.asarray(rerun_R, dtype=np.float64),
        t=np.asarray(rerun_t, dtype=np.float64),
        success=bool(np.isfinite(rerun_R).all() and np.isfinite(rerun_t).all()),
        solver="official_patch_pnp",
        message=message,
        num_points=int(np.count_nonzero(sample.support_mask)),
    )
