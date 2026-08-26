from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .common import CapturedPoseCall, PosePrediction


def unwrap_model(model: nn.Module) -> nn.Module:
    for attr in ("module", "_module"):
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            if isinstance(candidate, nn.Module):
                return candidate
    return model


def make_model_kwargs(batch: Dict[str, Any], do_loss: bool) -> Dict[str, Any]:
    return dict(
        gt_xyz=batch.get("roi_xyz"),
        gt_xyz_bin=batch.get("roi_xyz_bin"),
        gt_mask_trunc=batch.get("roi_mask_trunc"),
        gt_mask_visib=batch.get("roi_mask_visib"),
        gt_mask_full=batch.get("roi_mask_full"),
        gt_mask_obj=batch.get("roi_mask_obj"),
        gt_region=batch.get("roi_region"),
        gt_ego_rot=batch.get("ego_rot"),
        gt_trans=batch.get("trans"),
        gt_trans_ratio=batch.get("roi_trans_ratio"),
        gt_points=batch.get("roi_points"),
        sym_infos=batch.get("sym_info"),
        roi_classes=batch.get("roi_cls"),
        roi_cams=batch.get("roi_cam"),
        roi_whs=batch.get("roi_wh"),
        roi_centers=batch.get("roi_center"),
        resize_ratios=batch.get("resize_ratio"),
        roi_coord_2d=batch.get("roi_coord_2d"),
        roi_coord_2d_rel=batch.get("roi_coord_2d_rel"),
        roi_extents=batch.get("roi_extent"),
        do_loss=do_loss,
    )


def model_input_from_batch(cfg, batch: Dict[str, Any]) -> torch.Tensor:
    if cfg.INPUT.WITH_DEPTH:
        return torch.cat([batch["roi_img"], batch["roi_depth"]], dim=1)
    return batch["roi_img"]


class _HeadForwardCapture:
    def __init__(self, head: nn.Module) -> None:
        self.head = head
        self.call: Optional[CapturedPoseCall] = None
        self._orig = None

    def __enter__(self):
        self._orig = self.head.forward
        owner = self

        def wrapped(coor_feat, *args, **kwargs):
            region = kwargs.get("region", args[0] if len(args) > 0 else None)
            extents = kwargs.get("extents", args[1] if len(args) > 1 else None)
            mask_attention = kwargs.get("mask_attention", args[2] if len(args) > 2 else None)
            out = owner._orig(coor_feat, *args, **kwargs)
            raw_rot, raw_t = out
            owner.call = CapturedPoseCall(
                coor_feat=coor_feat,
                region=region,
                extents=extents,
                mask_attention=mask_attention,
                raw_rot=raw_rot,
                raw_t=raw_t,
            )
            return out

        self.head.forward = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        self.head.forward = self._orig
        return False


def capture_model_pose_call(model: nn.Module, cfg, batch: Dict[str, Any], do_loss: bool = False):
    base = unwrap_model(model)
    head = base.pnp_net
    inp = model_input_from_batch(cfg, batch)
    with _HeadForwardCapture(head) as capture:
        output = base(inp, **make_model_kwargs(batch, do_loss=do_loss))
    if capture.call is None:
        raise RuntimeError("Pose head forward was not reached; cannot capture pose inputs.")
    return output, capture.call


def decode_raw_pose(cfg, raw_rot: torch.Tensor, raw_t: torch.Tensor, batch: Dict[str, Any], is_train: bool = False) -> PosePrediction:
    from core.gdrn_modeling.models.model_utils import get_rot_mat
    from core.gdrn_modeling.models.pose_from_pred import pose_from_pred
    from core.gdrn_modeling.models.pose_from_pred_centroid_z import pose_from_pred_centroid_z
    from core.gdrn_modeling.models.pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs

    pnp_cfg = cfg.MODEL.POSE_NET.PNP_NET
    rot_type = pnp_cfg.ROT_TYPE
    pred_rot_m = get_rot_mat(raw_rot, rot_type)
    if pnp_cfg.TRANS_TYPE == "centroid_z":
        pred_rot, pred_trans = pose_from_pred_centroid_z(
            pred_rot_m,
            pred_centroids=raw_t[:, :2],
            pred_z_vals=raw_t[:, 2:3],
            roi_cams=batch["roi_cam"],
            roi_centers=batch["roi_center"],
            resize_ratios=batch["resize_ratio"],
            roi_whs=batch["roi_wh"],
            eps=1e-4,
            is_allo="allo" in rot_type,
            z_type=pnp_cfg.Z_TYPE,
            is_train=is_train,
        )
    elif pnp_cfg.TRANS_TYPE == "centroid_z_abs":
        pred_rot, pred_trans = pose_from_pred_centroid_z_abs(
            pred_rot_m,
            pred_centroids=raw_t[:, :2],
            pred_z_vals=raw_t[:, 2:3],
            roi_cams=batch["roi_cam"],
            eps=1e-4,
            is_allo="allo" in rot_type,
            is_train=is_train,
        )
    elif pnp_cfg.TRANS_TYPE == "trans":
        pred_rot, pred_trans = pose_from_pred(
            pred_rot_m,
            raw_t,
            eps=1e-4,
            is_allo="allo" in rot_type,
            is_train=is_train,
        )
    else:
        raise ValueError(f"Unsupported TRANS_TYPE={pnp_cfg.TRANS_TYPE}")
    return PosePrediction(rot=pred_rot, trans=pred_trans, raw_rot=raw_rot, raw_t=raw_t)


def call_head(head: nn.Module, call: CapturedPoseCall, *, coor_feat=None, region=None, extents=None, mask_attention=None):
    return head(
        call.coor_feat if coor_feat is None else coor_feat,
        region=call.region if region is None else region,
        extents=call.extents if extents is None else extents,
        mask_attention=call.mask_attention if mask_attention is None else mask_attention,
    )


@contextlib.contextmanager
def temporary_scalar(parameter: torch.Tensor, value: float):
    old = parameter.detach().clone()
    with torch.no_grad():
        parameter.fill_(float(value))
    try:
        yield
    finally:
        with torch.no_grad():
            parameter.copy_(old)


@contextlib.contextmanager
def linear_input_transform(module: nn.Module, transform):
    def hook(_module, args):
        if not args:
            return None
        changed = transform(args[0])
        return (changed,) + tuple(args[1:])

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()
