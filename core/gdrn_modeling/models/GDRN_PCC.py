"""EXP022 frozen-visual-backbone model without legacy decoder or pose head."""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn

from core.utils.solver_utils import build_optimizer_with_params

from .GDRN_double_mask import get_backbone_init_args
from .heads.progressive_pcc_head import ProgressivePCCHead
from .net_factory import BACKBONES


class GDRN_PCC(nn.Module):
    def __init__(self, backbone: nn.Module, pcc_head: ProgressivePCCHead):
        super().__init__()
        self.backbone = backbone
        self.pcc_head = pcc_head

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x, roi_classes=None, gt_xyz=None, gt_mask_visib=None,
                do_loss: bool = False, return_pcc_debug: bool = False, **_unused):
        if roi_classes is None:
            raise ValueError("EXP022 requires roi_classes")
        with torch.no_grad():
            backbone_feature = self.backbone(x)
        if isinstance(backbone_feature, (tuple, list)):
            backbone_feature = backbone_feature[0]
        if do_loss:
            losses, stats = self.pcc_head(
                backbone_feature, roi_classes, gt_xyz_norm=gt_xyz,
                gt_mask=gt_mask_visib,
            )
            out = {"_train_stats": {
                "pcc_symmetry_branch_mean": stats["selected_symmetry_branch_mean"],
                **{f"pcc_fusion_gate_{index+1}": gate
                   for index, gate in enumerate(stats["fusion_gates"])}
            }}
            return out, losses
        output = self.pcc_head(backbone_feature, roi_classes)
        xyz = output["xyz_norm"]
        result = {
            "coor_x": xyz[:, 0:1], "coor_y": xyz[:, 1:2], "coor_z": xyz[:, 2:3],
            "mask": output["mask_logit"],
        }
        if return_pcc_debug:
            result.update(output)
        return result


def build_model_optimizer(cfg, is_test=False):
    net = cfg.MODEL.POSE_NET
    if net.NAME != "GDRN_PCC" or not net.BACKBONE.FREEZE or cfg.INPUT.WITH_DEPTH:
        raise ValueError("EXP022 V1 requires GDRN_PCC, frozen RGB backbone")
    if not net.PCC_HEAD.ENABLED:
        raise ValueError("EXP022 PCC_HEAD must be enabled")
    backbone_type, args = get_backbone_init_args(cfg)
    backbone = BACKBONES[backbone_type](**args)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    hierarchy_path = os.path.expandvars(os.path.expanduser(str(net.PCC_HEAD.HIERARCHY_PATH)))
    if "$" in hierarchy_path:
        raise ValueError(f"Unresolved EXP022 hierarchy path: {hierarchy_path}")
    kwargs = copy.deepcopy(net.PCC_HEAD.INIT_CFG)
    head = ProgressivePCCHead(hierarchy_path=hierarchy_path, **kwargs)
    model = GDRN_PCC(backbone, head)
    optimizer = None if is_test else build_optimizer_with_params(cfg, [{
        "params": list(head.parameters()), "lr": float(cfg.SOLVER.BASE_LR),
    }])
    return model.to(torch.device(cfg.MODEL.DEVICE)), optimizer
