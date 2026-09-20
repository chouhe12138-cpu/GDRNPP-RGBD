"""EXP022 visual backbone and PCC head without the legacy decoder or pose head."""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn

from core.utils.solver_utils import build_optimizer_with_params

from .backbone_factory import get_backbone_init_args
from .heads.progressive_pcc_head import ProgressivePCCHead
from .net_factory import BACKBONES
from core.gdrn_modeling.datasets.research_context import resolve_dataset_context


class GDRN_PCC(nn.Module):
    def __init__(self, backbone: nn.Module, pcc_head: ProgressivePCCHead):
        super().__init__()
        self.backbone = backbone
        self.pcc_head = pcc_head

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, x, roi_classes=None, gt_xyz=None, gt_mask_visib=None,
                do_loss: bool = False, return_pcc_debug: bool = False,
                collect_pcc_diagnostics: bool = False, **_unused):
        if roi_classes is None:
            raise ValueError("EXP022 requires roi_classes")
        if any(parameter.requires_grad for parameter in self.backbone.parameters()) and self.training:
            backbone_feature = self.backbone(x)
        else:
            with torch.no_grad():
                backbone_feature = self.backbone(x)
        if isinstance(backbone_feature, (tuple, list)):
            backbone_feature = backbone_feature[0]
        if do_loss:
            losses, stats = self.pcc_head(
                backbone_feature, roi_classes, gt_xyz_norm=gt_xyz,
                gt_mask=gt_mask_visib, collect_diagnostics=collect_pcc_diagnostics,
            )
            out = {"_train_stats": {
                "pcc_symmetry_branch_mean": stats["selected_symmetry_branch_mean"],
                **{f"pcc_fusion_gate_{index+1}": gate
                   for index, gate in enumerate(stats["fusion_gates"])},
                **{f"pcc_{name}_{index+1}": value
                   for name in ("fusion_update_ratio", "route_entropy", "top1_route_prob",
                                "top2_route_prob_mass")
                   for index, value in enumerate(stats.get(name, ()))}
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
    if net.NAME != "GDRN_PCC" or cfg.INPUT.WITH_DEPTH:
        raise ValueError("EXP022 requires GDRN_PCC with RGB input")
    if not net.PCC_HEAD.ENABLED:
        raise ValueError("EXP022 PCC_HEAD must be enabled")
    context = resolve_dataset_context(cfg)
    backbone_type, args = get_backbone_init_args(cfg)
    if not net.BACKBONE.FREEZE and not cfg.MODEL.WEIGHTS:
        checkpoint = os.path.expandvars(str(args.get("checkpoint_path", "")))
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"EXP022 ImageNet backbone checkpoint missing: {checkpoint}")
        args["checkpoint_path"] = checkpoint
    backbone = BACKBONES[backbone_type](**args)
    for parameter in backbone.parameters():
        parameter.requires_grad_(not net.BACKBONE.FREEZE)
    hierarchy_path = os.path.expandvars(os.path.expanduser(str(net.PCC_HEAD.HIERARCHY_PATH)))
    if "$" in hierarchy_path:
        raise ValueError(f"Unresolved EXP022 hierarchy path: {hierarchy_path}")
    kwargs = copy.deepcopy(net.PCC_HEAD.INIT_CFG)
    kwargs["expected_object_ids"] = context.object_ids
    kwargs["dataset_key"] = context.key
    head = ProgressivePCCHead(hierarchy_path=hierarchy_path, **kwargs)
    model = GDRN_PCC(backbone, head)
    groups = [{"params": list(head.parameters()), "lr": float(cfg.SOLVER.BASE_LR)}]
    if not net.BACKBONE.FREEZE:
        groups.append({"params": list(backbone.parameters()),
                       "lr": float(cfg.SOLVER.BASE_LR) * float(net.BACKBONE.get("LR_MULT", 1.0))})
    optimizer = None if is_test else build_optimizer_with_params(cfg, groups)
    return model.to(torch.device(cfg.MODEL.DEVICE)), optimizer
