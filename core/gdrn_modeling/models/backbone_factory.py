"""Backbone construction shared by active models without importing pose heads."""
from __future__ import annotations

import copy
import logging

import timm

from core.utils.timm_utils import my_create_timm_model
from .backbones.mm_nets.resnet import ResNet, ResNetV1c
from .backbones.mm_nets.mmcls_resnet import ResNetV1d
from .backbones.mm_nets.resnest import ResNeSt
from .backbones.mm_nets.darknet import Darknet
from .backbones.resnet_backbone import get_resnet
from .backbones.pvnet_net.model_repository import (
    Resnet18_8s, Resnet34_8s, Resnet50_8s, Resnet50_8s_2o,
)
from .backbones.resnet_d2 import ResNet50_GN_D2

logger = logging.getLogger(__name__)

BACKBONES = {
    "Resnet18_8s": Resnet18_8s,
    "Resnet34_8s": Resnet34_8s,
    "Resnet50_8s": Resnet50_8s,
    "Resnet50_8s_2o": Resnet50_8s_2o,
    "mm/ResNet": ResNet,
    "mm/ResNetV1c": ResNetV1c,
    "mm/ResNetV1d": ResNetV1d,
    "mm/ResNeSt": ResNeSt,
    "mm/Darknet": Darknet,
    "resnet50_gn_d2": ResNet50_GN_D2,
}
for backbone_name in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
    BACKBONES[f"tv/{backbone_name}"] = get_resnet
for backbone_name in timm.list_models(pretrained=True):
    BACKBONES[f"timm/{backbone_name}"] = my_create_timm_model


def get_backbone_init_args(cfg):
    """Build backbone arguments without network access for full checkpoints."""
    init_args = copy.deepcopy(cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG)
    backbone_type = init_args.pop("type")
    if "timm/" in backbone_type or "tv/" in backbone_type:
        init_args["model_name"] = backbone_type.split("/")[-1]
    if cfg.MODEL.WEIGHTS and init_args.get("pretrained", False):
        logger.info("Disable backbone pretrained download because MODEL.WEIGHTS is set")
        init_args["pretrained"] = False
    return backbone_type, init_args
