"""EXP025 backbone wrapper; explicit PnP remains in the existing evaluator."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
from torch import nn

from core.utils.solver_utils import build_optimizer_with_params
from .backbone_factory import BACKBONES, get_backbone_init_args
from .heads.hierarchical_cad_attention_head import HierarchicalCADAttentionHead
from core.gdrn_modeling.datasets.research_context import resolve_dataset_context
from research.cad_hierarchy.contracts import require_configured_hierarchy


def dataset_context(cfg):
    context = resolve_dataset_context(
        cfg, hierarchy_path=cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH)
    require_configured_hierarchy(cfg, context)
    return context


EXP025_CHECKPOINT_TENSORS = ('cad_attention_head.t3_classifier.weight',
                             'cad_attention_head.residual_predictor.final.weight',
                             'cad_attention_head.mask_predictor.weight')


def require_full_checkpoint(weights):
    """Reject anything that cannot restore every EXP025 component.

    The hierarchy buffers are non-persistent and the head is built from random
    initialization, so an evaluation pointed at a legacy backbone-only checkpoint would
    score a random head without reporting anything unusual.  Fail instead.
    """
    path = str(weights or '')
    if not path:
        raise ValueError('EXP025 requires a complete GDRN_CAD checkpoint in MODEL.WEIGHTS, got an empty path')
    if not Path(path).is_file():
        raise FileNotFoundError(f'EXP025 checkpoint not found: {path}')
    state = torch.load(path, map_location='cpu')
    state = state.get('model', state.get('state_dict', state))
    keys = {key[len('_module.'):] if key.startswith('_module.') else key for key in state}
    missing = [name for name in EXP025_CHECKPOINT_TENSORS if name not in keys]
    if not any(key.startswith('backbone.') for key in keys):
        missing.append('backbone.*')
    if missing:
        raise ValueError(f'{path} is not a complete EXP025 checkpoint; missing {missing}')
    return path


def load_official_backbone(model, path):
    state = torch.load(path, map_location='cpu')
    state = state.get('model', state.get('state_dict', state))
    source = {key[len('backbone.'):]: value for key, value in state.items() if key.startswith('backbone.')}
    model.backbone.load_state_dict(source, strict=True)
    if any(not torch.equal(value.cpu(), source[key].cpu()) for key, value in model.backbone.state_dict().items()):
        raise RuntimeError('Official backbone not loaded exactly')
    return len(source)


class GDRN_CAD(nn.Module):
    def __init__(self, backbone, cad_attention_head):
        super().__init__()
        self.backbone, self.cad_attention_head = backbone, cad_attention_head

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def backbone_feature(self, image):
        """The single [B,1024,8,8] tensor the head consumes, with the frozen-backbone rule."""
        if self.training and any(p.requires_grad for p in self.backbone.parameters()):
            feature = self.backbone(image)
        else:
            with torch.no_grad():
                feature = self.backbone(image)
        if isinstance(feature, (tuple, list)):
            feature = feature[0]
        return feature

    def predict(self, image, classes, diagnostics=None):
        return self.cad_attention_head(self.backbone_feature(image), classes, diagnostics=diagnostics)

    def forward(self, x, roi_classes=None, gt_xyz=None, gt_mask_visib=None, do_loss=False,
                return_cad_debug=False, diagnostics=None, **_unused):
        if roi_classes is None:
            raise ValueError('EXP025 requires ROI classes')
        prediction = self.predict(x, roi_classes, diagnostics=diagnostics)
        if do_loss:
            if gt_xyz is None or gt_mask_visib is None:
                raise ValueError('EXP025 requires visible XYZ targets')
            losses, stats = self.cad_attention_head.loss(prediction, roi_classes, gt_xyz, gt_mask_visib)
            return {'_train_stats': stats}, losses
        xyz = self.cad_attention_head.decode(prediction, roi_classes)
        result = dict(coor_x=xyz[:, :1], coor_y=xyz[:, 1:2], coor_z=xyz[:, 2:3], mask=prediction['mask_logit'])
        if return_cad_debug:
            result.update(prediction, xyz_norm=xyz)
        return result


def build_model_optimizer(cfg, is_test=False):
    net = cfg.MODEL.POSE_NET
    if net.NAME != 'GDRN_CAD' or cfg.INPUT.WITH_DEPTH or not net.CAD_ATTENTION_HEAD.ENABLED:
        raise ValueError('EXP025 requires RGB GDRN_CAD')
    if is_test or bool(cfg.TEST.get('SAVE_RESULTS_ONLY', False)):
        # Evaluation is the one path that never trains the head: it must be told where a
        # complete checkpoint is, and fail before anything is scored.
        require_full_checkpoint(cfg.MODEL.WEIGHTS)
    if bool(net.BACKBONE.FREEZE) == bool(cfg.TRAIN_BACKBONE):
        raise ValueError('Backbone controls disagree; edit the selected arm config or use the tool override')
    context = dataset_context(cfg)
    backbone_type, args = get_backbone_init_args(cfg)
    source = str(cfg.BACKBONE_INIT)
    if source not in ('official_lmo', 'imagenet'):
        raise ValueError(f'Unknown initialization: {source}')
    if source == 'imagenet':
        path = os.path.expandvars(str(args.get('checkpoint_path', '')))
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f'EXP025 ImageNet checkpoint missing: {path}')
        args['checkpoint_path'] = path
    elif args.get('checkpoint_path'):
        raise ValueError('Official initialization must not also load ImageNet weights')
    backbone = BACKBONES[backbone_type](**args)
    for p in backbone.parameters():
        p.requires_grad_(not net.BACKBONE.FREEZE)
    head = HierarchicalCADAttentionHead(context.hierarchy_path, expected_object_ids=context.object_ids,
        dataset_key=context.key, **copy.deepcopy(net.CAD_ATTENTION_HEAD.INIT_CFG))
    model = GDRN_CAD(backbone, head)
    # Keep initialization independent of freeze status. Main entry can subsequently
    # restore a full EXP025 checkpoint through its unchanged checkpointer.
    if source == 'official_lmo':
        from research.exp025.configuration import OFFICIAL_WEIGHTS
        load_official_backbone(model, OFFICIAL_WEIGHTS)
    lr = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    groups = [dict(params=list(head.parameters()), lr=lr, name='cad_head')]
    if not net.BACKBONE.FREEZE:
        groups.append(dict(params=list(backbone.parameters()), lr=lr * float(net.BACKBONE.LR_MULT), name='backbone'))
    cfg.SOLVER.BASE_LR = lr
    optimizer = None if is_test else build_optimizer_with_params(cfg, groups)
    return model.to(torch.device(cfg.MODEL.DEVICE)), optimizer
