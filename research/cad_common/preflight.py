"""Dataset-neutral CAD model initialization and optimizer audits."""
from __future__ import annotations

import re

import torch


def audit_optimizer(model, optimizer, cfg):
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    registered = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(registered) != len(set(registered)) or set(registered) != trainable:
        raise RuntimeError('Optimizer parameter coverage/duplicates')
    groups = []
    for group in optimizer.param_groups:
        expected = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
        if group['name'] == 'backbone':
            expected *= float(cfg.BACKBONE_LR_MULT)
        if abs(group['lr'] - expected) > 1e-12:
            raise RuntimeError('Optimizer LR mismatch')
        groups.append(dict(name=group['name'], lr=group['lr'],
                           parameters=sum(p.numel() for p in group['params'])))
    return groups


def verify_imagenet_backbone(model, path):
    """Verify every ConvNeXt tensor against the official Facebook checkpoint."""
    raw = torch.load(path, map_location='cpu', weights_only=False)
    raw = raw.get('model', raw.get('state_dict', raw))
    current = model.backbone.state_dict()
    converted = {}
    for name, tensor in raw.items():
        if name.startswith(('head.', 'norm.')):
            continue
        name = name.replace('downsample_layers.0.0.', 'stem_0.')
        name = name.replace('downsample_layers.0.1.', 'stem_1.')
        name = re.sub(r'downsample_layers\.(\d+)\.(\d+)',
                      lambda match: f'stages_{match.group(1)}.downsample.{match.group(2)}', name)
        name = re.sub(r'stages\.(\d+)\.(\d+)',
                      lambda match: f'stages_{match.group(1)}.blocks.{match.group(2)}', name)
        name = name.replace('dwconv', 'conv_dw').replace('pwconv', 'mlp.fc')
        if name not in current:
            raise RuntimeError(f'Unmapped ImageNet tensor: {name}')
        converted[name] = tensor.reshape(current[name].shape)
    if current.keys() != converted.keys():
        raise RuntimeError(f'ImageNet backbone tensor mismatch: {current.keys() - converted.keys()}')
    if any(not torch.equal(current[name].cpu(), converted[name]) for name in current):
        raise RuntimeError('ImageNet backbone weights were not loaded exactly')
    return len(converted)
