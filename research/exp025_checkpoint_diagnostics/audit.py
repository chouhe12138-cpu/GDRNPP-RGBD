"""Fail-closed identity, finite-tensor and strict-restore audit for E15 Full."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from research.exp025.configuration import require_hierarchy
from research.exp025.preflight import read_config

CONFIG = Path('configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_imagenet_full.py')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit(checkpoint, output, imagenet_weights, config=CONFIG, expected_arm=None):
    checkpoint, output = Path(checkpoint), Path(output)
    os.environ['GDRN_CONVNEXT_BASE_WEIGHTS'] = str(Path(imagenet_weights).resolve())
    cfg = read_config(config)
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = str(checkpoint)
    if expected_arm:
        if cfg.get('EXP026_ARM') != expected_arm or expected_arm != 'adaptive_l1_full':
            raise RuntimeError('Not the EXP026 adaptive_l1_full config')
        if cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG.residual_target_mode != 'predicted_route':
            raise RuntimeError('Residual target is not predicted_route')
    elif cfg.EXP025_ARM != 'imagenet_full':
        raise RuntimeError('Not the EXP025 imagenet_full config')
    if cfg.BACKBONE_INIT != 'imagenet' or not cfg.TRAIN_BACKBONE:
        raise RuntimeError('Not ImageNet Full')
    hierarchy = (sha256_file(cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH)
                 if expected_arm else require_hierarchy(cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH, 'lmo'))
    if expected_arm and hierarchy != cfg.CAD_HIERARCHY_CONTRACT.SHA256:
        raise RuntimeError('EXP026 hierarchy SHA mismatch')
    raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(raw, dict) or not isinstance(raw.get('model'), dict):
        raise RuntimeError('Expected formal checkpoint with model state')
    state = {key.removeprefix('_module.'): value for key, value in raw['model'].items()}
    nonfinite = []
    groups = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise RuntimeError(f'Non-tensor model state: {key}')
        if not torch.isfinite(value).all():
            nonfinite.append(key)
        group = key.split('.')[0]
        item = groups.setdefault(group, dict(tensors=0, numel=0))
        item['tensors'] += 1
        item['numel'] += value.numel()
    if nonfinite:
        raise RuntimeError(f'Nonfinite model tensors: {nonfinite[:8]}')
    model, _ = build_model_optimizer(cfg, is_test=True)
    model.load_state_dict(state, strict=True)
    report = dict(status='PASS', checkpoint=str(checkpoint.resolve()),
                  checkpoint_sha256=sha256_file(checkpoint), checkpoint_size=checkpoint.stat().st_size,
                  config=str(config), arm=expected_arm or cfg.EXP025_ARM, hierarchy_sha256=hierarchy,
                  residual_target_mode=cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG.get('residual_target_mode'),
                  top_level_keys=sorted(raw), iteration=raw.get('iteration'), epoch=raw.get('epoch'),
                  state_tensors=len(state), state_groups=groups,
                  model_parameters=sum(p.numel() for p in model.parameters()),
                  strict_restore=True, nonfinite_tensors=nonfinite)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--imagenet-weights', required=True)
    parser.add_argument('--config', default=str(CONFIG))
    parser.add_argument('--expected-arm')
    args = parser.parse_args()
    print(json.dumps(audit(args.checkpoint, args.output, args.imagenet_weights,
                           args.config, args.expected_arm), indent=2))


if __name__ == '__main__':
    main()
