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


def audit(checkpoint, output, imagenet_weights):
    checkpoint, output = Path(checkpoint), Path(output)
    os.environ['GDRN_CONVNEXT_BASE_WEIGHTS'] = str(Path(imagenet_weights).resolve())
    cfg = read_config(CONFIG)
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = str(checkpoint)
    if cfg.EXP025_ARM != 'imagenet_full' or cfg.BACKBONE_INIT != 'imagenet' or not cfg.TRAIN_BACKBONE:
        raise RuntimeError('Not the EXP025 imagenet_full config')
    hierarchy = require_hierarchy(cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH, 'lmo')
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
                  config=str(CONFIG), arm=cfg.EXP025_ARM, hierarchy_sha256=hierarchy,
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
    args = parser.parse_args()
    print(json.dumps(audit(args.checkpoint, args.output, args.imagenet_weights), indent=2))


if __name__ == '__main__':
    main()
