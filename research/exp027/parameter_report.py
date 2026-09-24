"""Count actual, non-overlapping EXP026/027 model parameters."""
from __future__ import annotations

import argparse
import json

from mmcv import Config

from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from research.exp027.preflight import ARMS

BASELINE = 'configs/gdrn/lmo_pbr/research/exp026_residual_aligned_sampling_ablation/train_adaptive_full.py'


def count(module):
    return sum(p.numel() for p in module.parameters())


def report(config):
    cfg = Config.fromfile(str(config))
    cfg.MODEL.DEVICE = 'cpu'
    model, _ = build_model_optimizer(cfg)
    head = model.cad_attention_head
    components = dict(
        fpn_lateral=count(head.laterals) + head.lateral_alpha.numel() if hasattr(head, 'laterals') else 0,
        cross_query=count(head.cross_attention) +
            sum(count(getattr(head, name)) for name in
                ('query_parents', 'pixel_projection', 'query_projection') if hasattr(head, name)) +
            (head.query_parent_alpha.numel() if hasattr(head, 'query_parent_alpha') else 0),
        classifier=count(head.t3_classifier) if hasattr(head, 't3_classifier') else 0,
        residual_mask=count(head.residual_predictor) + count(head.mask_predictor),
    )
    head_total = count(head)
    components['other_head'] = head_total - sum(components.values())
    if components['other_head'] < 0 or count(model) != count(model.backbone) + head_total:
        raise RuntimeError('Parameter categories overlap or omit a model component')
    return dict(total=count(model), trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                backbone=count(model.backbone), head=head_total, **components)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args = parser.parse_args()
    rows = {'EXP026_adaptive': report(BASELINE)}
    for arm, path in ARMS.items():
        rows[arm] = report(path)
    baseline = rows['EXP026_adaptive']['total']
    for row in rows.values():
        row['delta_vs_exp026'] = row['total'] - baseline
        row['delta_percent'] = 100 * (row['total'] - baseline) / baseline
    rendered = json.dumps(rows, indent=2, sort_keys=True)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
