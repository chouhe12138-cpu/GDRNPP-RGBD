"""Exercise the legacy LM evaluator with GT-oracle poses on eight real images.

This validates dataset/evaluator wiring and metrics, not CAD model accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmcv import Config
from detectron2.data import DatasetCatalog
from transforms3d.quaternions import quat2mat

from core.gdrn_modeling.engine.gdrn_custom_evaluator import GDRN_EvaluatorCustom
from core.gdrn_modeling.models.GDRN_CAD import dataset_context

CONFIG = Path('configs/gdrn/lm/research/candidate_cad/eval_smoke.py')


def run(cfg, output):
    if cfg.VAL.USE_BOP or cfg.TEST.TEST_BBOX_TYPE != 'gt' or cfg.VAL.RENDERER_TYPE != 'cpp':
        raise ValueError('Legacy LM GT-box/CPP evaluator required')
    context = dataset_context(cfg)
    if context.test_dataset != 'lm_13_test_smoke':
        raise ValueError('Expected the eight-image LM test smoke split')
    output.mkdir(parents=True, exist_ok=False)
    cfg.EXP_ID = 'lm13_cad_candidate_eval_smoke'
    evaluator = GDRN_EvaluatorCustom(cfg, context.test_dataset, distributed=False,
                                     output_dir=str(output), train_objs=context.object_names)
    evaluator.reset()
    samples = DatasetCatalog.get(context.test_dataset)
    predictions = []
    for sample in samples:
        for annotation in sample['annotations']:
            predictions.append(dict(cls_name=context.object_names[annotation['category_id']],
                                    file_name=sample['file_name'], score=1.0,
                                    R=quat2mat(annotation['quat']), t=annotation['trans'], time=0.0))
    if not predictions:
        raise RuntimeError('No LM GT oracle samples')
    evaluator._predictions = predictions
    evaluator.evaluate()
    result = dict(status='PASS', kind='legacy_oracle_evaluator_smoke', samples=len(samples),
                  predictions=len(predictions), dataset=context.key, gt_box=True,
                  configured_renderer='cpp', renderer_invoked=False,
                  bop_evaluator=False, model_accuracy=False)
    (output / 'report.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(Config.fromfile(str(args.config)), args.output), indent=2))


if __name__ == '__main__':
    main()
