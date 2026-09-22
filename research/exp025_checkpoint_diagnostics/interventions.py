"""Small matched E15 backbone and CAD cross-attention inference interventions."""
from __future__ import annotations

import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cv2
from detectron2.data import DatasetCatalog

from core.gdrn_modeling.datasets.data_loader import GDRN_DatasetFromList
from core.gdrn_modeling.engine.gdrn_evaluator import GDRN_Evaluator
from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer, load_official_backbone
from core.gdrn_modeling.models.heads.hierarchical_cad_attention_head import nested_targets
from research.exp025.preflight import read_config
from .oracle import CONFIG, add_s_error, gt_geometry, pnp_pose, pose_error, select_records, summary


def linear_cka(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a -= a.mean(0, keepdims=True)
    b -= b.mean(0, keepdims=True)
    cross = np.linalg.norm(a.T @ b, 'fro') ** 2
    denom = np.linalg.norm(a.T @ a, 'fro') * np.linalg.norm(b.T @ b, 'fro')
    return float(cross / denom) if denom else None


def run(args):
    torch.set_num_threads(4)
    os.environ['GDRN_CONVNEXT_BASE_WEIGHTS'] = str(Path(args.imagenet_weights).resolve())
    cfg = read_config(CONFIG)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = str(Path(args.checkpoint).resolve())
    model, _ = build_model_optimizer(cfg, is_test=True)
    imagenet = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
    raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict({k.removeprefix('_module.'): v for k, v in raw['model'].items()}, strict=True)
    full = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
    model.eval()
    records = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    all_selected = select_records(records, model.cad_attention_head.object_ids.tolist(),
                                  args.manifest_per_object, args.seed)
    selected = [all_selected[i * args.manifest_per_object + j]
                for i in range(8) for j in range(args.per_object)]
    mapper = GDRN_DatasetFromList(cfg, split='test', lst=[v[3] for v in selected], flatten=False,
                                  copy=True, serialize=False)
    evaluator = GDRN_Evaluator(cfg, cfg.DATASETS.TEST[0], distributed=False, output_dir=str(args.output))
    points = {}
    for obj_id, data in zip(evaluator.obj_ids, evaluator.models_3d):
        vertices = np.asarray(data['pts'], dtype=np.float32)
        chosen = np.random.default_rng(args.seed + obj_id).choice(
            len(vertices), min(2000, len(vertices)), replace=False)
        points[obj_id] = vertices[chosen]
    samples = []
    for index, (obj_id, sid, ann, rec) in enumerate(selected):
        mapped = mapper[index]
        gt, valid, gt_pose = gt_geometry(rec, ann, mapped)
        samples.append((obj_id, sid, mapped, gt, valid, gt_pose))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'manifest.json').write_text(json.dumps(
        dict(seed=args.seed, samples=[dict(object_id=int(o), scene_im_id=s,
                                           original_annotation_index=int(a))
                                for o, s, a, _ in selected]), indent=2))

    rows = defaultdict(list)
    dashboards = []
    activations = defaultdict(lambda: defaultdict(list))
    hooks = []
    stage_cache = {}
    for level in range(4):
        stage = getattr(model.backbone, f'stages_{level}')
        hooks.append(stage.register_forward_hook(
            lambda _m, _x, y, level=level: stage_cache.__setitem__(str(level),
                y[0].detach().float().mean((2, 3)).cpu().numpy()[0]
                if isinstance(y, (tuple, list)) else y.detach().float().mean((2, 3)).cpu().numpy()[0])))

    def evaluate(name, sample, prediction):
        obj_id, sid, mapped, gt, valid, gt_pose = sample
        label = int(mapped['roi_cls'][0])
        classes = torch.tensor([label], device=args.device)
        with torch.inference_mode():
            xyz = model.cad_attention_head.decode(prediction, classes)[0].permute(1, 2, 0)
            xyz = ((xyz - .5) * model.cad_attention_head.extents[label]).cpu().numpy()
            route = prediction['t3_logits'].argmax(1)[0]
            mask = prediction['mask_logit'][0, 0].sigmoid().cpu().numpy()
            if name == 'full':
                head = model.cad_attention_head
                gt_t = torch.from_numpy(gt).to(args.device)
                gt_ids = nested_targets(gt_t.reshape(1, -1, 3), classes,
                                        [getattr(head, f'level{d}_anchors') for d in (1, 2, 3)])[2]
                anchor = head.level3_anchors[label, route].float()
                radius = head.level3_radii[label, route].float()
                anchor_error = (anchor - gt_t).norm(dim=-1).cpu().numpy()
                representable = (((anchor - gt_t).norm(dim=-1) / radius) <= 1).cpu().numpy()
                img = mapped['roi_img'][0].numpy().transpose(1, 2, 0)
                if cfg.INPUT.FORMAT.upper() == 'BGR':
                    img = img[..., ::-1]
                dashboards.append(dict(image=np.clip(img, 0, 1), gt_mask=valid,
                                       pred_mask=mask > cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST,
                                       route_correct=np.where(valid, (route == gt_ids.reshape(64, 64)).cpu().numpy(),
                                                              np.nan),
                                       xyz_error=np.where(valid, np.linalg.norm(xyz - gt, axis=-1), np.nan),
                                       residual_gain=np.where(valid, anchor_error - np.linalg.norm(xyz - gt, axis=-1),
                                                              np.nan),
                                       representable=np.where(valid, representable, np.nan)))
        cv2.setRNGSeed(args.seed + int(mapped['im_H'][0]) + int(mapped['inst_id'][0]))
        pose, n_points = pnp_pose(evaluator, cfg, mapped, xyz, mask)
        error = np.linalg.norm(xyz - gt, axis=-1)
        add_s = add_s_error(pose, gt_pose, points[obj_id], obj_id in (10, 11))
        diameter = float(model.cad_attention_head.diameters[label])
        rows[name].append(dict(object_id=int(obj_id), scene_im_id=sid, pnp_points=n_points,
                               xyz_error_m=float(error[valid].mean()),
                               route_top1_mode=int(np.bincount(route.flatten().cpu().numpy(), minlength=512).argmax()),
                               add_s_m=add_s, add_s_0_1d=0. if add_s is None else float(add_s < .1 * diameter),
                               **pose_error(pose, gt_pose)))

    with torch.inference_mode():
        for name, backbone_state in (('full', full), ('imagenet_reset', imagenet), ('official_transplant', None)):
            if backbone_state is None:
                load_official_backbone(model, 'pretrained_models/lmo_pbr/model_final_wo_optim.pth')
            else:
                model.backbone.load_state_dict(backbone_state, strict=True)
            model.eval()
            for index, sample in enumerate(samples):
                _, _, mapped, _, valid, _ = sample
                if valid.sum() < 16:
                    continue
                classes = mapped['roi_cls'].to(args.device)
                image = mapped['roi_img'].to(args.device)
                stage_cache.clear()
                feature = model.backbone_feature(image)
                for stage, vector in stage_cache.items():
                    activations[name][stage].append(vector)
                baseline = model.cad_attention_head(feature, classes)
                evaluate(name, sample, baseline)
                if name != 'full':
                    continue
                for level in (1, 2, 3, 'all', 'scramble_t3_channels'):
                    temporary = []
                    if level == 'scramble_t3_channels':
                        generator = torch.Generator().manual_seed(args.seed)
                        perm = torch.stack([torch.randperm(512, generator=generator)
                                            for _ in range(model.cad_attention_head.token_dim)], 1).to(args.device)
                        temporary.append(model.cad_attention_head.cross_attention[3].register_forward_pre_hook(
                            lambda _m, inp, perm=perm: (inp[0], torch.gather(
                                inp[1], 1, perm[None].expand(inp[1].shape[0], -1, -1)))))
                    else:
                        indices = range(1, 4) if level == 'all' else (level,)
                        for chosen in indices:
                            temporary.append(model.cad_attention_head.cross_attention[chosen].register_forward_hook(
                                lambda _m, inp, _out: inp[0]))
                    try:
                        prediction = model.cad_attention_head(feature, classes)
                        evaluate(f'bypass_{level}' if level != 'scramble_t3_channels' else level,
                                 sample, prediction)
                    finally:
                        for hook in temporary:
                            hook.remove()
                if (index + 1) % 8 == 0:
                    print(f'interventions {index + 1}/{len(samples)}', flush=True)
    for hook in hooks:
        hook.remove()
    names = ('full', 'imagenet_reset', 'official_transplant')
    cka = {str(level): {f'{a}_vs_{b}': linear_cka(activations[a][str(level)], activations[b][str(level)])
                        for a, b in ((names[0], names[1]), (names[0], names[2]), (names[1], names[2]))}
           for level in range(4)}
    report = dict(sample_count=len(samples),
                  variants={name: {metric: summary([row[metric] for row in items if row[metric] is not None])
                                   for metric in ('xyz_error_m', 'add_s_m', 'add_s_0_1d',
                                                  'rotation_deg', 'translation_m')}
                            for name, items in rows.items()}, cka=cka,
                  backbone_weight_relative_l2={
                      'full_vs_imagenet': float(np.sqrt(sum((full[k].float()-imagenet[k].float()).square().sum().item()
                                                   for k in full)) /
                                                   np.sqrt(sum(imagenet[k].float().square().sum().item()
                                                               for k in full)))})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure_dir = args.output / 'figures'
    figure_dir.mkdir()
    sorted_idx = sorted(range(len(rows['full'])), key=lambda i: float('inf') if rows['full'][i]['add_s_m'] is None
                        else rows['full'][i]['add_s_m'])
    chosen = sorted(set(sorted_idx[:4] + sorted_idx[-4:]))
    figures = []
    for i in chosen:
        data = dashboards[i]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        layers = ((data['image'], 'ROI RGB', None),
                  (data['gt_mask'].astype(float) + data['pred_mask'].astype(float), 'GT+pred mask', 'viridis'),
                  (data['route_correct'], 'T3 correct', 'RdYlGn'),
                  (data['xyz_error'] * 1000, 'XYZ error mm', 'magma'),
                  (data['residual_gain'] * 1000, 'Residual gain mm', 'coolwarm'),
                  (data['representable'], 'Pred cell reachable', 'RdYlGn'))
        for ax, (layer, title, cmap) in zip(axes.flat, layers):
            image = ax.imshow(layer, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
            if title in ('XYZ error mm', 'Residual gain mm'):
                fig.colorbar(image, ax=ax, fraction=.046)
        item = rows['full'][i]
        fig.suptitle(f"obj {item['object_id']} | {item['scene_im_id']} | ADD-S {item['add_s_m']}")
        fig.tight_layout()
        path = figure_dir / f"sample_{i:03d}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        figures.append(str(path))
    report['figures'] = figures
    (args.output / 'per_roi.json').write_text(json.dumps(rows, indent=2))
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--imagenet-weights', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--manifest-per-object', type=int, default=16)
    parser.add_argument('--per-object', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260922)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
