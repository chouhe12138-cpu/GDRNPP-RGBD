"""Matched GT-box E15 correspondence/oracle diagnostic on BOP LM-O test ROIs.

This is a diagnostic sample, not the formal BOP evaluator. The PnP point extraction
and solver are the exact functions and settings used by GDRN_Evaluator.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.data import DatasetCatalog
from scipy.spatial import cKDTree

from core.gdrn_modeling.datasets.data_loader import GDRN_DatasetFromList
from core.gdrn_modeling.engine.gdrn_evaluator import GDRN_Evaluator
from core.gdrn_modeling.models.GDRN_CAD import build_model_optimizer
from core.gdrn_modeling.models.heads.hierarchical_cad_attention_head import hierarchy_log_probabilities, nested_targets
from lib.pysixd import misc
from research.exp025.preflight import read_config

CONFIG = 'configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_imagenet_full.py'
VARIANTS = ('normal', 'anchor_only', 'oracle_pred_route', 'gt_xyz_pred_mask',
            'normal_gt_mask', 'gt_xyz_gt_mask')


def select_records(records, object_ids, per_object, seed):
    by_object = defaultdict(list)
    for record in records:
        for index, annotation in enumerate(record['annotations']):
            by_object[int(annotation['category_id'])].append((record, index))
    rng = random.Random(seed)
    selected = []
    for label, obj_id in enumerate(object_ids):
        candidates = by_object[label]
        if len(candidates) < per_object:
            raise RuntimeError(f'Only {len(candidates)} samples for object {obj_id}')
        for record, index in rng.sample(candidates, per_object):
            item = copy.deepcopy(record)
            item['annotations'] = [item['annotations'][index]]
            selected.append((obj_id, record['scene_im_id'], index, item))
    return selected


def gt_geometry(record, annotation, mapped):
    """Backproject measured BOP test depth at the evaluator's 64x64 ROI coordinates."""
    scene, image = map(int, record['scene_im_id'].split('/'))
    root = Path('datasets/BOP_DATASETS/lmo/test') / f'{scene:06d}'
    depth = cv2.imread(str(root / 'depth' / f'{image:06d}.png'), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(root / 'mask_visib' / f'{image:06d}_{annotation:06d}.png'), cv2.IMREAD_GRAYSCALE)
    if depth is None or mask is None:
        raise FileNotFoundError((scene, image, annotation))
    xy = mapped['roi_coord_2d'][0].numpy().transpose(1, 2, 0)
    im_h, im_w = depth.shape
    xx = np.rint(xy[..., 0] * im_w).astype(np.int32).clip(0, im_w - 1)
    yy = np.rint(xy[..., 1] * im_h).astype(np.int32).clip(0, im_h - 1)
    z = depth[yy, xx].astype(np.float32) / float(record['depth_factor'])
    valid = (mask[yy, xx] > 0) & (z > 0)
    k = record['cam']
    camera = np.stack(((xx - k[0, 2]) * z / k[0, 0], (yy - k[1, 2]) * z / k[1, 1], z), -1)
    pose = record['annotations'][0]['pose']
    xyz = (camera - pose[:, 3]) @ pose[:, :3]
    xyz[~valid] = 0
    return xyz.astype(np.float32), valid, pose.astype(np.float32)


def pnp_pose(evaluator, cfg, mapped, xyz, mask):
    coord = mapped['roi_coord_2d'][0].numpy().transpose(1, 2, 0)
    extent = mapped['roi_extent'][0].numpy()
    xyz_norm = xyz / extent + .5
    img, obj = evaluator.get_img_model_points_with_coords2d(
        mask.astype(np.float32), xyz_norm.copy(), coord.copy(),
        int(mapped['im_H'][0]), int(mapped['im_W'][0]), extent,
        mask_thr=float(cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST))
    if len(img) < 4:
        return None, len(img)
    try:
        pose = misc.pnp_v2(obj, img, mapped['cam'][0].numpy().copy(), method=cv2.SOLVEPNP_EPNP,
                           ransac=True, ransac_reprojErr=3, ransac_iter=100)
        if not np.isfinite(pose).all() or np.linalg.norm(pose[:, 3]) > 10:
            return None, len(img)
        return pose, len(img)
    except cv2.error:
        return None, len(img)


def pose_error(pose, gt):
    if pose is None:
        return dict(rotation_deg=None, translation_m=None)
    cosine = np.clip((np.trace(pose[:, :3] @ gt[:, :3].T) - 1) / 2, -1, 1)
    return dict(rotation_deg=float(np.degrees(np.arccos(cosine))),
                translation_m=float(np.linalg.norm(pose[:, 3] - gt[:, 3])))


def add_s_error(pose, gt, points, symmetric):
    if pose is None:
        return None
    estimated = points @ pose[:, :3].T + pose[:, 3]
    actual = points @ gt[:, :3].T + gt[:, 3]
    if symmetric:
        return float(cKDTree(actual).query(estimated, workers=1)[0].mean())
    return float(np.linalg.norm(estimated - actual, axis=1).mean())


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)))


def run(args):
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['GDRN_CONVNEXT_BASE_WEIGHTS'] = str(Path(args.imagenet_weights).resolve())
    cfg = read_config(args.config)
    if args.expected_arm and cfg.get('EXP026_ARM') != args.expected_arm:
        raise RuntimeError(f'Unexpected diagnostic arm: {cfg.get("EXP026_ARM")}')
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = str(Path(args.checkpoint).resolve())
    model, _ = build_model_optimizer(cfg, is_test=True)
    raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict({k.removeprefix('_module.'): v for k, v in raw['model'].items()}, strict=True)
    model.eval()
    records = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    selected = select_records(records, model.cad_attention_head.object_ids.tolist(), args.per_object, args.seed)
    if args.manifest:
        expected = json.loads(args.manifest.read_text())
        actual = [dict(object_id=int(obj), scene_im_id=sid, original_annotation_index=int(index))
                  for obj, sid, index, _ in selected]
        if expected['seed'] != args.seed or expected['samples'] != actual:
            raise RuntimeError('Selected diagnostic records differ from canonical manifest')
    mapper = GDRN_DatasetFromList(cfg, split='test', lst=[v[3] for v in selected], flatten=False,
                                  copy=True, serialize=False)
    evaluator = GDRN_Evaluator(cfg, cfg.DATASETS.TEST[0], distributed=False, output_dir=str(args.output))
    cad_points = {}
    for obj_id, model_data in zip(evaluator.obj_ids, evaluator.models_3d):
        vertices = np.asarray(model_data['pts'], dtype=np.float32)
        chosen = np.random.default_rng(args.seed + obj_id).choice(
            len(vertices), min(2000, len(vertices)), replace=False)
        cad_points[obj_id] = vertices[chosen]
    manifest = [dict(object_id=int(obj), scene_im_id=sid, original_annotation_index=int(index))
                for obj, sid, index, _ in selected]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'manifest.json').write_text(json.dumps(dict(seed=args.seed, samples=manifest), indent=2))
    rows = []
    for index, (obj_id, _, original_ann, record) in enumerate(selected):
        mapped = mapper[index]
        xyz_gt, valid, gt_pose = gt_geometry(record, original_ann, mapped)
        if valid.sum() < 16:
            continue
        label = int(mapped['roi_cls'][0])
        classes = torch.tensor([label], device=args.device)
        image = mapped['roi_img'].to(args.device)
        with torch.inference_mode():
            prediction = model.predict(image, classes)
            lp1, lp2, lp3 = hierarchy_log_probabilities(prediction['t3_logits'])
            xyz = torch.from_numpy(xyz_gt).to(args.device)
            head = model.cad_attention_head
            route = prediction['t3_logits'].argmax(1)[0]
            anchor = head.level3_anchors[label, route].float()
            radius = head.level3_radii[label, route].float()
            branches = []
            for branch in range(int(head.symmetry_counts[label])):
                transform = head.symmetry_transforms[label, branch].float()
                equivalent = (xyz - transform[:3, 3]) @ transform[:3, :3]
                branches.append(equivalent)
            valid_t = torch.from_numpy(valid).to(args.device)
            branch_errors = [float((candidate - anchor).norm(dim=-1)[valid_t].mean())
                             for candidate in branches]
            branch = int(np.argmin(branch_errors))
            label_xyz = branches[branch]
            ids = nested_targets(label_xyz.reshape(1, -1, 3), classes,
                                 [getattr(head, f'level{d}_anchors') for d in (1, 2, 3)])
            gt_ids = [v.reshape(64, 64) for v in ids]
            residual = prediction['residual'][0].permute(1, 2, 0).float()
            full = anchor + radius[..., None] * residual
            delta = (label_xyz - anchor) / radius[..., None]
            oracle = anchor + radius[..., None] * delta / delta.norm(dim=-1, keepdim=True).clamp_min(1.)
            rho = delta.norm(dim=-1)
            route_correct = route == gt_ids[2]
            e_anchor = (anchor - label_xyz).norm(dim=-1)
            e_full = (full - label_xyz).norm(dim=-1)
            gt_rank = (lp3[0] > lp3[0].gather(0, gt_ids[2][None]).squeeze(0)).sum(0) + 1
            gt_prob = lp3[0].exp().gather(0, gt_ids[2][None]).squeeze(0)
            mask_pred = prediction['mask_logit'][0, 0].sigmoid().cpu().numpy()
            arrays = dict(normal=full.cpu().numpy(), anchor_only=anchor.cpu().numpy(),
                          oracle_pred_route=oracle.cpu().numpy(), gt_xyz_pred_mask=xyz_gt,
                          normal_gt_mask=full.cpu().numpy(), gt_xyz_gt_mask=xyz_gt)
            pose = {}
            for name in VARIANTS:
                mask_use = valid.astype(np.float32) if name.endswith('gt_mask') else mask_pred
                cv2.setRNGSeed(args.seed + index)
                estimate, points = pnp_pose(evaluator, cfg, mapped, arrays[name], mask_use)
                add_s = add_s_error(estimate, gt_pose, cad_points[obj_id], obj_id in (10, 11))
                diameter = float(model.cad_attention_head.diameters[label])
                pose[name] = dict(points=points, add_s_m=add_s,
                                  add_s_0_1d=0. if add_s is None else float(add_s < .1 * diameter),
                                  **pose_error(estimate, gt_pose))
            row = dict(index=index, object_id=int(selected[index][0]), scene_im_id=selected[index][1],
                       symmetry_branch=branch,
                       valid_pixels=int(valid.sum()), gt_mask_pixels=int(valid.size),
                       predicted_mask_pixels=int((mask_pred > cfg.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST).sum()),
                       t1_top1=float((lp1[0].argmax(0)[valid_t] == gt_ids[0][valid_t]).float().mean()),
                       t2_top1=float((lp2[0].argmax(0)[valid_t] == gt_ids[1][valid_t]).float().mean()),
                       t3_top1=float(route_correct[valid_t].float().mean()),
                       t3_top5=float((gt_rank[valid_t] <= 5).float().mean()),
                       gt_rank_median=float(gt_rank[valid_t].float().median()),
                       gt_probability_mean=float(gt_prob[valid_t].mean()),
                       representable=float((rho[valid_t] <= 1).float().mean()),
                       anchor_error_m=float(e_anchor[valid_t].mean()),
                       full_error_m=float(e_full[valid_t].mean()),
                       residual_gain_m=float((e_anchor[valid_t] - e_full[valid_t]).mean()),
                       residual_help_fraction=float((e_full[valid_t] < e_anchor[valid_t]).float().mean()),
                       pose=pose)
            rows.append(row)
        if (index + 1) % 16 == 0:
            print(f'completed {index + 1}/{len(selected)}', flush=True)
    (args.output / 'per_roi.json').write_text(json.dumps(rows, indent=2))
    scalar = [key for key in rows[0] if key in ('t1_top1', 't2_top1', 't3_top1', 't3_top5',
              'gt_rank_median', 'gt_probability_mean', 'representable', 'anchor_error_m',
              'full_error_m', 'residual_gain_m', 'residual_help_fraction')]
    aggregate = dict(sample_count=len(rows), valid_pixels=sum(r['valid_pixels'] for r in rows),
                     metrics={key: summary([row[key] for row in rows]) for key in scalar},
                     pose={name: {measure: summary([row['pose'][name][measure] for row in rows
                                      if row['pose'][name][measure] is not None])
                                   for measure in ('rotation_deg', 'translation_m', 'add_s_m', 'add_s_0_1d')}
                           for name in VARIANTS})
    (args.output / 'summary.json').write_text(json.dumps(aggregate, indent=2))
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--imagenet-weights', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--per-object', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--expected-arm')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
