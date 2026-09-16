#!/usr/bin/env python3
"""EXP022 fixed-support correspondence/RANSAC comparison; optional EXP021 arm."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import inference_context
from mmcv import Config

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, get_out_coor, get_out_mask
from core.gdrn_modeling.models.GDRN_PCC import build_model_optimizer as build_pcc
from core.utils.my_checkpoint import MyCheckpointer
from lib.utils.mask_utils import cocosegm2mask
from research.diagnostics.exp019_epro.correspondence import historical_gt_reprojection_errors, roi2d_norm_to_pixels
from research.diagnostics.exp019_epro.repo_adapter import (
    _dataset_lookups, _depth_to_object, _prediction_valid_mask, _sample_nearest, _target_counts,
)
from research.exp020.matched_pnp_eval import (
    SEED_BASE, _full_target_total, _load_model, build_fixed_plan, solve_arm_fixed,
)
from research.exp021.matched_pnp_eval import (
    _configured, _forward as exp021_forward, _git_state, _surface_lookups,
    _symmetry_diagnostics, aggregate, export_bop,
)
from research.exp022.preflight import CONFIG_ROOT, EXPERIMENT_ID


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CONFIG = ROOT / "configs/gdrn/lmo_pbr/research/exp021_global_hierarchical_cad/a_official_eval.py"
OFFICIAL_WEIGHTS = ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth"


def _pcc_config(checkpoint: Path, device: str):
    cfg = Config.fromfile(str(CONFIG_ROOT / "train_reused.py"))
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.pretrained = False
    cfg.DATASETS.TRAIN = ("lmo_bop_test",)
    cfg.DATASETS.TEST = ("lmo_bop_test",)
    cfg.DATASETS.DET_FILES_TEST = ()
    cfg.MODEL.LOAD_DETS_TEST = False
    cfg.TEST.TEST_BBOX_TYPE = "gt"
    cfg.TEST.USE_PNP = False
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.PERSISTENT_WORKERS = False
    cfg.OUTPUT_DIR = str(ROOT / "output/experiments/EXP022-config-only")
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    return cfg


def _pcc_model(cfg, checkpoint: Path):
    model, _ = build_pcc(cfg, is_test=True)
    MyCheckpointer(model, save_dir=str(checkpoint.parent), prefix_to_remove="_module.").resume_or_load(
        str(checkpoint), resume=False
    )
    return model.eval()


def _forward_pcc(model, cfg, batch):
    with inference_context(model), torch.no_grad():
        result = model(batch["roi_img"], roi_classes=batch["roi_cls"], return_pcc_debug=True)
    xyz = get_out_coor(cfg, result["coor_x"], result["coor_y"], result["coor_z"])
    return {
        "xyz": xyz.float().cpu().numpy(),
        "vis": get_out_mask(cfg, result["mask"]).float().cpu().numpy(),
        "routes": [item.cpu().numpy() for item in result["route_ids"]],
        "beams": [item.cpu().numpy() for item in result["route_beams"]],
    }


def _routing(support, gt_xyz, predicted, beams, anchors, transform):
    result = {}
    for depth in range(4):
        size = 8 * (2 ** depth)
        coverage = cv2.resize(support.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        averaged = cv2.resize(np.where(support[..., None], gt_xyz, 0.0),
                              (size, size), interpolation=cv2.INTER_AREA)
        valid = coverage > 0.5
        target = averaged[valid] / np.maximum(coverage[valid][:, None], 1e-6)
        target = (target - transform[:3, 3]) @ transform[:3, :3]
        parent = np.zeros(len(target), dtype=np.int64)
        for ancestor in range(depth + 1):
            candidates = anchors[ancestor].reshape(-1, 8, 3)[parent]
            child = np.linalg.norm(target[:, None] - candidates, axis=-1).argmin(-1)
            parent = parent * 8 + child
        actual = predicted[depth][valid]
        beam = beams[depth][valid]
        result[f"level{depth+1}_num_valid"] = len(parent)
        result[f"level{depth+1}_top1_accuracy"] = float(np.mean(actual == parent)) if len(parent) else None
        result[f"level{depth+1}_top2_accuracy"] = float(np.mean(np.any(beam == parent[:, None], axis=-1))) if len(parent) else None
    return result


def _summary(rows, variants):
    summary = aggregate(rows, variants)
    summary["per_object"] = {}
    for obj_id in sorted({row["obj_id"] for row in rows}):
        subset = [row for row in rows if row["obj_id"] == obj_id]
        summary["per_object"][str(obj_id)] = aggregate(subset, variants)["variants"]
    for key in ("pcc",):
        for scope, data in [("all", rows)] + [(str(obj), [row for row in rows if row["obj_id"] == obj])
                                              for obj in sorted({row["obj_id"] for row in rows})]:
            routing = [row[key]["routing"] for row in data if row[key].get("routing")]
            if routing:
                metrics = {name: float(np.mean([r[name] for r in routing if r[name] is not None]))
                           for name in routing[0]}
                destination = summary["variants"][key] if scope == "all" else summary["per_object"][scope][key]
                destination.update({f"macro_mean_{name}": value for name, value in metrics.items()})
            native = [row[key]["own_mask_supplemental"] for row in data]
            destination = summary["variants"][key] if scope == "all" else summary["per_object"][scope][key]
            destination["own_mask_supplemental"] = {
                "macro_mean_num_selected": float(np.mean([item["num_selected"] for item in native])),
                "solve_successes": sum(bool(item["pose"]["success"]) for item in native),
                "solve_success_rate": float(np.mean([item["pose"]["success"] for item in native])),
            }
    return summary


def run_evaluation(official: Path, pcc_checkpoint: Path, output: Path, device: str,
                   limit: int | None, max_correspondences: int, exp021_config: Path | None = None,
                   exp021_checkpoint: Path | None = None):
    if bool(exp021_config) != bool(exp021_checkpoint):
        raise ValueError("EXP021 comparator config and checkpoint must be provided together")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg_a = _configured(OFFICIAL_CONFIG, official, device)
    cfg_p = _pcc_config(pcc_checkpoint, device)
    cfg_e = _configured(exp021_config, exp021_checkpoint, device) if exp021_config else None
    register_datasets_in_cfg(cfg_a)
    metadata = MetadataCatalog.get("lmo_bop_test")
    data_ref = ref.__dict__[metadata.ref_key]
    object_ids = [int(data_ref.obj2id[name]) for name in metadata.objs]
    official_model = _load_model(cfg_a, official, device)
    pcc_model = _pcc_model(cfg_p, pcc_checkpoint)
    exp021_model = _load_model(cfg_e, exp021_checkpoint, device) if cfg_e else None
    images, annotations = _dataset_lookups("lmo_bop_test")
    target_counts = _target_counts()
    surfaces = _surface_lookups(object_ids)
    loader = build_gdrn_test_loader(cfg_a, "lmo_bop_test", train_objs=metadata.objs, batch_size=1)
    commit, branch, dirty = _git_state()
    meta = {"status": "RUNNING", "experiment_id": EXPERIMENT_ID,
            "evaluation": "fixed_support_matched_ransac_pnp",
            "fixed_support_source": "official_pred_visible INTERSECT gt_visible INTERSECT valid_depth",
            "official_checkpoint": str(official), "pcc_checkpoint": str(pcc_checkpoint),
            "exp021_config": str(exp021_config) if exp021_config else None,
            "exp021_checkpoint": str(exp021_checkpoint) if exp021_checkpoint else None,
            "source_commit": commit, "source_branch": branch, "source_tree_dirty": dirty,
            "limit": limit, "max_correspondences": max_correspondences,
            "epro_started": False, "alpha_sweep": False}
    meta_path = output / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cv2.setNumThreads(0)
    np.random.seed(SEED_BASE)
    torch.manual_seed(SEED_BASE)
    rows, occurrences, depth_cache = [], {}, {}
    selected_per_object = {obj: 0 for obj in object_ids}
    quota = math.ceil(limit / len(object_ids)) if limit is not None and limit <= 32 else None
    max_gt_reproj = 0.0
    started = time.perf_counter()
    try:
        for inputs in loader:
            if not isinstance(inputs, list):
                inputs = [inputs]
            batch = batch_data(cfg_a, inputs, device=device, phase="test")
            a = exp021_forward(official_model, cfg_a, batch)
            p = _forward_pcc(pcc_model, cfg_p, batch)
            e = exp021_forward(exp021_model, cfg_e, batch, beam_ks=(4,)) if exp021_model else None
            flat_index = -1
            for item in inputs:
                for local_index in range(len(item["roi_img"])):
                    flat_index += 1
                    class_index = int(item["roi_cls"][local_index])
                    obj_id = object_ids[class_index]
                    scene_im_id = item["scene_im_id"][local_index]
                    target_key = (scene_im_id, obj_id)
                    occurrence = occurrences.get(target_key, 0)
                    if occurrence >= target_counts.get(target_key, 0):
                        continue
                    occurrences[target_key] = occurrence + 1
                    if quota is not None and selected_per_object[obj_id] >= quota:
                        continue
                    instance_id = int(item["inst_id"][local_index])
                    gt = annotations[(scene_im_id, instance_id)]
                    image = images[scene_im_id]
                    scene_id, im_id = map(int, scene_im_id.split("/"))
                    K = item["cam"][local_index].cpu().numpy().astype(np.float64)
                    height, width = int(item["im_H"][local_index]), int(item["im_W"][local_index])
                    extent = item["roi_extent"][local_index].cpu().numpy()
                    roi2d_norm = item["roi_coord_2d"][local_index].cpu().numpy()
                    roi2d_px = np.moveaxis(roi2d_norm_to_pixels(roi2d_norm, np.array([height, width])), 0, -1)
                    if scene_im_id not in depth_cache:
                        raw = cv2.imread(str((ROOT / image["depth_file"]).resolve()), cv2.IMREAD_UNCHANGED)
                        if raw is None:
                            raise FileNotFoundError(image["depth_file"])
                        depth_cache[scene_im_id] = raw.astype(np.float64) / float(image["depth_factor"])
                    visible_full = cocosegm2mask(gt["segmentation"], height, width).astype(bool)
                    visible_sampled, in_image = _sample_nearest(visible_full, roi2d_px)
                    R_gt, t_gt = np.asarray(gt["pose"][:, :3]), np.asarray(gt["pose"][:, 3])
                    gt_xyz, depth_valid = _depth_to_object(depth_cache[scene_im_id], roi2d_px, K, R_gt, t_gt)
                    gt_visible = visible_sampled & in_image & depth_valid
                    a_xyz = a["xyz"][0][flat_index]
                    a_vis = a["vis"][flat_index, 0]
                    support = _prediction_valid_mask(
                        a_xyz.transpose(1, 2, 0), a_vis, extent,
                        cfg_a.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST
                    ) & gt_visible
                    sanity = historical_gt_reprojection_errors(gt_xyz, roi2d_px, gt_visible, R_gt, t_gt, K)
                    if len(sanity):
                        max_gt_reproj = max(max_gt_reproj, float(sanity.max()))
                        if sanity.max() >= 0.5:
                            raise RuntimeError(f"GT XYZ reprojection {sanity.max():.4f}px >= 0.5px")
                    plan_kwargs = dict(target_id=f"{scene_id:06d}/{im_id:06d}/{obj_id:06d}/{instance_id:03d}",
                                       scene_id=scene_id, im_id=im_id, obj_id=obj_id, instance_id=instance_id,
                                       roi2d_px_map=roi2d_px, xyz_gt_m=gt_xyz, K=K, extent_m=extent,
                                       R_gt=R_gt, t_gt=t_gt, max_correspondences=max_correspondences)
                    plan = build_fixed_plan(support_mask=support, **plan_kwargs)
                    row = {"target_id": plan.target_id, "scene_id": scene_id, "im_id": im_id,
                           "obj_id": obj_id, "instance_id": instance_id,
                           "num_fixed_support": plan.num_fixed_support, "num_selected": plan.num_selected}
                    transforms = pcc_model.pcc_head.symmetry_transforms[class_index].cpu().numpy()
                    count = int(pcc_model.pcc_head.symmetry_counts[class_index])
                    transforms = transforms[:count]
                    p_xyz = p["xyz"][flat_index]
                    a_result, _, _ = _symmetry_diagnostics(plan, a_xyz, transforms, surfaces[obj_id], SEED_BASE + len(rows))
                    p_result, branch_id, per_pixel = _symmetry_diagnostics(
                        plan, p_xyz, transforms, surfaces[obj_id], SEED_BASE + len(rows)
                    )
                    if per_pixel is not None:
                        anchors = [getattr(pcc_model.pcc_head, f"level{d}_anchors")[class_index].cpu().numpy()
                                   for d in range(1, 5)]
                        p_result["routing"] = _routing(
                            support, gt_xyz, [level[flat_index] for level in p["routes"]],
                            [level[flat_index] for level in p["beams"]], anchors, transforms[branch_id]
                        )
                    native = _prediction_valid_mask(
                        p_xyz.transpose(1, 2, 0), p["vis"][flat_index, 0], extent,
                        cfg_p.MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST
                    ) & gt_visible
                    native_plan = build_fixed_plan(support_mask=native, **plan_kwargs)
                    p_result["own_mask_supplemental"] = {
                        "num_selected": native_plan.num_selected,
                        "pose": solve_arm_fixed(native_plan, p_xyz, SEED_BASE + len(rows))["pose"],
                    }
                    row.update(a=a_result, pcc=p_result)
                    if e is not None:
                        e_result, _, _ = _symmetry_diagnostics(plan, e["xyz"][4][flat_index],
                                                                transforms, surfaces[obj_id], SEED_BASE + len(rows))
                        row["exp021_k4"] = e_result
                    rows.append(row)
                    selected_per_object[obj_id] += 1
                    if limit is not None and len(rows) >= limit:
                        break
                if limit is not None and len(rows) >= limit:
                    break
            if limit is not None and len(rows) >= limit:
                break
    except Exception as exc:
        meta.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", num_targets=len(rows))
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        raise
    finally:
        official_model.cpu()
        pcc_model.cpu()
        if exp021_model:
            exp021_model.cpu()
    expected = limit if limit is not None else _full_target_total(target_counts)
    if len(rows) != expected:
        raise RuntimeError(f"Processed {len(rows)} targets; expected {expected}")
    variants = ["a", "pcc"] + (["exp021_k4"] if exp021_model else [])
    summary = _summary(rows, variants)
    with (output / "poses.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    meta.update(status="COMPLETE", num_targets=len(rows), max_gt_xyz_reprojection_px=max_gt_reproj,
                elapsed_seconds=time.perf_counter() - started)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summary


def bop_evaluate(output: Path):
    rows = [json.loads(line) for line in (output / "poses.jsonl").read_text().splitlines() if line]
    variants = [key for key in ("a", "pcc", "exp021_k4") if key in rows[0]]
    result_dir, eval_dir = output / "bop_results", output / "bop_eval"
    names = export_bop(rows, variants, result_dir)
    environment = os.environ.copy()
    toolkit, renderer = ROOT / ".local/bop_toolkit", ROOT / ".local/bop_renderer/build"
    environment.update(
        PYTHONPATH=os.pathsep.join([str(ROOT), str(toolkit), str(renderer), environment.get("PYTHONPATH", "")]),
        BOP_PATH=str((ROOT / "datasets/BOP_DATASETS").resolve()),
        BOP_RESULTS_PATH=str(result_dir), BOP_EVAL_PATH=str(eval_dir),
        BOP_RENDERER_PATH=str(renderer), BOP_NUM_WORKERS="1",
    )
    command = [
        sys.executable, str(ROOT / "lib/pysixd/scripts/eval_pose_results_more.py"),
        f"--results_path={result_dir}", f"--eval_path={eval_dir}",
        f"--result_filenames={','.join(names)}", "--renderer_type=cpp",
        "--error_types=mspd,mssd,vsd,ad,reS,teS", "--targets_filename=test_targets_bop19.json",
        "--n_top=1", "--dataset=lmo",
    ]
    subprocess.run(command, check=True, cwd=ROOT, env=environment)
    metrics = {}
    for variant in variants:
        root = eval_dir / f"{variant}_lmo-test"
        bop = json.loads((root / "scores_bop19.json").read_text())
        add_files = list(root.glob("error=ad_ntop=*/scores_th=0.100_min-visib=-1.000.json"))
        add_files += list(root.glob("error:ad_ntop:*/scores_th:0.100_min-visib:-1.000.json"))
        if len(add_files) != 1:
            raise RuntimeError(f"Expected one ADD score for {variant}, got {add_files}")
        add = json.loads(add_files[0].read_text())
        metrics[variant] = {
            "bop": float(bop["bop19_average_recall"]), "add": float(add["recall"]),
            "reS": float(bop["bop19_average_recall_reS"]),
            "teS": float(bop["bop19_average_recall_teS"]),
        }
    (output / "bop_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", type=Path, default=OFFICIAL_WEIGHTS)
    parser.add_argument("--pcc-checkpoint", type=Path, required=True)
    parser.add_argument("--exp021-config", type=Path)
    parser.add_argument("--exp021-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-correspondences", type=int, default=0)
    parser.add_argument("--bop-eval", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} requested but CUDA is unavailable")
    if args.bop_eval and args.limit is not None:
        raise ValueError("--bop-eval requires a complete matched evaluation")
    result = run_evaluation(args.official_checkpoint, args.pcc_checkpoint, args.output, args.device,
                            args.limit, args.max_correspondences, args.exp021_config, args.exp021_checkpoint)
    if args.bop_eval:
        result["bop"] = bop_evaluate(args.output.resolve())
    print(json.dumps({"status": "COMPLETE", "summary": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
