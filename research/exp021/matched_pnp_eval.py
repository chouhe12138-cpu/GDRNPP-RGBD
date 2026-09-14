#!/usr/bin/env python3
"""EXP021 fixed-support matched RANSAC-PnP and correspondence evaluator."""

from __future__ import annotations

import argparse
import csv
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
from scipy.spatial import cKDTree

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, get_out_coor, get_out_mask
from lib.pysixd import inout
from lib.utils.mask_utils import cocosegm2mask
from research.diagnostics.exp019_epro.correspondence import (
    historical_gt_reprojection_errors,
    project_points,
    roi2d_norm_to_pixels,
)
from research.diagnostics.exp019_epro.repo_adapter import (
    _configure,
    _dataset_lookups,
    _depth_to_object,
    _prediction_valid_mask,
    _sample_nearest,
    _target_counts,
)
from research.exp020.matched_pnp_eval import (
    SEED_BASE,
    _decode_selected,
    _full_target_total,
    _load_model,
    build_fixed_plan,
    solve_arm_fixed,
)
from research.exp021.build_cad_hierarchy import DEFAULT_SAMPLES, DEFAULT_SEED, _sample_surface


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "EXP-20260914-021-global-guided-hierarchical-cad-correspondence"
DEFAULT_OFFICIAL = ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth"
CONFIG_ROOT = ROOT / "configs/gdrn/lmo_pbr/research/exp021_global_hierarchical_cad"
DEFAULT_KS = (1, 2, 4, 8)


def _git_state() -> tuple[str, str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return commit, branch, dirty


def _configured(path: Path, checkpoint: Path, device: str) -> Config:
    cfg = _configure(path, checkpoint, device)
    cfg.OUTPUT_DIR = str(ROOT / "output" / "experiments" / "EXP021-config-only")
    return cfg


def _forward(model, cfg: Config, batch: dict, beam_ks=DEFAULT_KS) -> dict:
    kwargs = dict(
        roi_classes=batch["roi_cls"],
        roi_cams=batch["roi_cam"],
        roi_whs=batch["roi_wh"],
        roi_centers=batch["roi_center"],
        resize_ratios=batch["resize_ratio"],
        roi_coord_2d=batch.get("roi_coord_2d"),
        roi_coord_2d_rel=batch.get("roi_coord_2d_rel"),
        roi_extents=batch["roi_extent"],
        return_dense_only=True,
        return_cad_debug=True,
    )
    if model.cad_head is not None:
        kwargs["cad_beam_ks"] = beam_ks
    with inference_context(model), torch.no_grad():
        output = model(batch["roi_img"], **kwargs)
    vis = get_out_mask(cfg, output["mask"].detach()).float().cpu().numpy()
    if model.cad_head is None:
        xyz = get_out_coor(
            cfg, output["coor_x"].detach(), output["coor_y"].detach(), output["coor_z"].detach()
        ).float().cpu().numpy()
        return {"xyz": {0: xyz}, "vis": vis}
    return {
        "xyz": {int(k): value.detach().float().cpu().numpy() for k, value in output["cad_xyz_by_k"].items()},
        "vis": vis,
        "coarse_logits": output["cad_coarse_logits"].detach().float().cpu().numpy(),
        "query": output["cad_query"].detach().float().cpu().numpy(),
        "fine_tokens": output["cad_fine_tokens"].detach().float().cpu().numpy(),
        "parents": {int(k): value.detach().cpu().numpy() for k, value in output["cad_parent_by_k"].items()},
        "children": {int(k): value.detach().cpu().numpy() for k, value in output["cad_child_by_k"].items()},
    }


def _surface_lookups(object_ids: list[int]) -> dict[int, tuple[cKDTree, np.ndarray]]:
    lookups = {}
    model_dir = Path(ref.lm_full.model_dir)
    for obj_id in object_ids:
        model = inout.load_ply(
            str(model_dir / f"obj_{obj_id:06d}.ply"), vertex_scale=ref.lm_full.vertex_scale
        )
        points, normals = _sample_surface(
            model, DEFAULT_SAMPLES, np.random.default_rng(DEFAULT_SEED + obj_id)
        )
        lookups[obj_id] = (cKDTree(points), normals.astype(np.float64))
    return lookups


def _symmetry_diagnostics(plan, xyz_norm, transforms, surface_lookup, seed):
    started = time.perf_counter()
    base = solve_arm_fixed(plan, xyz_norm, seed=seed)
    base["ransac_seconds"] = time.perf_counter() - started
    predicted = _decode_selected(plan, xyz_norm)
    if not len(predicted) or not np.isfinite(predicted).all():
        return base, 0, None
    tree, surface_normals = surface_lookup
    candidates = []
    for transform in transforms:
        rotation = np.asarray(transform[:3, :3], dtype=np.float64)
        translation = np.asarray(transform[:3, 3], dtype=np.float64)
        target = (plan.xyz_gt_m - translation) @ rotation
        nearest = tree.query(target, k=1)[1]
        normals = surface_normals[nearest]
        error = predicted - target
        normal_error = np.abs(np.sum(error * normals, axis=1))
        tangent_error = np.sqrt(
            np.maximum(np.sum(error * error, axis=1) - normal_error * normal_error, 0.0)
        )
        equiv_R = plan.R_gt @ rotation
        equiv_t = plan.R_gt @ translation + plan.t_gt
        projected = project_points(predicted, equiv_R, equiv_t, plan.K)
        reprojection = np.linalg.norm(projected - plan.xy2d_px, axis=1)
        candidates.append(
            {
                "target": target,
                "euclidean_mm": np.linalg.norm(error, axis=1) * 1000.0,
                "normal_mm": normal_error * 1000.0,
                "tangent_mm": tangent_error * 1000.0,
                "reprojection_px": reprojection,
            }
        )
    branch = int(np.argmin([np.median(item["euclidean_mm"]) for item in candidates]))
    chosen = candidates[branch]
    base.update(
        {
            "canonical_corr_mean_mm": base.pop("corr_err_mm"),
            "canonical_reproj_mean_px": base.pop("reproj_err_px"),
            "symmetry_branch": branch,
            "corr_mean_mm": float(np.mean(chosen["euclidean_mm"])),
            "corr_median_mm": float(np.median(chosen["euclidean_mm"])),
            "normal_median_mm": float(np.median(chosen["normal_mm"])),
            "tangent_median_mm": float(np.median(chosen["tangent_mm"])),
            "reproj_mean_px": float(np.mean(chosen["reprojection_px"])),
            "reproj_median_px": float(np.median(chosen["reprojection_px"])),
        }
    )
    return base, branch, chosen


def _routing_diagnostics(debug, index, beam_k, plan, hierarchy, branch, per_pixel):
    flat = plan.flat_indices[plan.selected]
    query = debug["query"][index, flat]
    logits = debug["coarse_logits"][index].reshape(64, -1).T[flat]
    parents = debug["parents"][beam_k][index].reshape(-1)[flat]
    children = debug["children"][beam_k][index].reshape(-1)[flat]
    transform = hierarchy["symmetry_transforms"][branch]
    target = (plan.xyz_gt_m - transform[:3, 3]) @ transform[:3, :3]
    coarse_anchors = hierarchy["coarse_anchors"]
    fine_anchors = hierarchy["fine_anchors"]
    coarse_gt = np.linalg.norm(target[:, None] - coarse_anchors[None], axis=-1).argmin(-1)
    fine_gt = np.empty(len(target), dtype=np.int64)
    fine_pred = np.empty_like(fine_gt)
    fine_tokens = debug["fine_tokens"][index]
    for parent in np.unique(coarse_gt):
        chosen = coarse_gt == parent
        fine_gt[chosen] = np.linalg.norm(
            target[chosen, None] - fine_anchors[parent][None], axis=-1
        ).argmin(-1)
        scores = query[chosen] @ fine_tokens[parent].T / math.sqrt(query.shape[-1])
        fine_pred[chosen] = scores.argmax(-1)
    topk = np.argpartition(logits, -beam_k, axis=1)[:, -beam_k:]
    coarse_correct = logits.argmax(-1) == coarse_gt
    coarse_wrong = ~coarse_correct
    result = {
        "coarse_top1_accuracy": float(np.mean(coarse_correct)),
        "coarse_topk_accuracy": float(np.mean(np.any(topk == coarse_gt[:, None], axis=1))),
        "fine_gt_parent_accuracy": float(np.mean(fine_pred == fine_gt)),
        "joint_leaf_accuracy": float(np.mean((parents == coarse_gt) & (children == fine_gt))),
        "coarse_wrong_pixels": int(coarse_wrong.sum()),
        "coarse_wrong_beam_recovery": None,
        "coarse_wrong_tangent_median_mm": None,
    }
    if np.any(coarse_wrong):
        result["coarse_wrong_beam_recovery"] = float(
            np.mean((parents[coarse_wrong] == coarse_gt[coarse_wrong]) & (children[coarse_wrong] == fine_gt[coarse_wrong]))
        )
        result["coarse_wrong_tangent_median_mm"] = float(
            np.median(per_pixel["tangent_mm"][coarse_wrong])
        )
    return result


def aggregate(rows: list[dict], variants: list[str]) -> dict:
    summary = {"num_targets": len(rows), "variants": {}}
    metrics = (
        "corr_mean_mm",
        "corr_median_mm",
        "normal_median_mm",
        "tangent_median_mm",
        "reproj_mean_px",
        "reproj_median_px",
        "ransac_seconds",
    )
    for variant in variants:
        entries = [row[variant] for row in rows]
        solved = sum(bool(item["pose"]["success"]) for item in entries)
        result = {
            "solve_successes": solved,
            "solve_failures": len(entries) - solved,
            "solve_success_rate": solved / len(entries) if entries else None,
        }
        for metric in metrics:
            values = [item[metric] for item in entries if item.get(metric) is not None]
            result[f"macro_mean_{metric}"] = float(np.mean(values)) if values else None
        routing_keys = (
            "coarse_top1_accuracy",
            "coarse_topk_accuracy",
            "fine_gt_parent_accuracy",
            "joint_leaf_accuracy",
            "coarse_wrong_beam_recovery",
            "coarse_wrong_tangent_median_mm",
        )
        for key in routing_keys:
            values = [item["routing"][key] for item in entries if item.get("routing") and item["routing"].get(key) is not None]
            if values:
                result[f"macro_mean_{key}"] = float(np.mean(values))
        summary["variants"][variant] = result
    return summary


def mechanism_gates(summary: dict, bop: dict | None = None) -> dict:
    values = summary["variants"]
    a, b = values["a"], values["b_k4"]
    ratio = lambda new, old: (new - old) / old
    b_checks = {
        "tangent_reduction_at_least_5pct": ratio(
            b["macro_mean_tangent_median_mm"], a["macro_mean_tangent_median_mm"]
        ) <= -0.05,
        "corr_not_worse_than_3pct": ratio(
            b["macro_mean_corr_median_mm"], a["macro_mean_corr_median_mm"]
        ) <= 0.03,
        "reprojection_not_worse_than_3pct": ratio(
            b["macro_mean_reproj_median_px"], a["macro_mean_reproj_median_px"]
        ) <= 0.03,
    }
    report = {"b": {"checks": b_checks, "pass": all(b_checks.values())}, "c": None}
    if "c_k4" in values and bop is not None:
        c, bop_b, bop_c = values["c_k4"], bop["b_k4"], bop["c_k4"]
        pose_changes = [ratio(bop_c[key], bop_b[key]) for key in ("bop", "add")]
        geometry_changes = [
            ratio(c[key], b[key])
            for key in (
                "macro_mean_corr_median_mm",
                "macro_mean_tangent_median_mm",
                "macro_mean_reproj_median_px",
            )
        ]
        c_checks = {
            "bop_and_add_non_decreasing": all(change >= 0.0 for change in pose_changes),
            "one_pose_metric_improves_3pct": max(pose_changes) >= 0.03,
            "one_geometry_metric_improves_3pct": min(geometry_changes) <= -0.03,
            "geometry_metrics_not_worse_than_3pct": max(geometry_changes) <= 0.03,
        }
        report["c"] = {"checks": c_checks, "pass": all(c_checks.values())}
    return report


def run_evaluation(
    official_checkpoint: Path,
    checkpoint_b: Path,
    checkpoint_c: Path | None,
    output: Path,
    device: str,
    limit: int | None,
    max_correspondences: int,
) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit, branch, dirty = _git_state()
    cfgs = {
        "a": _configured(CONFIG_ROOT / "a_official_eval.py", official_checkpoint, device),
        "b": _configured(CONFIG_ROOT / "b_hierarchical.py", checkpoint_b, device),
    }
    checkpoints = {"a": official_checkpoint, "b": checkpoint_b}
    if checkpoint_c is not None:
        cfgs["c"] = _configured(CONFIG_ROOT / "c_global.py", checkpoint_c, device)
        checkpoints["c"] = checkpoint_c
    register_datasets_in_cfg(cfgs["a"])
    metadata = MetadataCatalog.get("lmo_bop_test")
    data_ref = ref.__dict__[metadata.ref_key]
    object_ids = [int(data_ref.obj2id[name]) for name in metadata.objs]
    models = {name: _load_model(cfgs[name], path, device) for name, path in checkpoints.items()}
    hierarchy_head = models["b"].cad_head
    hierarchy_by_class = []
    for index in range(len(object_ids)):
        hierarchy_by_class.append(
            {
                name: getattr(hierarchy_head, name)[index].detach().cpu().numpy()
                for name in ("coarse_anchors", "fine_anchors", "symmetry_transforms")
            }
        )
    images, annotations = _dataset_lookups("lmo_bop_test")
    target_counts = _target_counts()
    surfaces = _surface_lookups(object_ids)
    loader = build_gdrn_test_loader(cfgs["a"], "lmo_bop_test", train_objs=metadata.objs, batch_size=1)
    metadata_dict = {
        "status": "RUNNING",
        "experiment_id": EXPERIMENT_ID,
        "evaluation": "fixed_support_matched_ransac_pnp",
        "fixed_support_source": "official_pred_visible INTERSECT gt_visible INTERSECT valid_depth",
        "beam_ks": list(DEFAULT_KS),
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "source_commit": commit,
        "source_branch": branch,
        "source_tree_dirty": dirty,
        "limit": limit,
        "max_correspondences": max_correspondences,
        "epro_started": False,
        "alpha_sweep": False,
    }
    metadata_path = output / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")
    cv2.setNumThreads(0)
    np.random.seed(SEED_BASE)
    torch.manual_seed(SEED_BASE)
    depth_cache, occurrences, rows = {}, {}, []
    selected_per_object = {obj_id: 0 for obj_id in object_ids}
    quota = math.ceil(limit / len(object_ids)) if limit is not None and limit <= 32 else None
    started = time.perf_counter()
    max_gt_reproj = 0.0
    try:
        for inputs in loader:
            if not isinstance(inputs, list):
                inputs = [inputs]
            batch = batch_data(cfgs["a"], inputs, device=device, phase="test")
            outputs = {name: _forward(models[name], cfgs[name], batch) for name in models}
            flat_index = -1
            for input_item in inputs:
                for local_index in range(len(input_item["roi_img"])):
                    flat_index += 1
                    class_index = int(input_item["roi_cls"][local_index])
                    obj_id = object_ids[class_index]
                    scene_im_id = input_item["scene_im_id"][local_index]
                    target_key = (scene_im_id, obj_id)
                    occurrence = occurrences.get(target_key, 0)
                    if occurrence >= target_counts.get(target_key, 0):
                        continue
                    occurrences[target_key] = occurrence + 1
                    if quota is not None and selected_per_object[obj_id] >= quota:
                        continue
                    instance_id = int(input_item["inst_id"][local_index])
                    gt = annotations[(scene_im_id, instance_id)]
                    image = images[scene_im_id]
                    scene_id, im_id = map(int, scene_im_id.split("/"))
                    K = input_item["cam"][local_index].cpu().numpy().astype(np.float64)
                    height, width = int(input_item["im_H"][local_index]), int(input_item["im_W"][local_index])
                    extent = input_item["roi_extent"][local_index].cpu().numpy()
                    roi2d_norm = input_item["roi_coord_2d"][local_index].cpu().numpy()
                    roi2d_px_map = np.moveaxis(
                        roi2d_norm_to_pixels(roi2d_norm, np.array([height, width])), 0, -1
                    )
                    if scene_im_id not in depth_cache:
                        raw = cv2.imread(str((ROOT / image["depth_file"]).resolve()), cv2.IMREAD_UNCHANGED)
                        if raw is None:
                            raise FileNotFoundError(image["depth_file"])
                        depth_cache[scene_im_id] = raw.astype(np.float64) / float(image["depth_factor"])
                    visible_full = cocosegm2mask(gt["segmentation"], height, width).astype(bool)
                    visible_sampled, in_image = _sample_nearest(visible_full, roi2d_px_map)
                    R_gt, t_gt = np.asarray(gt["pose"][:, :3]), np.asarray(gt["pose"][:, 3])
                    gt_xyz_m, depth_valid = _depth_to_object(
                        depth_cache[scene_im_id], roi2d_px_map, K, R_gt, t_gt
                    )
                    gt_visible = visible_sampled & in_image & depth_valid
                    ref_xyz = outputs["a"]["xyz"][0][flat_index]
                    ref_hwc = ref_xyz.transpose(1, 2, 0)
                    ref_mask = outputs["a"]["vis"][flat_index, 0]
                    support = _prediction_valid_mask(
                        ref_hwc, ref_mask, extent, cfgs["a"].MODEL.POSE_NET.GEO_HEAD.MASK_THR_TEST
                    ) & gt_visible
                    sanity = historical_gt_reprojection_errors(
                        gt_xyz_m, roi2d_px_map, gt_visible, R_gt, t_gt, K
                    )
                    if len(sanity):
                        max_gt_reproj = max(max_gt_reproj, float(sanity.max()))
                        if sanity.max() >= 0.5:
                            raise RuntimeError(f"GT XYZ reprojection {sanity.max():.4f}px >= 0.5px")
                    plan = build_fixed_plan(
                        target_id=f"{scene_id:06d}/{im_id:06d}/{obj_id:06d}/{instance_id:03d}",
                        scene_id=scene_id,
                        im_id=im_id,
                        obj_id=obj_id,
                        instance_id=instance_id,
                        support_mask=support,
                        roi2d_px_map=roi2d_px_map,
                        xyz_gt_m=gt_xyz_m,
                        K=K,
                        extent_m=extent,
                        R_gt=R_gt,
                        t_gt=t_gt,
                        max_correspondences=max_correspondences,
                    )
                    row = {
                        "target_id": plan.target_id,
                        "scene_id": scene_id,
                        "im_id": im_id,
                        "obj_id": obj_id,
                        "instance_id": instance_id,
                        "num_fixed_support": plan.num_fixed_support,
                        "num_selected": plan.num_selected,
                    }
                    hierarchy = hierarchy_by_class[class_index]
                    count = int(hierarchy_head.symmetry_counts[class_index].item())
                    transforms = hierarchy["symmetry_transforms"][:count]
                    variants = [("a", outputs["a"]["xyz"][0][flat_index], None, 0)]
                    for arm in ("b", "c"):
                        if arm in outputs:
                            variants.extend(
                                (f"{arm}_k{k}", outputs[arm]["xyz"][k][flat_index], outputs[arm], k)
                                for k in DEFAULT_KS
                            )
                    for name, xyz, debug, beam_k in variants:
                        result, sym_branch, per_pixel = _symmetry_diagnostics(
                            plan, xyz, transforms, surfaces[obj_id], SEED_BASE + len(rows)
                        )
                        if debug is not None and per_pixel is not None:
                            result["routing"] = _routing_diagnostics(
                                debug, flat_index, beam_k, plan, hierarchy, sym_branch, per_pixel
                            )
                        row[name] = result
                    rows.append(row)
                    selected_per_object[obj_id] += 1
                    if limit is not None and len(rows) >= limit:
                        break
                if limit is not None and len(rows) >= limit:
                    break
            if limit is not None and len(rows) >= limit:
                break
    except Exception as exc:
        metadata_dict.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", num_targets=len(rows))
        metadata_path.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")
        raise
    finally:
        for model in models.values():
            model.cpu()
    expected = limit if limit is not None else _full_target_total(target_counts)
    if len(rows) != expected:
        raise RuntimeError(f"processed {len(rows)} targets, expected {expected}")
    variants = ["a"] + [f"{arm}_k{k}" for arm in ("b", "c") if arm in models for k in DEFAULT_KS]
    summary = aggregate(rows, variants)
    with (output / "poses.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata_dict.update(
        status="COMPLETE",
        num_targets=len(rows),
        max_gt_xyz_reprojection_px=max_gt_reproj,
        elapsed_seconds=time.perf_counter() - started,
    )
    metadata_path.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")
    return summary


def export_bop(rows: list[dict], variants: list[str], directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=False)
    names = []
    for variant in variants:
        name = f"{variant}_lmo-test.csv"
        names.append(name)
        with (directory / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("scene_id", "im_id", "obj_id", "score", "R", "t", "time"))
            writer.writeheader()
            for row in rows:
                pose = row[variant]["pose"]
                if not pose["success"]:
                    continue
                writer.writerow(
                    {
                        "scene_id": row["scene_id"],
                        "im_id": row["im_id"],
                        "obj_id": row["obj_id"],
                        "score": 1.0,
                        "R": " ".join(f"{v:.9g}" for v in np.asarray(pose["R"]).reshape(-1)),
                        "t": " ".join(f"{v:.9g}" for v in np.asarray(pose["t"]) * 1000.0),
                        "time": -1.0,
                    }
                )
    return names


def bop_evaluate(run_dir: Path) -> dict:
    rows = [json.loads(line) for line in (run_dir / "poses.jsonl").read_text().splitlines() if line]
    variants = [key for key in rows[0] if key == "a" or key.startswith(("b_k", "c_k"))]
    result_dir, eval_dir = run_dir / "bop_results", run_dir / "bop_eval"
    names = export_bop(rows, variants, result_dir)
    environment = os.environ.copy()
    toolkit, renderer = ROOT / ".local/bop_toolkit", ROOT / ".local/bop_renderer/build"
    environment.update(
        PYTHONPATH=os.pathsep.join([str(ROOT), str(toolkit), str(renderer), environment.get("PYTHONPATH", "")]),
        BOP_PATH=str((ROOT / "datasets/BOP_DATASETS").resolve()),
        BOP_RESULTS_PATH=str(result_dir),
        BOP_EVAL_PATH=str(eval_dir),
        BOP_RENDERER_PATH=str(renderer),
        BOP_NUM_WORKERS="1",
    )
    command = [
        sys.executable,
        str(ROOT / "lib/pysixd/scripts/eval_pose_results_more.py"),
        f"--results_path={result_dir}",
        f"--eval_path={eval_dir}",
        f"--result_filenames={','.join(names)}",
        "--renderer_type=cpp",
        "--error_types=mspd,mssd,vsd,ad,reS,teS",
        "--targets_filename=test_targets_bop19.json",
        "--n_top=1",
        "--dataset=lmo",
    ]
    subprocess.run(command, check=True, cwd=ROOT, env=environment)
    result = {}
    for variant in variants:
        root = eval_dir / f"{variant}_lmo-test"
        bop = json.loads((root / "scores_bop19.json").read_text())
        add_files = list(root.glob("error=ad_ntop=*/scores_th=0.100_min-visib=-1.000.json"))
        add_files += list(root.glob("error:ad_ntop:*/scores_th:0.100_min-visib:-1.000.json"))
        if len(add_files) != 1:
            raise RuntimeError(f"Expected one ADD score for {variant}, got {add_files}")
        add = json.loads(add_files[0].read_text())
        result[variant] = {
            "bop": float(bop["bop19_average_recall"]),
            "add": float(add["recall"]),
            "reS": float(bop["bop19_average_recall_reS"]),
            "teS": float(bop["bop19_average_recall_teS"]),
        }
    (run_dir / "bop_metrics.json").write_text(json.dumps(result, indent=2))
    summary = json.loads((run_dir / "summary.json").read_text())
    (run_dir / "gate_report.json").write_text(json.dumps(mechanism_gates(summary, result), indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--checkpoint-c", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-correspondences", type=int, default=0)
    parser.add_argument("--bop-eval", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} requested but CUDA is unavailable")
    summary = run_evaluation(
        args.official_checkpoint,
        args.checkpoint_b,
        args.checkpoint_c,
        args.output,
        args.device,
        args.limit,
        args.max_correspondences,
    )
    if args.bop_eval:
        if args.limit is not None:
            raise RuntimeError("--bop-eval requires a complete run")
        bop_evaluate(args.output.resolve())
    else:
        (args.output.resolve() / "gate_report.json").write_text(
            json.dumps(mechanism_gates(summary), indent=2)
        )
    print(json.dumps({"status": "COMPLETE", "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
