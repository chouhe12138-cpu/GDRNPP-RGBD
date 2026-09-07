"""Run frozen EXP019 Patch-PnP/RANSAC/EPro inference; never trains."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import EXPECTED_LMO_TARGETS, ExperimentConfig
from .correspondence import build_correspondences, interpolate_xyz
from .epropnp_adapter import EProPnPWorker
from .ransac_solver import solve_ransac_pnp
from .repo_adapter import build_context, iter_samples, solve_patch_pnp
from .types import failed_pose


ROOT = Path(__file__).resolve().parents[3]


def _pose_json(pose):
    pose.validate()
    return {
        "success": bool(pose.success),
        "solver": pose.solver,
        "message": pose.message,
        "num_points": int(pose.num_points),
        "R": np.asarray(pose.R, dtype=float).reshape(3, 3).tolist(),
        "t": np.asarray(pose.t, dtype=float).reshape(3).tolist(),
    }


def _git_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return commit, branch, dirty


def run(args):
    cfg = ExperimentConfig()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit, branch, dirty = _git_state()
    metadata = {
        "status": "RUNNING",
        "experiment_id": cfg.experiment_id,
        "diagnostic_only": True,
        "training": False,
        "dataset": "lmo_bop_test",
        "bbox_source": "gt",
        "expected_full_targets": EXPECTED_LMO_TARGETS,
        "alphas": list(cfg.alphas),
        "weight_mode": cfg.weight_mode,
        "fixed_support": "pred_visible_valid_xyz INTERSECT gt_visible INTERSECT valid_depth",
        "source_commit": commit,
        "source_branch": branch,
        "source_tree_dirty": dirty,
        "gdrn_config": str(args.gdrn_config),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "limit": args.limit,
    }
    metadata_path = output / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    started = time.perf_counter()
    processed = 0
    max_gt_reprojection = 0.0
    failure_counts = {"patch": 0, "ransac": 0, "epro": 0}
    try:
        context = build_context(
            cfg_path=args.gdrn_config,
            checkpoint=args.checkpoint,
            device=args.device,
            limit=args.limit,
        )
        with EProPnPWorker(cfg, args.device, root=args.epropnp_root) as epro, (
            output / "poses.jsonl"
        ).open("w", encoding="utf-8") as stream:
            for sample in iter_samples(context):
                sample.validate()
                max_gt_reprojection = max(
                    max_gt_reprojection,
                    float(sample.metadata.get("gt_xyz_max_reprojection_px", 0.0)),
                )
                for alpha_index, alpha in enumerate(cfg.alphas):
                    xyz = interpolate_xyz(
                        sample.pred_xyz_norm,
                        sample.gt_xyz_norm,
                        alpha,
                        sample.support_mask,
                    )
                    patch = solve_patch_pnp(context, sample, xyz)
                    try:
                        corr = build_correspondences(
                            sample,
                            xyz,
                            min_correspondences=cfg.min_correspondences,
                            max_correspondences=cfg.max_correspondences,
                        )
                    except ValueError as exc:
                        points = int(np.count_nonzero(sample.support_mask))
                        ransac = failed_pose("ransac_epnp", str(exc), points)
                        epro_pose = failed_pose("epropnp_v2", str(exc), points)
                    else:
                        ransac = solve_ransac_pnp(
                            corr, seed=cfg.seed + processed * len(cfg.alphas) + alpha_index
                        )
                        epro_pose = epro.solve(corr)
                    poses = {"patch": patch, "ransac": ransac, "epro": epro_pose}
                    for consumer, pose in poses.items():
                        failure_counts[consumer] += int(not pose.success)
                    row = {
                        "target_id": sample.target_id,
                        "scene_id": int(sample.scene_id),
                        "im_id": int(sample.im_id),
                        "instance_id": int(sample.instance_id),
                        "obj_id": int(sample.obj_id),
                        "alpha": float(alpha),
                        "num_support": int(np.count_nonzero(sample.support_mask)),
                        "pred_visible_points": int(sample.metadata["pred_visible_points"]),
                        "gt_visible_points": int(sample.metadata["gt_visible_points"]),
                        "gt_xyz_max_reprojection_px": float(
                            sample.metadata["gt_xyz_max_reprojection_px"]
                        ),
                        **{name: _pose_json(pose) for name, pose in poses.items()},
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                processed += 1
                print(f"EXP019 targets: {processed}", flush=True)
        expected = args.limit if args.limit else EXPECTED_LMO_TARGETS
        if processed != expected:
            raise RuntimeError(f"processed {processed} targets, expected {expected}")
        metadata.update(
            {
                "status": "COMPLETE",
                "num_targets": processed,
                "rows": processed * len(cfg.alphas),
                "failure_counts": failure_counts,
                "max_gt_xyz_reprojection_px": max_gt_reprojection,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    except Exception as exc:
        metadata.update(
            {
                "status": "FAILED",
                "num_targets": processed,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gdrn-config",
        default="configs/gdrn/lmo_pbr/research/_base_/lmo_gt_eval.py",
    )
    parser.add_argument(
        "--checkpoint", default="pretrained_models/lmo_pbr/model_final_wo_optim.pth"
    )
    parser.add_argument("--epropnp-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
