"""Engineering-only EXP019 checks: synthetic EPro solve and official model construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .config import ExperimentConfig
from .correspondence import Correspondences, project_points
from .epropnp_adapter import EProPnPWorker


def _rotation_error_deg(a, b):
    cosine = np.clip((np.trace(a @ b.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def synthetic_correspondences():
    x3d = np.array(
        [
            [-0.08, -0.05, -0.04], [0.08, -0.05, -0.04],
            [-0.08, 0.05, -0.04], [0.08, 0.05, -0.04],
            [-0.08, -0.05, 0.04], [0.08, -0.05, 0.04],
            [-0.08, 0.05, 0.04], [0.08, 0.05, 0.04],
            [0.0, 0.0, 0.0], [0.03, -0.01, 0.02],
        ],
        dtype=np.float64,
    )
    R, _ = cv2.Rodrigues(np.array([0.12, -0.18, 0.07], dtype=np.float64))
    t = np.array([0.015, -0.01, 0.72], dtype=np.float64)
    K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1]], dtype=np.float64)
    x2d = project_points(x3d, R, t, K)
    corr = Correspondences(x3d, x2d, np.arange(len(x3d)), K)
    return corr, R, t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epropnp-root", required=True)
    parser.add_argument(
        "--gdrn-config", default="configs/gdrn/lmo_pbr/research/_base_/lmo_gt_eval.py"
    )
    parser.add_argument(
        "--checkpoint", default="pretrained_models/lmo_pbr/model_final_wo_optim.pth"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = ExperimentConfig()
    corr, expected_R, expected_t = synthetic_correspondences()
    with EProPnPWorker(cfg, args.device, root=args.epropnp_root) as worker:
        pose = worker.solve(corr)
    if not pose.success:
        raise RuntimeError(pose.message)
    rotation_error = _rotation_error_deg(pose.R, expected_R)
    translation_error = float(np.linalg.norm(pose.t - expected_t))
    if rotation_error > 0.1 or translation_error > 1e-3:
        raise RuntimeError(
            f"Synthetic EPro mismatch: {rotation_error:.6f} deg, {translation_error:.6g} m"
        )
    # Import the GDRN side only after the spawned EPro worker exits.  This
    # keeps child startup small and avoids loading GDRN's unrelated lib there.
    from .repo_adapter import build_context

    context = build_context(
        args.gdrn_config, args.checkpoint, args.device, limit=1
    )
    result = {
        "status": "PASS",
        "synthetic_epro_rotation_error_deg": rotation_error,
        "synthetic_epro_translation_error_m": translation_error,
        "official_model": type(context.model).__name__,
        "dataset": context.dataset,
        "pose_corrector_absent": getattr(context.model, "pose_corrector", None) is None,
        "training_or_real_inference_executed": False,
        "epropnp_root": str(Path(args.epropnp_root).resolve()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
