import json

import cv2
import numpy as np

from research.diagnostics.exp019_epro.config import ExperimentConfig
from research.diagnostics.exp019_epro.correspondence import (
    build_correspondences,
    historical_gt_reprojection_errors,
    interpolate_xyz,
    roi2d_norm_to_pixels,
    xyz_norm_to_metric,
)
from research.diagnostics.exp019_epro.gates import evaluate_gates, spearman
from research.diagnostics.exp019_epro.evaluate import reproduction_report
from research.diagnostics.exp019_epro.ransac_solver import solve_ransac_pnp
from research.diagnostics.exp019_epro.repo_adapter import _configure
from research.diagnostics.exp019_epro.types import DiagnosticSample


def _sample():
    height = width = 4
    pred = np.full((3, height, width), 0.5, np.float32)
    gt = np.full((3, height, width), 0.75, np.float32)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    roi = np.stack([xx / 8.0, yy / 8.0], axis=0).astype(np.float32)
    return DiagnosticSample(
        target_id="synthetic",
        scene_id=1,
        im_id=1,
        instance_id=0,
        obj_id=1,
        pred_xyz_norm=pred,
        gt_xyz_norm=gt,
        roi2d_norm=roi,
        extent_m=np.array([0.2, 0.4, 0.6], np.float32),
        K=np.array([[500, 0, 4], [0, 500, 4], [0, 0, 1]], np.float32),
        image_hw=np.array([8, 8], np.float32),
        support_mask=np.ones((height, width), bool),
    )


def test_interpolation_changes_only_fixed_support():
    sample = _sample()
    support = sample.support_mask.copy()
    support[0, 0] = False
    result = interpolate_xyz(sample.pred_xyz_norm, sample.gt_xyz_norm, 0.5, support)
    np.testing.assert_array_equal(result[:, 0, 0], sample.pred_xyz_norm[:, 0, 0])
    np.testing.assert_allclose(result[:, 1, 1], 0.625)
    np.testing.assert_array_equal(
        interpolate_xyz(sample.pred_xyz_norm, sample.gt_xyz_norm, 0, support),
        sample.pred_xyz_norm,
    )


def test_coordinate_contract_and_shared_set():
    sample = _sample()
    metric = xyz_norm_to_metric(sample.gt_xyz_norm, sample.extent_m)
    np.testing.assert_allclose(metric[:, 0, 0], [0.05, 0.10, 0.15])
    pixels = roi2d_norm_to_pixels(sample.roi2d_norm, sample.image_hw)
    np.testing.assert_allclose(pixels[:, 2, 3], [3, 2])
    corr = build_correspondences(sample, sample.gt_xyz_norm)
    assert corr.n == 16
    assert corr.x3d_m.shape == (16, 3)
    assert corr.x2d_px.shape == (16, 2)


def test_historical_gt_reprojection_normalizes_nonorthogonal_lmo_rotation():
    # Raw LM-O scene 2/image 942/object 5 rotation (determinant 1.0137).
    rotation = np.array(
        [
            [-0.73724288, -0.65025438, 0.20564938],
            [-0.56502450, 0.41166663, -0.72155121],
            [0.38288973, -0.64536140, -0.66798783],
        ],
        dtype=np.float64,
    )
    translation = np.array([-0.17434816, -0.18885588, 0.73448116])
    camera = np.array([[[0.01, -0.02, 0.8], [0.03, 0.01, 0.82]]])
    K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1]], dtype=np.float64)
    image_h = camera @ K.T
    image_points = image_h[..., :2] / image_h[..., 2:]
    # Preserve the historical raw-R inverse approximation used to build GT XYZ.
    xyz = (camera - translation.reshape(1, 1, 3)) @ rotation
    errors = historical_gt_reprojection_errors(
        xyz, image_points, np.ones((1, 2), dtype=bool), rotation, translation, K
    )
    rotation_vector = cv2.Rodrigues(rotation)[0]
    expected = cv2.projectPoints(xyz.reshape(-1, 3), rotation_vector, translation, K, None)[
        0
    ].reshape(-1, 2)
    np.testing.assert_allclose(
        errors, np.linalg.norm(expected - image_points.reshape(-1, 2), axis=1)
    )


def test_historical_ransac_parameters_recover_clean_pose():
    rng = np.random.default_rng(42)
    x3d = rng.uniform(-0.1, 0.1, size=(32, 3))
    R, _ = cv2.Rodrigues(np.array([0.1, -0.05, 0.08]))
    t = np.array([0.01, -0.02, 0.8])
    K = np.array([[570, 0, 320], [0, 570, 240], [0, 0, 1]], np.float64)
    camera = x3d @ R.T + t
    uvw = camera @ K.T
    uv = uvw[:, :2] / uvw[:, 2:]
    from research.diagnostics.exp019_epro.correspondence import Correspondences

    pose = solve_ransac_pnp(Correspondences(x3d, uv, np.arange(32), K), seed=42)
    assert pose.success
    np.testing.assert_allclose(pose.R, R, atol=1e-5)
    np.testing.assert_allclose(pose.t, t, atol=1e-5)


def test_preregistered_gates():
    assert abs(spearman([0, 0.25, 0.5, 0.75, 1], [1, 2, 3, 4, 5]) - 1) < 1e-12
    cfg = ExperimentConfig()
    summary = {
        "epro": {
            "0.00": {"add": 0.55, "bop": 0.70},
            "0.25": {"add": 0.65, "bop": 0.75},
            "0.50": {"add": 0.78, "bop": 0.82},
            "0.75": {"add": 0.90, "bop": 0.91},
            "1.00": {"add": 0.99, "bop": 0.99},
        },
        "ransac": {
            "0.00": {"add": 0.53841, "bop": 0.69255},
            "0.25": {"add": 0.61592, "bop": 0.71769},
            "0.50": {"add": 0.73910, "bop": 0.78356},
            "0.75": {"add": 0.85329, "bop": 0.85392},
            "1.00": {"add": 0.99377, "bop": 0.99377},
        },
    }
    report = evaluate_gates(cfg, summary)
    assert report["gate_A_gt_xyz_wiring"]
    assert report["gate_B_geometry_responsiveness"]
    json.dumps(report)


def test_historical_reproduction_report():
    cfg = ExperimentConfig()
    summary = {
        "patch": {"0.00": {"add": 0.50242, "bop": 0.69021}},
        "ransac": {
            "0.00": {"add": 0.53841, "bop": 0.69255},
            "1.00": {"add": 0.99377, "bop": 0.99377},
        },
    }
    assert reproduction_report(cfg, summary)["status"] == "PASS"


def test_current_test_forward_exposes_dense_outputs_without_pnp():
    from pathlib import Path

    cfg = _configure(
        Path("configs/gdrn/lmo_pbr/research/_base_/lmo_gt_eval.py"),
        Path("pretrained_models/lmo_pbr/model_final_wo_optim.pth"),
        "cpu",
    )
    assert cfg.TEST.SAVE_RESULTS_ONLY
    assert not cfg.TEST.USE_PNP
