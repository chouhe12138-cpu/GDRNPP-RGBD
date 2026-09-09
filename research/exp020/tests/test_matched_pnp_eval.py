"""CPU-only unit tests for the EXP020 matched PnP/RANSAC evaluator.

These tests validate the review-fix protocol without a GPU or real data:
the fixed support is built once from the reference geometry and A/B can only
swap their predicted XYZ on those frozen indices (no per-arm support rebuild,
no EPro, no alpha sweep, no Patch-PnP as the primary consumer).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from research.exp020.matched_pnp_eval import (
    ARMS,
    MIN_FIXED_SUPPORT,
    build_fixed_plan,
    solve_arm_fixed,
)

ROOT = Path(__file__).resolve().parents[3]


def _synthetic_target(h: int = 8, w: int = 8, support_rows: int = 2):
    """A GT-consistent synthetic target: point (u,v) projects back to (u,v).

    z varies slightly with the row so EPNP has non-coplanar structure.
    ``xyz_gt_m`` is stored in the *object* frame, exactly like the depth
    back-projection in the EXP019 protocol (camera - t rotated by R).
    """
    fx = fy = 100.0
    cx = 100.0
    cy = 100.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    extent = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    zz = 2.0 + 0.1 * (vv % 2)  # avoid coplanarity
    xx = (uu - cx) / fx * zz
    yy = (vv - cy) / fy * zz
    xyz_cam_m = np.stack((xx, yy, zz), axis=-1)  # [H,W,3] camera frame
    # R = I, so object frame = camera frame - t.
    xyz_obj_m = xyz_cam_m - t.reshape(1, 1, 3)
    xyz_norm = xyz_obj_m / extent.reshape(1, 1, 3) + 0.5  # [H,W,3]

    support = np.zeros((h, w), dtype=bool)
    support[support_rows : h - support_rows, support_rows : w - support_rows] = True
    roi2d_px_map = np.stack((uu, vv), axis=-1)
    return {
        "support_mask": support,
        "roi2d_px_map": roi2d_px_map,
        "xyz_gt_m": xyz_obj_m,
        "K": K,
        "R_gt": R,
        "t_gt": t,
        "extent_m": extent,
        "xyz_norm": np.ascontiguousarray(np.moveaxis(xyz_norm, -1, 0)),  # [3,H,W]
    }


def _make_plan(t):
    return build_fixed_plan(
        target_id="000000/000000/000001/000",
        scene_id=0,
        im_id=0,
        obj_id=1,
        instance_id=0,
        support_mask=t["support_mask"],
        roi2d_px_map=t["roi2d_px_map"],
        xyz_gt_m=t["xyz_gt_m"],
        K=t["K"],
        extent_m=t["extent_m"],
        R_gt=t["R_gt"],
        t_gt=t["t_gt"],
    )


def test_fixed_support_plan_has_stable_flat_and_2d_points():
    raw = _synthetic_target()
    plan = _make_plan(raw)
    plan2 = _make_plan(_synthetic_target())
    assert plan.num_fixed_support == int(raw["support_mask"].sum())
    assert plan.num_selected == plan.num_fixed_support
    # Identical reference geometry must give bit-identical frozen support.
    assert np.array_equal(plan.flat_indices, plan2.flat_indices)
    assert np.array_equal(plan.selected, plan2.selected)
    assert np.array_equal(plan.xy2d_px, plan2.xy2d_px)
    assert plan.xy2d_px.shape[0] >= MIN_FIXED_SUPPORT


def test_same_checkpoint_as_a_and_b_gives_identical_solver_output():
    # The A==B smoke equivalent at the unit level: identical XYZ maps must
    # produce identical RANSAC poses and identical metrics on the frozen plan.
    t = _synthetic_target()
    plan = _make_plan(t)
    result_a = solve_arm_fixed(plan, t["xyz_norm"], seed=20260730)
    result_b = solve_arm_fixed(plan, t["xyz_norm"], seed=20260730)
    assert result_a["pose"]["success"] is result_b["pose"]["success"]
    if result_a["pose"]["success"]:
        assert np.array_equal(
            np.asarray(result_a["pose"]["R"]), np.asarray(result_b["pose"]["R"])
        )
        assert np.array_equal(
            np.asarray(result_a["pose"]["t"]), np.asarray(result_b["pose"]["t"])
        )
    assert result_a["corr_err_mm"] == pytest.approx(result_b["corr_err_mm"])
    assert result_a["reproj_err_px"] == pytest.approx(result_b["reproj_err_px"])


def test_arm_xyz_swap_never_rebuilds_the_fixed_support():
    # A/B are allowed to replace only predicted XYZ.  Even if one arm emits a
    # non-finite value exactly on a fixed-support pixel, the solver must fail
    # on the FULL frozen set (num_points unchanged) instead of silently
    # rebuilding a smaller per-arm support.
    t = _synthetic_target()
    plan = _make_plan(t)
    bad_xyz = t["xyz_norm"].copy()
    flat = int(plan.flat_indices[0])
    h, w = plan.map_h, plan.map_w
    bad_xyz[:, flat // w, flat % w] = float("nan")
    result = solve_arm_fixed(plan, bad_xyz, seed=20260730)
    assert result["pose"]["success"] is False
    assert result["pose"]["message"] == "nonfinite_fixed_support_xyz"
    assert result["pose"]["num_points"] == plan.num_selected


def test_gt_consistent_xyz_has_near_zero_correspondence_and_reprojection_error():
    t = _synthetic_target()
    plan = _make_plan(t)
    result = solve_arm_fixed(plan, t["xyz_norm"], seed=20260730)
    # GT-consistent producer geometry: metric error and GT-pose reprojection
    # error are both at the float floor, whatever the RANSAC outcome.
    assert result["corr_err_mm"] == pytest.approx(0.0, abs=1e-6)
    assert result["reproj_err_px"] == pytest.approx(0.0, abs=1e-6)


def test_row_has_no_epro_patch_or_alpha_keys():
    t = _synthetic_target()
    plan = _make_plan(t)
    arm_result = solve_arm_fixed(plan, t["xyz_norm"], seed=20260730)
    keys = set(arm_result)
    forbidden = {"epro", "patch", "alpha"}
    assert forbidden.isdisjoint(keys)
    assert keys == {"pose", "corr_err_mm", "reproj_err_px"}


def test_cli_has_no_epropnp_root_flag():
    # The evaluator must not even advertise the EXP019 EPro root flag.
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.exp020.matched_pnp_eval",
            "--help",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={"PYTHONPATH": str(ROOT), "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert out.returncode == 0, out.stderr
    assert "--epropnp-root" not in out.stdout
    assert "--reference-checkpoint" in out.stdout
    assert "--checkpoint-a" in out.stdout
    assert "--checkpoint-b" in out.stdout
    assert "--limit" in out.stdout
