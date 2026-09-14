from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.engine.engine_utils import geometry_supervision_enabled
from core.gdrn_modeling.models.heads.global_hierarchical_cad_head import (
    GlobalGuidedHierarchicalCADHead,
    load_hierarchy,
)
from core.gdrn_modeling.models.heads.top_down_doublemask_xyz_region_head import (
    TopDownDoubleMaskXyzRegionHead,
)
from research.exp021.calibrate_loss_weights import recommended_weights
from research.exp021.matched_pnp_eval import mechanism_gates
from research.exp021.profile_inference import resource_gate


ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "configs/gdrn/lmo_pbr/research/exp021_global_hierarchical_cad"


@pytest.fixture()
def hierarchy_path(tmp_path):
    rng = np.random.default_rng(7)
    coarse = rng.normal(size=(8, 64, 3)).astype(np.float32) * 0.02
    fine = coarse[:, :, None] + rng.normal(size=(8, 64, 64, 3)).astype(np.float32) * 0.002
    identity = np.tile(np.eye(4, dtype=np.float32), (8, 1, 1, 1))
    path = tmp_path / "hierarchy.npz"
    np.savez_compressed(
        path,
        object_ids=np.array([1, 5, 6, 8, 9, 10, 11, 12]),
        extents=np.ones((8, 3), np.float32) * 0.2,
        diameters=np.ones(8, np.float32) * 0.3,
        coarse_anchors=coarse,
        coarse_normals=np.tile(np.array([0, 0, 1], np.float32), (8, 64, 1)),
        coarse_radii=np.ones((8, 64), np.float32) * 0.02,
        fine_anchors=fine,
        fine_normals=np.tile(np.array([0, 0, 1], np.float32), (8, 64, 64, 1)),
        fine_radii=np.ones((8, 64, 64), np.float32) * 0.01,
        symmetry_transforms=identity,
        symmetry_counts=np.ones(8, np.int64),
        generator_version=np.asarray(1, np.int64),
        sample_count=np.asarray(200_000, np.int64),
        seed=np.asarray(20260914, np.int64),
    )
    return path


def test_hierarchy_contract(hierarchy_path):
    arrays = load_hierarchy(hierarchy_path)
    assert arrays["coarse_anchors"].shape == (8, 64, 3)
    assert arrays["fine_anchors"].shape == (8, 64, 64, 3)


def test_hierarchy_rejects_wrong_generation_contract(hierarchy_path):
    with np.load(hierarchy_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    arrays["seed"] = np.asarray(0, np.int64)
    np.savez_compressed(hierarchy_path, **arrays)
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_hierarchy(hierarchy_path)


def test_bounded_residual_stays_inside_radius(hierarchy_path):
    head = GlobalGuidedHierarchicalCADHead(str(hierarchy_path))
    raw = torch.randn(100, 3) * 100
    radius = torch.rand(100) + 0.01
    delta = head.bounded_residual(raw, radius)
    assert torch.all(torch.linalg.vector_norm(delta, dim=-1) <= radius + 1e-6)


@pytest.mark.parametrize("global_guidance", [False, True])
def test_decode_shapes_and_beams(hierarchy_path, global_guidance):
    torch.manual_seed(42)
    head = GlobalGuidedHierarchicalCADHead(
        str(hierarchy_path), use_global_guidance=global_guidance
    )
    classes = torch.tensor([0, 1])
    backbone = torch.randn(2, 1024, 8, 8)
    enhanced, bias = head.enhance_backbone(backbone, classes)
    assert enhanced.shape == backbone.shape
    if global_guidance:
        assert torch.equal(enhanced, backbone)  # zero-init residual at construction
        assert bias.shape == (2, 64)
    else:
        assert bias is None
    state, losses, _ = head(
        torch.randn(2, 256, 64, 64), classes, bias, beam_ks=(1, 2, 4, 8)
    )
    assert not losses
    assert set(state["decoded"]) == {1, 2, 4, 8}
    assert all(value["xyz_norm"].shape == (2, 3, 64, 64) for value in state["decoded"].values())


def test_teacher_forced_losses_are_foreground_only_and_backward(hierarchy_path):
    torch.manual_seed(42)
    head = GlobalGuidedHierarchicalCADHead(str(hierarchy_path))
    feature = torch.randn(1, 256, 8, 8, requires_grad=True)
    gt = torch.rand(1, 3, 8, 8) * 0.2 + 0.4
    mask = torch.zeros(1, 8, 8)
    mask[:, 2:6, 2:6] = 1
    _state, losses, stats = head(feature, torch.tensor([0]), beam_ks=(4,), gt_xyz_norm=gt, gt_mask=mask)
    assert set(losses) == {"loss_cad_coarse", "loss_cad_fine", "loss_cad_xyz"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert int(stats["cad_foreground_pixels"]) == 16
    sum(losses.values()).backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()


def test_symmetry_targets_use_full_se3(hierarchy_path):
    head = GlobalGuidedHierarchicalCADHead(str(hierarchy_path))
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = torch.tensor([0.01, -0.02, 0.03])
    head.symmetry_counts[0] = 1
    head.symmetry_transforms[0, 0, :3, :3] = rotation
    head.symmetry_transforms[0, 0, :3, 3] = translation
    point = torch.tensor([[[0.04, 0.02, 0.03]]])
    target = head._symmetry_targets(point, 0)[0]
    assert torch.allclose(target, (point.reshape(-1, 3) - translation) @ rotation)


def test_geo_head_feature_api_is_backward_compatible():
    head = TopDownDoubleMaskXyzRegionHead(in_dim=32, feat_dim=32, num_gn_groups=8)
    x = torch.randn(1, 32, 8, 8)
    original = head(x)
    with_features = head(x, return_features=True)
    assert len(original) == 6 and len(with_features) == 7
    assert all(torch.equal(a, b) for a, b in zip(original, with_features[:6]))
    assert with_features[-1].shape == (1, 32, 64, 64)


def test_configs_isolate_b_and_c_and_enable_geometry():
    b = Config.fromfile(str(CONFIG_ROOT / "b_hierarchical.py"))
    c = Config.fromfile(str(CONFIG_ROOT / "c_global.py"))
    assert geometry_supervision_enabled(b) and geometry_supervision_enabled(c)
    assert b.INPUT.WITH_DEPTH is False and c.INPUT.WITH_DEPTH is False
    assert b.MODEL.POSE_NET.BACKBONE.FREEZE
    assert b.MODEL.POSE_NET.GEO_HEAD.FREEZE
    assert b.MODEL.POSE_NET.PNP_NET.FREEZE
    assert b.MODEL.POSE_NET.CAD_HEAD.INIT_CFG.coarse_loss_weight == 0.125
    assert b.MODEL.POSE_NET.CAD_HEAD.INIT_CFG.fine_loss_weight == 1.0
    assert b.MODEL.POSE_NET.CAD_HEAD.INIT_CFG.xyz_loss_weight == 16.0
    b_dict, c_dict = b.to_dict(), c.to_dict()
    b_global = b_dict["MODEL"]["POSE_NET"]["CAD_HEAD"]["INIT_CFG"].pop("use_global_guidance")
    c_global = c_dict["MODEL"]["POSE_NET"]["CAD_HEAD"]["INIT_CFG"].pop("use_global_guidance")
    assert (b_global, c_global) == (False, True)
    for key in ("OUTPUT_DIR",):
        b_dict.pop(key)
        c_dict.pop(key)
    assert b_dict == c_dict


def test_b_and_c_shared_parameters_start_identically(hierarchy_path):
    torch.manual_seed(42)
    b = GlobalGuidedHierarchicalCADHead(str(hierarchy_path), use_global_guidance=False)
    torch.manual_seed(42)
    c = GlobalGuidedHierarchicalCADHead(str(hierarchy_path), use_global_guidance=True)
    c_state = c.state_dict()
    assert all(torch.equal(value, c_state[name]) for name, value in b.state_dict().items())


def test_gradient_weight_recommendation_and_resource_gate():
    assert recommended_weights({"a": 1.0, "b": 2.0, "c": 5.0}) == {
        "a": 1.0,
        "b": 1.0,
        "c": 1.0,
    }
    weights = recommended_weights({"a": 0.01, "b": 1.0, "c": 10.0})
    assert weights["a"] > weights["b"] >= weights["c"]
    reports = [
        {"name": "a", "latency_median_ms": 10.0, "peak_allocated_bytes": 100},
        {
            "name": "c_k4",
            "latency_increase_vs_a": 0.25,
            "memory_increase_vs_a": 0.20,
        },
    ]
    assert resource_gate(reports)["pass"] is True
    reports[1]["latency_increase_vs_a"] = 0.251
    assert resource_gate(reports)["pass"] is False


def test_gate_policy():
    summary = {
        "variants": {
            "a": {
                "macro_mean_tangent_median_mm": 10.0,
                "macro_mean_corr_median_mm": 10.0,
                "macro_mean_reproj_median_px": 10.0,
            },
            "b_k4": {
                "macro_mean_tangent_median_mm": 9.4,
                "macro_mean_corr_median_mm": 10.2,
                "macro_mean_reproj_median_px": 10.2,
            },
        }
    }
    report = mechanism_gates(summary)
    assert report["b"]["pass"] is True and report["c"] is None
