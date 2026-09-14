from __future__ import annotations

import math
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


def _reference_loss(head, decoded_state, roi_classes, gt_xyz_norm, gt_mask):
    """Small-input oracle preserving the original per-instance implementation."""
    query = decoded_state["query"]
    coarse_logits = decoded_state["coarse_logits"]
    fine_tokens = decoded_state["fine_tokens_unique"].index_select(
        0, decoded_state["descriptor_inverse"]
    )
    coarse_anchors = head._select(head.coarse_anchors, roi_classes)
    fine_anchors = head._select(head.fine_anchors, roi_classes)
    fine_radii = head._select(head.fine_radii, roi_classes)
    extents = head._select(head.extents, roi_classes)
    component_losses = []
    selected_branches = []
    for batch_index in range(query.shape[0]):
        mask = gt_mask[batch_index].reshape(-1) > 0.5
        if not torch.any(mask):
            zero = query[batch_index].sum() * 0.0
            component_losses.append(torch.stack([zero, zero, zero]))
            selected_branches.append(0)
            continue
        gt_metric = (
            gt_xyz_norm[batch_index].permute(1, 2, 0).reshape(-1, 3) - 0.5
        ) * extents[batch_index]
        branch_losses = []
        for target_all in head._symmetry_targets(
            gt_metric, int(roi_classes[batch_index])
        ):
            target = target_all[mask]
            q = query[batch_index, mask]
            coarse_label = torch.cdist(target, coarse_anchors[batch_index]).argmin(-1)
            coarse_loss = torch.nn.functional.cross_entropy(
                coarse_logits[batch_index, mask], coarse_label
            )
            fine_logits = q.new_empty((q.shape[0], 64))
            fine_label = torch.empty(q.shape[0], dtype=torch.long, device=q.device)
            for parent in coarse_label.unique(sorted=True):
                parent_int = int(parent.item())
                chosen = coarse_label == parent
                anchors = fine_anchors[batch_index, parent_int]
                fine_label[chosen] = torch.cdist(target[chosen], anchors).argmin(-1)
                fine_logits[chosen] = (
                    q[chosen] @ fine_tokens[batch_index, parent_int].T
                ) / math.sqrt(head.token_dim)
            fine_loss = torch.nn.functional.cross_entropy(fine_logits, fine_label)
            anchor = fine_anchors[batch_index, coarse_label, fine_label]
            radius = fine_radii[batch_index, coarse_label, fine_label]
            leaf_token = fine_tokens[batch_index, coarse_label, fine_label]
            residual = head.bounded_residual(
                head.residual_head(torch.cat([q, leaf_token], dim=-1)), radius
            )
            pred_norm = (anchor + residual) / extents[batch_index] + 0.5
            target_norm = target / extents[batch_index] + 0.5
            xyz_loss = torch.nn.functional.smooth_l1_loss(
                pred_norm,
                target_norm,
                beta=head.xyz_smooth_l1_beta,
                reduction="mean",
            )
            branch_losses.append(torch.stack([coarse_loss, fine_loss, xyz_loss]))
        branches = torch.stack(branch_losses)
        totals = (
            branches[:, 0] * head.coarse_loss_weight
            + branches[:, 1] * head.fine_loss_weight
            + branches[:, 2] * head.xyz_loss_weight
        )
        selected = int(totals.detach().argmin().item())
        component_losses.append(branches[selected])
        selected_branches.append(selected)
    components = torch.stack(component_losses).mean(0)
    return {
        "loss_cad_coarse": components[0] * head.coarse_loss_weight,
        "loss_cad_fine": components[1] * head.fine_loss_weight,
        "loss_cad_xyz": components[2] * head.xyz_loss_weight,
    }, torch.tensor(selected_branches)


@pytest.fixture()
def hierarchy_path(tmp_path):
    rng = np.random.default_rng(7)
    coarse = rng.normal(size=(8, 64, 3)).astype(np.float32) * 0.02
    fine = coarse[:, :, None] + rng.normal(size=(8, 64, 64, 3)).astype(np.float32) * 0.002
    identity = np.tile(np.eye(4, dtype=np.float32), (8, 2, 1, 1))
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
    assert state["fine_tokens"].shape == (2, 64, 64, 256)
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


def test_vectorized_loss_matches_reference_values_and_gradients(hierarchy_path):
    torch.manual_seed(23)
    reference = GlobalGuidedHierarchicalCADHead(
        str(hierarchy_path),
        coarse_loss_weight=0.125,
        fine_loss_weight=1.0,
        xyz_loss_weight=16.0,
    )
    optimized = GlobalGuidedHierarchicalCADHead(
        str(hierarchy_path),
        coarse_loss_weight=0.125,
        fine_loss_weight=1.0,
        xyz_loss_weight=16.0,
    )
    optimized.load_state_dict(reference.state_dict())
    for head in (reference, optimized):
        head.symmetry_counts[0] = 2
        head.symmetry_transforms[0, 1, :3, :3] = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        )

    classes = torch.tensor([0, 0, 5])
    gt = torch.rand(3, 3, 8, 8) * 0.2 + 0.4
    mask = torch.zeros(3, 8, 8)
    mask[0, 1:7, 2:6] = 1
    mask[1, 2:6, 1:7] = 1
    mask[2] = 0  # preserve the original zero-loss empty-instance behavior
    reference_feature = torch.randn(3, 256, 8, 8, requires_grad=True)
    optimized_feature = reference_feature.detach().clone().requires_grad_(True)

    reference_state = reference._encoded_state(reference_feature, classes)[0]
    reference_losses, reference_selected = _reference_loss(
        reference, reference_state, classes, gt, mask
    )
    optimized_state = optimized._encoded_state(optimized_feature, classes)[0]
    optimized_losses, optimized_stats = optimized.loss(
        optimized_state, classes, gt, mask
    )
    for name in reference_losses:
        assert torch.allclose(
            optimized_losses[name], reference_losses[name], rtol=2e-5, atol=2e-6
        )
    assert torch.equal(
        optimized_stats["selected_symmetry_branch_mean"],
        reference_selected.float().mean(),
    )

    sum(reference_losses.values()).backward()
    sum(optimized_losses.values()).backward()
    assert torch.allclose(
        optimized_feature.grad, reference_feature.grad, rtol=3e-5, atol=3e-6
    )
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in optimized.named_parameters():
        reference_grad = reference_parameters[name].grad
        if reference_grad is None:
            assert parameter.grad is None
        else:
            assert torch.allclose(
                parameter.grad, reference_grad, rtol=3e-5, atol=3e-6
            ), name


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
