from __future__ import annotations

import torch

from core.gdrn_modeling.models.heads.glm_pose_net import GLMPoseLNet
from research.diagnostics.pose_structure.common import CapturedPoseCall
from research.diagnostics.pose_structure.f_glm_diagnostic import probe_glm_forward
from research.diagnostics.pose_structure.model_access import make_model_kwargs


def _small_head() -> GLMPoseLNet:
    torch.manual_seed(11)
    head = GLMPoseLNet(
        rot_dim=6,
        num_regions=64,
        mask_attention_type="mul",
        base_channels=16,
        mid_channels=24,
        high_channels=32,
        use_region_aux=True,
        region_aux_dim=8,
        coarse_grid_size=4,
        dropout=0.0,
        geometry_channels=(8, 12, 8),
        geometry_grid_size=8,
        geometry_scale_init=0.1,
        embed_channels=64,
        attn_heads=4,
        ffn_channels=128,
        shared_channels=64,
        depth_stats_dim=4,
    )
    head.eval()
    return head


def _inputs(batch: int = 3):
    torch.manual_seed(101)
    coor = torch.rand(batch, 5, 64, 64)
    region = torch.softmax(torch.randn(batch, 64, 64, 64), dim=1)
    extents = torch.rand(batch, 3) + 0.05
    mask = torch.zeros(batch, 1, 64, 64)
    mask[:, :, 8:56, 8:56] = 1.0
    depth = torch.rand(batch, 4)
    return coor, region, extents, mask, depth


def _call(head, batch: int = 3):
    coor, region, extents, mask, depth = _inputs(batch)
    with torch.no_grad():
        raw_r, raw_t = head(
            coor,
            region=region,
            extents=extents,
            mask_attention=mask,
            depth_stats=depth,
        )
    return CapturedPoseCall(
        coor_feat=coor,
        region=region,
        extents=extents,
        mask_attention=mask,
        raw_rot=raw_r,
        raw_t=raw_t,
        depth_stats=depth,
    )


def test_model_kwargs_preserve_roi_depth_stats():
    stats = torch.rand(2, 4)
    kwargs = make_model_kwargs({"roi_depth_stats": stats}, do_loss=False)
    assert kwargs["depth_stats"] is stats


def test_probe_normal_reproduces_glm_forward():
    head = _small_head()
    call = _call(head)
    with torch.no_grad():
        raw_r, raw_t, info = probe_glm_forward(
            head,
            call,
            depth_stats=call.depth_stats,
            seed=9,
        )
    assert torch.allclose(raw_r, call.raw_rot, atol=1e-7, rtol=0.0)
    assert torch.allclose(raw_t, call.raw_t, atol=1e-7, rtol=0.0)
    assert info["weights"].shape == (3, 256)
    assert info["valid"].shape == (3, 256)


def test_support_masked_pooling_has_zero_invalid_mass():
    head = _small_head()
    call = _call(head)
    with torch.no_grad():
        _r, _t, info = probe_glm_forward(
            head,
            call,
            depth_stats=call.depth_stats,
            support_masked=True,
            seed=9,
        )
    weights, valid = info["weights"], info["valid"]
    invalid_mass = (weights * (~valid).to(weights.dtype)).sum(dim=1)
    assert torch.allclose(invalid_mass, torch.zeros_like(invalid_mass), atol=1e-7)
    assert torch.allclose(weights.sum(dim=1), torch.ones(weights.shape[0]), atol=1e-6)


def test_depth_intervention_changes_translation_not_rotation():
    head = _small_head()
    call = _call(head)
    with torch.no_grad():
        r0, t0, _ = probe_glm_forward(head, call, depth_stats=call.depth_stats, seed=9)
        rz, tz, _ = probe_glm_forward(
            head,
            call,
            depth_stats=torch.zeros_like(call.depth_stats),
            seed=9,
        )
    assert torch.equal(r0, rz)
    assert not torch.equal(t0, tz)


def test_spatial_interventions_are_finite_and_shape_preserving():
    head = _small_head()
    call = _call(head)
    variants = (
        dict(pooling="uniform"),
        dict(position_mode="shuffle"),
        dict(token_shuffle=True),
    )
    with torch.no_grad():
        for kwargs in variants:
            raw_r, raw_t, _ = probe_glm_forward(
                head,
                call,
                depth_stats=call.depth_stats,
                seed=9,
                **kwargs,
            )
            assert raw_r.shape == call.raw_rot.shape
            assert raw_t.shape == call.raw_t.shape
            assert torch.isfinite(raw_r).all()
            assert torch.isfinite(raw_t).all()
