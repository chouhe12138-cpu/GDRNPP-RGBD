from __future__ import annotations

import inspect
import io

import pytest
import torch

from core.gdrn_modeling.models.heads.exp013_geometry_pnp_net import (
    GeometryAttentionResidualPnPNet,
    RTDecoupledGeometryPnPNet,
    XYZResidualBypassPnPNet,
)
from core.gdrn_modeling.models.net_factory import HEADS


HEADS_UNDER_TEST = (
    XYZResidualBypassPnPNet,
    GeometryAttentionResidualPnPNet,
    RTDecoupledGeometryPnPNet,
)


def _inputs(batch: int = 2, size: int = 64):
    torch.manual_seed(101)
    coor = torch.rand(batch, 5, size, size)
    region = torch.softmax(torch.randn(batch, 64, size, size), dim=1)
    extents = torch.rand(batch, 3) + 0.05
    mask = (torch.rand(batch, 1, size, size) > 0.2).float()
    return coor, region, extents, mask


def _small_kwargs():
    return dict(
        num_regions=64,
        rot_dim=6,
        base_channels=16,
        mid_channels=24,
        high_channels=32,
        region_aux_dim=8,
        coarse_grid_size=4,
        geometry_channels=(8, 12, 8),
        geometry_grid_size=8,
    )


@pytest.mark.parametrize("head_type", HEADS_UNDER_TEST)
def test_forward_shape_finite_and_parameter_gradients(head_type):
    model = head_type(**_small_kwargs())
    coor, region, extents, mask = _inputs()
    rotation, translation = model(coor, region, extents, mask)
    assert rotation.shape == (2, 6)
    assert translation.shape == (2, 3)
    assert torch.isfinite(rotation).all()
    assert torch.isfinite(translation).all()
    (rotation.square().mean() + translation.square().mean()).backward()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)


@pytest.mark.parametrize("head_type", HEADS_UNDER_TEST)
def test_masked_correspondence_values_cannot_leak(head_type):
    model = head_type(**_small_kwargs()).eval()
    coor, region, extents, mask = _inputs(batch=1)
    mask[:, :, :16, :20] = 0
    changed_coor = coor.clone()
    changed_region = region.clone()
    changed_coor[:, :, :16, :20] = 1.0e4
    changed_region[:, :, :16, :20] = -1.0e4
    with torch.no_grad():
        expected = model(coor, region, extents, mask)
        actual = model(changed_coor, changed_region, extents, mask)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_a_geometry_encoder_has_no_region_input_and_is_region_invariant():
    model = XYZResidualBypassPnPNet(**_small_kwargs()).eval()
    assert "region" not in inspect.signature(model._encode_geometry).parameters
    coor, region, extents, mask = _inputs(batch=1)
    metric, _, support = model._validate_and_mask_inputs(coor, region, extents, mask)
    with torch.no_grad():
        first, _ = model._encode_geometry(metric, support)
        second, _ = model._encode_geometry(metric, support)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_a_geometry_scale_zero_strictly_closes_residual():
    model = XYZResidualBypassPnPNet(**_small_kwargs()).eval()
    coor, region, extents, mask = _inputs(batch=1)
    with torch.no_grad():
        model.geometry_scale.zero_()
        metric, masked_region, _ = model._validate_and_mask_inputs(
            coor, region, extents, mask
        )
        latent, _, _, _ = model._encode_main(metric, masked_region)
        expected = model.pose_rotation(latent), model.pose_translation(latent)
        actual = model(coor, region, extents, mask)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_b_attention_is_finite_normalized_and_masks_invalid_neighbours():
    model = GeometryAttentionResidualPnPNet(**_small_kwargs()).eval()
    coor, region, extents, mask = _inputs(batch=1)
    mask[:, :, 24:40, 24:40] = 0
    metric, _, support = model._validate_and_mask_inputs(coor, region, extents, mask)
    with torch.no_grad():
        model._encode_geometry(metric, support)
    weights = model.geometry_attention.last_weights
    assert weights is not None and torch.isfinite(weights).all()
    sums = weights.sum(dim=1)
    valid_center = (
        torch.nn.functional.interpolate(support, size=sums.shape[1:], mode="nearest")[
            :, 0
        ]
        > 0
    )
    torch.testing.assert_close(sums[valid_center], torch.ones_like(sums[valid_center]))
    torch.testing.assert_close(
        sums[~valid_center], torch.zeros_like(sums[~valid_center])
    )
    assert torch.count_nonzero(weights[:, :, ~valid_center[0]]) == 0


def test_b_attention_scale_zero_matches_a_with_same_common_weights():
    torch.manual_seed(7)
    model_a = XYZResidualBypassPnPNet(**_small_kwargs()).eval()
    torch.manual_seed(8)
    model_b = GeometryAttentionResidualPnPNet(**_small_kwargs()).eval()
    incompatible = model_b.load_state_dict(model_a.state_dict(), strict=False)
    assert all(
        key.startswith(("geometry_attention.", "attention_scale"))
        for key in incompatible.missing_keys
    )
    assert not incompatible.unexpected_keys
    with torch.no_grad():
        model_b.attention_scale.zero_()
    inputs = _inputs(batch=1)
    with torch.no_grad():
        expected = model_a(*inputs)
        actual = model_b(*inputs)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_c_has_independent_late_paths_and_both_receive_gradients():
    model = RTDecoupledGeometryPnPNet(**_small_kwargs())
    assert isinstance(model, XYZResidualBypassPnPNet)
    assert not isinstance(model, GeometryAttentionResidualPnPNet)
    assert not hasattr(model, "geometry_attention")
    assert not hasattr(model, "attention_scale")
    assert not hasattr(model, "pose_fc1")
    assert not hasattr(model, "pose_fc2")
    inputs = _inputs(batch=1)
    rotation, translation = model(*inputs)
    rotation.sum().backward(retain_graph=True)
    assert model.rotation_fc1.weight.grad is not None
    assert model.translation_fc1.weight.grad is None
    assert model.geometry_scale_r.grad is not None
    assert model.geometry_scale_t.grad is None
    assert torch.isfinite(model.geometry_scale_r.grad).all()
    model.zero_grad(set_to_none=True)
    translation.sum().backward()
    assert model.translation_fc1.weight.grad is not None
    assert model.rotation_fc1.weight.grad is None
    assert model.geometry_scale_t.grad is not None
    assert model.geometry_scale_r.grad is None
    assert torch.isfinite(model.geometry_scale_t.grad).all()


@pytest.mark.parametrize("head_type", HEADS_UNDER_TEST)
def test_strict_checkpoint_roundtrip_is_value_exact(head_type):
    torch.manual_seed(19)
    model = head_type(**_small_kwargs()).eval()
    inputs = _inputs(batch=1)
    with torch.no_grad():
        expected = model(*inputs)
    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    restored = head_type(**_small_kwargs()).eval()
    restored.load_state_dict(torch.load(payload, map_location="cpu"), strict=True)
    with torch.no_grad():
        actual = restored(*inputs)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_all_three_heads_are_registered():
    assert HEADS["XYZResidualBypassPnPNet"] is XYZResidualBypassPnPNet
    assert HEADS["GeometryAttentionResidualPnPNet"] is GeometryAttentionResidualPnPNet
    assert HEADS["RTDecoupledGeometryPnPNet"] is RTDecoupledGeometryPnPNet
