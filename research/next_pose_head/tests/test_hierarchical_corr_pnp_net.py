from pathlib import Path

import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.models.heads.hierarchical_corr_pnp_net import (
    HierarchicalCorrespondencePnPNet,
)
from core.gdrn_modeling.models.model_utils import get_pnp_net
from core.gdrn_modeling.models.net_factory import HEADS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/gdrn/lmo_pbr/research/exp012_hierarchical_corr_head/train.py"
)


def _inputs(batch=2, height=16, width=16):
    torch.manual_seed(1201)
    coor_feat = torch.rand(batch, 5, height, width)
    region = torch.softmax(torch.randn(batch, 64, height, width), dim=1)
    mask = torch.rand(batch, 1, height, width)
    extents = torch.rand(batch, 3) + 0.1
    return coor_feat, region, mask, extents


def test_head_is_registered_and_has_expected_interface():
    assert HEADS["HierarchicalCorrespondencePnPNet"] is HierarchicalCorrespondencePnPNet
    head = HierarchicalCorrespondencePnPNet(
        num_regions=64,
        rot_dim=6,
        mask_attention_type="mul",
        use_region_aux=True,
    )
    coor_feat, region, mask, extents = _inputs()
    rotation, translation = head(
        coor_feat,
        region=region,
        mask_attention=mask,
        extents=extents,
    )
    assert rotation.shape == (2, 6)
    assert translation.shape == (2, 3)
    assert torch.isfinite(rotation).all()
    assert torch.isfinite(translation).all()


def test_head_supports_no_region_variant_without_changing_public_interface():
    head = HierarchicalCorrespondencePnPNet(
        num_regions=64,
        rot_dim=6,
        mask_attention_type="mul",
        use_region_aux=False,
    )
    coor_feat, _, mask, extents = _inputs()
    rotation, translation = head(
        coor_feat,
        region=None,
        mask_attention=mask,
        extents=extents,
    )
    assert rotation.shape == (2, 6)
    assert translation.shape == (2, 3)


def test_head_rejects_missing_required_support_inputs():
    head = HierarchicalCorrespondencePnPNet(mask_attention_type="mul")
    coor_feat, region, _, extents = _inputs()
    with pytest.raises(ValueError, match="mask_attention"):
        head(coor_feat, region=region, extents=extents)

    with pytest.raises(ValueError, match="region"):
        head(coor_feat, region=None, mask_attention=torch.ones(2, 1, 16, 16), extents=extents)


def test_head_forward_backward_has_finite_gradients():
    head = HierarchicalCorrespondencePnPNet(mask_attention_type="mul")
    coor_feat, region, mask, extents = _inputs()
    rotation, translation = head(
        coor_feat,
        region=region,
        mask_attention=mask,
        extents=extents,
    )
    loss = rotation.square().mean() + translation.square().mean()
    loss.backward()
    trainable = [parameter for parameter in head.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)


def test_region_auxiliary_branch_is_zero_started():
    head = HierarchicalCorrespondencePnPNet(use_region_aux=True)
    assert head.region_scale.item() == 0.0


def test_zero_support_prevents_background_correspondence_leakage():
    torch.manual_seed(1202)
    head = HierarchicalCorrespondencePnPNet(use_region_aux=True).eval()
    head.region_scale.data.fill_(0.5)
    coor_a, region_a, _, extents = _inputs(batch=1)
    coor_b = coor_a.clone()
    region_b = region_a.clone()
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1.0
    outside_coor = (mask == 0).expand_as(coor_b)
    outside_region = (mask == 0).expand_as(region_b)
    coor_b[outside_coor] = torch.randn_like(coor_b)[outside_coor] * 10.0
    region_b[outside_region] = torch.randn_like(region_b)[outside_region] * 10.0

    with torch.no_grad():
        output_a = head(coor_a, region_a, extents, mask)
        output_b = head(coor_b, region_b, extents, mask)
    assert all(torch.equal(left, right) for left, right in zip(output_a, output_b))


def test_coarse_grid_retains_spatial_arrangement_response():
    torch.manual_seed(1203)
    head = HierarchicalCorrespondencePnPNet(use_region_aux=False).eval()
    coor, _, _, extents = _inputs(batch=1)
    mask = torch.ones(1, 1, 16, 16)
    rearranged = coor.flip(dims=(3,))
    with torch.no_grad():
        output = head(coor, None, extents, mask)
        changed = head(rearranged, None, extents, mask)
    assert any(not torch.equal(left, right) for left, right in zip(output, changed))


def test_strict_checkpoint_round_trip_preserves_pose_outputs():
    torch.manual_seed(1204)
    head = HierarchicalCorrespondencePnPNet().eval()
    coor, region, mask, extents = _inputs()
    with torch.no_grad():
        expected = head(coor, region, extents, mask)
    reloaded = HierarchicalCorrespondencePnPNet().eval()
    incompatible = reloaded.load_state_dict(head.state_dict(), strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    with torch.no_grad():
        actual = reloaded(coor, region, extents, mask)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))


def test_parameter_count_is_stable():
    head = HierarchicalCorrespondencePnPNet(
        base_channels=64,
        mid_channels=96,
        high_channels=128,
        region_aux_dim=16,
    )
    parameter_count = sum(parameter.numel() for parameter in head.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in head.parameters() if parameter.requires_grad
    )
    assert parameter_count == trainable_count == 868_746
    assert parameter_count < 1_500_000


def test_exp012_config_constructs_through_model_utils():
    cfg = Config.fromfile(str(CONFIG_PATH))
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    pose_cfg = cfg.MODEL.POSE_NET
    assert pose_cfg.BACKBONE.FREEZE
    assert pose_cfg.GEO_HEAD.FREEZE
    assert not pose_cfg.PNP_NET.FREEZE
    assert pose_cfg.PNP_NET.INIT_CFG.type == "HierarchicalCorrespondencePnPNet"
    assert pose_cfg.PNP_NET.WITH_2D_COORD
    assert pose_cfg.PNP_NET.COORD_2D_TYPE == "abs"
    assert pose_cfg.PNP_NET.REGION_ATTENTION
    assert pose_cfg.PNP_NET.MASK_ATTENTION == "mul"
    assert pose_cfg.PNP_NET.INIT_CFG.coarse_grid_size == 4
    module, parameter_groups = get_pnp_net(cfg)
    assert isinstance(module, HierarchicalCorrespondencePnPNet)
    assert len(parameter_groups) == 1
    assert all(parameter.requires_grad for parameter in module.parameters())


def test_official_pose_head_keys_are_not_accidentally_accepted_as_new_head_keys():
    head = HierarchicalCorrespondencePnPNet()
    state = {"features.0.weight": torch.zeros(1)}
    missing, unexpected, errors = [], [], []
    head._load_from_state_dict(
        state,
        "",
        {},
        False,
        missing,
        unexpected,
        errors,
    )
    assert "features.0.weight" not in state
    assert not errors


def test_cpm_output_keys_are_filtered_but_exp012_output_keys_are_restored():
    head = HierarchicalCorrespondencePnPNet()
    legacy = {
        "rotation_head.weight": torch.zeros(1),
        "translation_head.weight": torch.zeros(1),
    }
    head._load_from_state_dict(legacy, "", {}, False, [], [], [])
    assert not legacy

    state = head.state_dict()
    assert "pose_rotation.weight" in state
    assert "pose_translation.weight" in state
    reloaded = HierarchicalCorrespondencePnPNet()
    reloaded.load_state_dict(state, strict=True)
    assert torch.equal(head.pose_rotation.weight, reloaded.pose_rotation.weight)
    assert torch.equal(head.pose_translation.weight, reloaded.pose_translation.weight)
