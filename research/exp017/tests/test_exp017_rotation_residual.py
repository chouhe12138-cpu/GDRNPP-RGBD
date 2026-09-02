from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.models.heads.exp013_geometry_pnp_net import (
    XYZResidualBypassPnPNet,
)
from core.gdrn_modeling.models.heads.exp017_rotation_residual_pnp_net import (
    SupportAwareRotationResidualPnPNet,
)
from core.gdrn_modeling.models.model_utils import get_pnp_net
from core.gdrn_modeling.models.net_factory import HEADS


ROOT = Path(__file__).resolve().parents[3]
A_CONFIG = ROOT / "configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual/train.py"
EXP017_CONFIG = ROOT / (
    "configs/gdrn/lmo_pbr/research/exp017/"
    "support_aware_rotation_residual/train.py"
)


def _small_kwargs() -> dict:
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
        adapter_token_channels=64,
        adapter_score_channels=32,
        alpha_r_init=1.0,
    )


def _a_small_kwargs() -> dict:
    kwargs = _small_kwargs()
    for key in (
        "adapter_token_channels",
        "adapter_score_channels",
        "alpha_r_init",
    ):
        kwargs.pop(key)
    return kwargs


def _inputs(batch: int = 2, partial: bool = True):
    torch.manual_seed(101)
    coor = torch.rand(batch, 5, 64, 64)
    region = torch.softmax(torch.randn(batch, 64, 64, 64), dim=1)
    extents = torch.rand(batch, 3) + 0.05
    mask = torch.ones(batch, 1, 64, 64)
    if partial:
        mask[:, :, :16, :24] = 0
        mask[:, :, 48:, 48:] = 0
    return coor, region, extents, mask


def _load_a_into_exp017(a, exp017):
    incompatible = exp017.load_state_dict(a.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        f"rotation_adapter.{name}" for name in exp017.rotation_adapter.state_dict()
    } | {"alpha_r"}


def test_config_registration_protocol_and_parameter_budget():
    assert HEADS["SupportAwareRotationResidualPnPNet"] is SupportAwareRotationResidualPnPNet
    cfg = Config.fromfile(str(EXP017_CONFIG))
    a_cfg = Config.fromfile(str(A_CONFIG))
    pose, a_pose = cfg.MODEL.POSE_NET, a_cfg.MODEL.POSE_NET
    assert cfg.EXPERIMENT_ID == "EXP-20260902-017-support-aware-rotation-residual"
    assert cfg.SEED == 42
    assert cfg.SOLVER.TOTAL_EPOCHS == 40
    assert cfg.SOLVER.IMS_PER_BATCH == 48
    assert cfg.SOLVER.OPTIMIZER_CFG == a_cfg.SOLVER.OPTIMIZER_CFG
    assert cfg.SOLVER.WARMUP_ITERS == 200
    assert cfg.SOLVER.CHECKPOINT_PERIOD == 5 and cfg.TEST.EVAL_PERIOD == 5
    assert pose.BACKBONE == a_pose.BACKBONE
    assert pose.GEO_HEAD == a_pose.GEO_HEAD
    assert not pose.PNP_NET.FREEZE
    assert pose.PNP_NET.INIT_CFG.type == "SupportAwareRotationResidualPnPNet"
    assert pose.PNP_NET.INIT_CFG.alpha_r_init == 1.0
    assert pose.PNP_NET.MASK_ATTENTION == "mul"
    assert pose.PNP_NET.COORD_2D_TYPE == "abs"
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    a_cfg.SOLVER.BASE_LR = float(a_cfg.SOLVER.OPTIMIZER_CFG.lr)
    exp017, _ = get_pnp_net(cfg)
    a, _ = get_pnp_net(a_cfg)
    assert exp017.adapter_parameter_count() == 13_000
    assert sum(p.numel() for p in exp017.parameters()) - sum(
        p.numel() for p in a.parameters()
    ) == 13_000
    assert exp017.adapter_parameter_count() <= 15_000


def test_a_state_load_is_initially_value_exact_and_translation_bitwise_equal():
    torch.manual_seed(7)
    a = XYZResidualBypassPnPNet(**_a_small_kwargs()).eval()
    torch.manual_seed(8)
    exp017 = SupportAwareRotationResidualPnPNet(**_small_kwargs()).eval()
    _load_a_into_exp017(a, exp017)
    inputs = _inputs(batch=2)
    with torch.no_grad():
        expected_r, expected_t = a(*inputs)
        actual_r, actual_t, info = exp017.forward_with_adapter_intervention(*inputs)
    assert torch.equal(exp017.alpha_r, torch.ones_like(exp017.alpha_r))
    assert torch.count_nonzero(info["delta_r"]) == 0
    assert torch.equal(actual_r, expected_r)
    assert torch.equal(actual_t, expected_t)
    assert torch.equal(actual_t, info["raw_t_a"])


def test_translation_only_loss_cannot_reach_adapter():
    head = SupportAwareRotationResidualPnPNet(**_small_kwargs())
    _r, t = head(*_inputs(batch=2))
    t.square().mean().backward()
    for parameter in head.adapter_parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_rotation_loss_reaches_zero_initialized_delta_output_directly():
    head = SupportAwareRotationResidualPnPNet(**_small_kwargs())
    r, _t = head(*_inputs(batch=2))
    r.sum().backward()
    gradient = head.rotation_adapter.delta_output.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert head.rotation_adapter.token_projection.weight.grad is not None
    assert torch.count_nonzero(head.rotation_adapter.token_projection.weight.grad) == 0


def test_after_output_step_internal_projection_pooling_and_position_get_gradients():
    head = SupportAwareRotationResidualPnPNet(**_small_kwargs())
    inputs = _inputs(batch=2)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    r, _t = head(*inputs)
    r.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    r, _t = head(*inputs)
    r.square().mean().backward()
    parameters = (
        head.rotation_adapter.token_projection.weight,
        head.rotation_adapter.pool_score[0].weight,
        head.rotation_adapter.pool_score[2].weight,
        head.rotation_adapter.position_embedding,
        head.rotation_adapter.delta_hidden.weight,
        head.alpha_r,
    )
    for parameter in parameters:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_adapter_is_region_free_and_interventions_never_change_translation():
    head = SupportAwareRotationResidualPnPNet(**_small_kwargs()).eval()
    coor, region, extents, mask = _inputs(batch=2)
    with torch.no_grad():
        metric_a, _region_a, support_a = head._validate_and_mask_inputs(
            coor, region, extents, mask
        )
        grid_a = head._encode_geometry_grid(metric_a, support_a)
        adapter_a = head.rotation_adapter(grid_a, support_a)
        for changed_region in (torch.zeros_like(region), torch.roll(region, 1, 0)):
            metric_b, _region_b, support_b = head._validate_and_mask_inputs(
                coor, changed_region, extents, mask
            )
            grid_b = head._encode_geometry_grid(metric_b, support_b)
            adapter_b = head.rotation_adapter(grid_b, support_b)
            torch.testing.assert_close(adapter_a, adapter_b, rtol=0, atol=0)

        _r0, t0, _ = head.forward_with_adapter_intervention(coor, region, extents, mask)
        variants = (
            dict(pooling="uniform"),
            dict(position_mode="shuffle"),
            dict(token_shuffle=True),
        )
        for kwargs in variants:
            _r, t, _ = head.forward_with_adapter_intervention(
                coor, region, extents, mask, **kwargs
            )
            assert torch.equal(t, t0)
        head.rotation_adapter.token_projection.weight.add_(0.1)
        head.rotation_adapter.position_embedding.add_(0.25)
        head.rotation_adapter.pool_score[0].weight.add_(0.1)
        head.rotation_adapter.pool_score[2].weight.add_(0.1)
        head.alpha_r.fill_(3.0)
        _r, changed_t, _ = head.forward_with_adapter_intervention(
            coor, region, extents, mask
        )
        assert torch.equal(changed_t, t0)


def test_support_outside_correspondence_pollution_cannot_change_descriptor():
    head = SupportAwareRotationResidualPnPNet(**_small_kwargs()).eval()
    coor, region, extents, mask = _inputs(batch=1)
    changed = coor.clone()
    outside = mask.expand_as(changed) == 0
    changed[outside] = 1.0e4
    with torch.no_grad():
        _r1, _t1, first = head.forward_with_adapter_intervention(
            coor, region, extents, mask
        )
        _r2, _t2, second = head.forward_with_adapter_intervention(
            changed, region, extents, mask
        )
    torch.testing.assert_close(first["descriptor"], second["descriptor"], rtol=0, atol=0)


def test_invalid_weights_are_zero_and_valid_weights_renormalize():
    head = SupportAwareRotationResidualPnPNet(**_small_kwargs()).eval()
    with torch.no_grad():
        _r, _t, info = head.forward_with_adapter_intervention(*_inputs(batch=2))
    weights, valid = info["weights"], info["valid"]
    assert torch.count_nonzero(weights[~valid]) == 0
    torch.testing.assert_close(
        weights.sum(dim=1), torch.ones(weights.shape[0]), rtol=0, atol=1e-7
    )


def test_partial_support_runs_and_all_zero_support_is_rejected():
    head = SupportAwareRotationResidualPnPNet(**_small_kwargs())
    outputs = head(*_inputs(batch=1, partial=True))
    assert all(torch.isfinite(value).all() for value in outputs)
    coor, region, extents, mask = _inputs(batch=1, partial=False)
    with pytest.raises(ValueError, match="non-zero visible support"):
        head(coor, region, extents, torch.zeros_like(mask))


@pytest.mark.parametrize("batch", [1, 3])
def test_cpu_forward_backward_batch_shapes(batch):
    head = SupportAwareRotationResidualPnPNet(**_small_kwargs())
    r, t = head(*_inputs(batch=batch))
    assert r.shape == (batch, 6) and t.shape == (batch, 3)
    assert torch.isfinite(r).all() and torch.isfinite(t).all()
    (r.square().mean() + t.square().mean()).backward()
    assert head.rotation_adapter.delta_output.weight.grad is not None


def test_checkpoint_roundtrip_is_value_exact():
    torch.manual_seed(19)
    head = SupportAwareRotationResidualPnPNet(**_small_kwargs()).eval()
    with torch.no_grad():
        head.rotation_adapter.delta_output.weight.normal_(std=0.01)
        head.rotation_adapter.delta_output.bias.normal_(std=0.01)
    inputs = _inputs(batch=1)
    with torch.no_grad():
        expected = head(*inputs)
    payload = io.BytesIO()
    torch.save(head.state_dict(), payload)
    payload.seek(0)
    restored = SupportAwareRotationResidualPnPNet(**_small_kwargs()).eval()
    restored.load_state_dict(torch.load(payload, map_location="cpu"), strict=True)
    with torch.no_grad():
        actual = restored(*inputs)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_read_only_intervention_interfaces_are_finite():
    head = SupportAwareRotationResidualPnPNet(**_small_kwargs()).eval()
    with torch.no_grad():
        head.rotation_adapter.delta_output.weight.normal_(std=0.01)
        variants = (
            dict(pooling="learned"),
            dict(pooling="uniform"),
            dict(position_mode="shuffle"),
            dict(token_shuffle=True),
        )
        for kwargs in variants:
            r, t, info = head.forward_with_adapter_intervention(
                *_inputs(batch=2), **kwargs
            )
            assert torch.isfinite(r).all() and torch.isfinite(t).all()
            assert torch.isfinite(info["descriptor"]).all()
            assert info["weights"].shape == (2, 64)


def test_smoke_and_audit_configs_are_isolated_one_epoch_runs():
    for name, workers, batch in (("smoke.py", 2, 4), ("audit48.py", 16, 48)):
        cfg = Config.fromfile(str(EXP017_CONFIG.with_name(name)))
        assert tuple(cfg.DATASETS.TRAIN) == ("lmo_pbr_stage3_local_train",)
        assert tuple(cfg.DATASETS.TEST) == ()
        assert cfg.SOLVER.TOTAL_EPOCHS == 1
        assert cfg.SOLVER.IMS_PER_BATCH == batch
        assert cfg.DATALOADER.NUM_WORKERS == workers
        assert cfg.TEST.EVAL_PERIOD == 0
