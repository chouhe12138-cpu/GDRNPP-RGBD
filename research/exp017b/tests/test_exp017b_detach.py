from __future__ import annotations

import torch
from mmcv import Config

from core.gdrn_modeling.models.heads.exp017_rotation_residual_pnp_net import (
    DetachedSupportAwareRotationResidualPnPNet,
    SupportAwareRotationResidualPnPNet,
)
from core.gdrn_modeling.models.net_factory import HEADS
from research.exp017.tests.test_exp017_rotation_residual import _inputs, _small_kwargs
from research.exp017b.preflight import (
    CONFIG,
    EXPERIMENT_ID,
    PARENT_CONFIG,
    PROJECT_ROOT,
    run_preflight,
)
from research.run_contract import validate_research_run_config


def test_config_registration_and_renderer_contract():
    assert (
        HEADS["DetachedSupportAwareRotationResidualPnPNet"]
        is DetachedSupportAwareRotationResidualPnPNet
    )
    cfg = Config.fromfile(str(PROJECT_ROOT / CONFIG))
    parent = Config.fromfile(str(PROJECT_ROOT / PARENT_CONFIG))
    contract = validate_research_run_config(
        cfg, mode="formal", expected_experiment_id=EXPERIMENT_ID
    )
    assert cfg.MODEL.POSE_NET.PNP_NET.INIT_CFG.type == (
        "DetachedSupportAwareRotationResidualPnPNet"
    )
    assert contract["training_renderer"] is None
    assert contract["training_geometry_supervision"] is False
    assert contract["evaluation_renderer"] == "cpp"
    for section in ("DATASETS", "DATALOADER", "SOLVER", "TEST", "TRAIN"):
        assert cfg[section] == parent[section]
    assert cfg.MODEL.WEIGHTS == parent.MODEL.WEIGHTS
    candidate_init = dict(cfg.MODEL.POSE_NET.PNP_NET.INIT_CFG)
    parent_init = dict(parent.MODEL.POSE_NET.PNP_NET.INIT_CFG)
    candidate_init.pop("type")
    parent_init.pop("type")
    assert candidate_init == parent_init


def test_detach_is_forward_exact_but_blocks_only_adapter_upstream_path():
    torch.manual_seed(17)
    normal = SupportAwareRotationResidualPnPNet(**_small_kwargs())
    detached = DetachedSupportAwareRotationResidualPnPNet(**_small_kwargs())
    with torch.no_grad():
        normal.rotation_adapter.delta_output.weight.normal_(std=0.01)
        normal.rotation_adapter.delta_output.bias.normal_(std=0.01)
    detached.load_state_dict(normal.state_dict(), strict=True)
    inputs = _inputs(batch=2)

    normal_r, normal_t, normal_info = normal.forward_with_adapter_intervention(*inputs)
    detached_r, detached_t, detached_info = detached.forward_with_adapter_intervention(
        *inputs
    )
    assert torch.equal(normal_r, detached_r)
    assert torch.equal(normal_t, detached_t)
    assert normal_info["adapter_geometry_detached"] is False
    assert detached_info["adapter_geometry_detached"] is True

    normal_grid_grad = torch.autograd.grad(
        normal_info["delta_r"].square().mean(), normal_info["geometry_grid"]
    )[0]
    assert torch.isfinite(normal_grid_grad).all()
    assert torch.count_nonzero(normal_grid_grad) > 0
    detached_grid_grad = torch.autograd.grad(
        detached_info["delta_r"].square().mean(),
        detached_info["geometry_grid"],
        allow_unused=True,
    )[0]
    assert detached_grid_grad is None

    adapter_grads = torch.autograd.grad(
        detached_info["delta_r"].square().mean(),
        list(detached.adapter_parameters()),
        allow_unused=True,
    )
    assert any(g is not None and torch.count_nonzero(g) for g in adapter_grads)


def test_exp017b_preflight_passes():
    result = run_preflight()
    assert result["status"] == "PASS"
    assert result["adapter_to_shared_geometry_gradient_blocked"] is True
