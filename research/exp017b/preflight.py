#!/usr/bin/env python3
"""CPU preflight for the EXP017-B graph-isolation candidate."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.heads.exp017_rotation_residual_pnp_net import (
    DetachedSupportAwareRotationResidualPnPNet,
    SupportAwareRotationResidualPnPNet,
)
from core.gdrn_modeling.models.model_utils import get_pnp_net
from research.exp017.preflight import PROJECT_ROOT, _head_inputs
from research.run_contract import validate_research_run_config


CONFIG = "configs/gdrn/lmo_pbr/research/exp017/" "b_detached_adapter_geometry/train.py"
PARENT_CONFIG = (
    "configs/gdrn/lmo_pbr/research/exp017/" "support_aware_rotation_residual/train.py"
)
EXPERIMENT_ID = "EXP-20260903-017-b-detached-adapter-geometry"


def _shared_geometry_parameters(head):
    prefixes = (
        "geometry_input_projection.",
        "geometry_local_fine.",
        "geometry_downsample_mid.",
        "geometry_local_mid.",
        "geometry_downsample_high.",
        "geometry_local_high.",
    )
    return [p for name, p in head.named_parameters() if name.startswith(prefixes)]


def _flat_grads(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).flatten()
            for g, p in zip(grads, parameters)
        ]
    )


def run_preflight() -> dict[str, object]:
    cfg = Config.fromfile(str(PROJECT_ROOT / CONFIG))
    parent_cfg = Config.fromfile(str(PROJECT_ROOT / PARENT_CONFIG))
    contract = validate_research_run_config(
        cfg, mode="formal", expected_experiment_id=EXPERIMENT_ID
    )
    if contract["training_renderer"] is not None:
        raise RuntimeError("EXP017-B training renderer must remain disabled")
    if contract["evaluation_renderer"] != "cpp":
        raise RuntimeError("EXP017-B BOP evaluation renderer must remain cpp")

    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    parent_cfg.SOLVER.BASE_LR = float(parent_cfg.SOLVER.OPTIMIZER_CFG.lr)
    parent, _ = get_pnp_net(parent_cfg)
    candidate, _ = get_pnp_net(cfg)
    if not isinstance(parent, SupportAwareRotationResidualPnPNet):
        raise RuntimeError("Wrong EXP017 parent head")
    if not isinstance(candidate, DetachedSupportAwareRotationResidualPnPNet):
        raise RuntimeError("Wrong EXP017-B head")
    candidate.load_state_dict(parent.state_dict(), strict=True)
    if candidate.adapter_parameter_count() != 13_000:
        raise RuntimeError("EXP017-B changed the adapter parameter budget")

    inputs = _head_inputs(torch.device("cpu"), batch=2)
    with torch.no_grad():
        parent_r, parent_t, _ = parent.forward_with_adapter_intervention(*inputs)
        candidate_r, candidate_t, _ = candidate.forward_with_adapter_intervention(*inputs)
    if not torch.equal(parent_r, candidate_r) or not torch.equal(parent_t, candidate_t):
        raise RuntimeError("Detach changed forward values")
    # Move past the identity initialization so the adapter has a real upstream
    # gradient, while keeping both heads state-identical.
    with torch.no_grad():
        parent.rotation_adapter.delta_output.weight.normal_(std=0.01)
        parent.rotation_adapter.delta_output.bias.normal_(std=0.01)
    candidate.load_state_dict(parent.state_dict(), strict=True)
    parent_r, _, _ = parent.forward_with_adapter_intervention(*inputs)
    candidate_r, _, _ = candidate.forward_with_adapter_intervention(*inputs)
    if not torch.equal(parent_r, candidate_r):
        raise RuntimeError("Normal and detached graph forwards differ")
    parent_shared = _shared_geometry_parameters(parent)
    candidate_shared = _shared_geometry_parameters(candidate)
    parent_grad = _flat_grads(parent_r.square().mean(), parent_shared)
    candidate_grad = _flat_grads(candidate_r.square().mean(), candidate_shared)
    injected = (parent_grad - candidate_grad).norm()
    if not torch.isfinite(injected) or float(injected) == 0.0:
        raise RuntimeError("Did not detect EXP017 adapter gradient injection")

    candidate_r, candidate_t = candidate(*inputs)
    adapter_grads = torch.autograd.grad(
        candidate_r.square().mean(),
        list(candidate.adapter_parameters()),
        allow_unused=True,
    )
    if not any(g is not None and torch.count_nonzero(g) for g in adapter_grads):
        raise RuntimeError("EXP017-B adapter no longer receives rotation gradients")
    translation_grads = torch.autograd.grad(
        candidate_t.square().mean(),
        list(candidate.adapter_parameters()),
        allow_unused=True,
    )
    if any(g is not None and torch.count_nonzero(g) for g in translation_grads):
        raise RuntimeError("EXP017-B adapter received translation gradients")

    return {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "sole_graph_change": "rotation_adapter(geometry_grid.detach(), support)",
        "forward_value_exact_to_exp017": True,
        "translation_bitwise_equal_to_exp017": True,
        "adapter_parameters": candidate.adapter_parameter_count(),
        "adapter_gradient_nonzero": True,
        "translation_to_adapter_gradient_zero": True,
        "adapter_to_shared_geometry_gradient_blocked": True,
        "normal_exp017_injected_shared_gradient_norm": float(injected),
        "training_renderer": contract["training_renderer"],
        "evaluation_renderer": contract["evaluation_renderer"],
    }


def main() -> int:
    print(json.dumps(run_preflight(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
