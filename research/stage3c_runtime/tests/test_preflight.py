from pathlib import Path

import pytest
import torch
from mmcv import Config

from research.stage3c_runtime.preflight import validate_config
from research.stage3c_runtime.verify_checkpoint_isolation import compare_states


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs/gdrn/lmo_pbr"


@pytest.mark.parametrize(
    "role,name",
    [
        ("B", "convnext_stage3c0_pnp_only_lmo.py"),
        ("C2", "convnext_stage3c2_pnp_quality_coverage_lmo.py"),
        ("B", "convnext_stage3c0_pnp_only_smoke_lmo.py"),
        ("C2", "convnext_stage3c2_pnp_quality_coverage_smoke_lmo.py"),
    ],
)
def test_b_and_c2_configs_pass_static_preflight(role, name):
    result = validate_config(role, Config.fromfile(str(CONFIG_ROOT / name)))
    assert result["role"] == role
    assert result["pnp_lr"] == pytest.approx(8e-5)


def test_c2_uses_distinct_learning_rates():
    result = validate_config(
        "C2",
        Config.fromfile(
            str(CONFIG_ROOT / "convnext_stage3c2_pnp_quality_coverage_lmo.py")
        ),
    )
    assert result["quality_lr"] == pytest.approx(8e-4)


def test_checkpoint_isolation_classifies_b_and_c2_changes():
    official = {
        "backbone.weight": torch.tensor([1.0]),
        "pnp_net.weight": torch.tensor([2.0]),
    }
    b = {
        "backbone.weight": torch.tensor([1.0]),
        "pnp_net.weight": torch.tensor([3.0]),
    }
    c2 = {
        **b,
        "quality_coverage_net.weight": torch.tensor([4.0]),
    }
    assert compare_states("B", official, b)["changed_frozen"] == []
    result = compare_states("C2", official, c2)
    assert result["changed_allowed"] == ["pnp_net.weight"]
    assert result["added"] == ["quality_coverage_net.weight"]


def test_checkpoint_isolation_detects_frozen_change():
    official = {"backbone.weight": torch.tensor([1.0])}
    trained = {"backbone.weight": torch.tensor([0.0])}
    assert compare_states("B", official, trained)["changed_frozen"] == [
        "backbone.weight"
    ]
