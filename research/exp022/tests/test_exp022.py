"""CPU contracts for the progressive PCC implementation."""

import numpy as np
import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.models.heads.progressive_pcc_head import (
    ProgressivePCCHead,
    load_pcc_hierarchy,
)
from core.utils.solver_utils import build_lr_scheduler
from research.exp022.build_hierarchy import SOURCE_DEFAULT, _reused_object
from research.run_contract import validate_research_run_config


HIERARCHY = ".local/dataset_cache/exp022/reused_v1.npz"


@pytest.fixture(scope="module")
def head():
    torch.manual_seed(42)
    return ProgressivePCCHead(HIERARCHY, token_dim=32)


def test_reused_tree_preserves_source_leaves():
    with np.load(SOURCE_DEFAULT) as data:
        for index in range(8):
            levels, indices = _reused_object(data, index)
            np.testing.assert_array_equal(np.sort(indices), np.arange(4096))
            np.testing.assert_array_equal(
                levels[-1][0][np.argsort(indices)], data["fine_anchors"][index].reshape(4096, 3)
            )


def test_hierarchy_schema():
    arrays = load_pcc_hierarchy(HIERARCHY)
    assert tuple(arrays["level4_anchors"].shape) == (8, 4096, 3)
    assert tuple(arrays["symmetry_counts"].shape) == (8,)


def test_sparse_resize_probability_and_path_identity():
    ids = torch.tensor([[[[3, 7], [3, 7]], [[3, 7], [3, 7]]]])
    scores = torch.tensor([[[[-0.2, -1.7], [-0.2, -1.7]], [[-0.2, -1.7], [-0.2, -1.7]]]])
    ids_new, log_new = ProgressivePCCHead._resize_sparse_paths(ids, scores)
    assert tuple(ids_new.shape) == (1, 4, 4, 2)
    assert torch.all(ids_new[..., 0] == 3)
    assert torch.allclose(log_new.exp().sum(-1), torch.ones(1, 4, 4), atol=1e-6)


def test_head_train_infer_and_gradients(head):
    feature = torch.randn(2, 1024, 8, 8)
    classes = torch.tensor([0, 1])
    xyz = torch.full((2, 3, 64, 64), 0.5)
    mask = torch.ones(2, 1, 64, 64)
    losses, stats = head(feature, classes, xyz, mask)
    assert set(losses) == {"loss_pcc_route", "loss_pcc_residual", "loss_pcc_mask"}
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert head.stages[0].query.weight.grad is not None
    assert head.token_encoder[0].weight.grad is not None
    assert torch.isfinite(head.stages[0].query.weight.grad).all()
    assert torch.isfinite(stats["selected_symmetry_branch_mean"])
    with torch.no_grad():
        output = head(feature[:1], classes[:1])
    assert tuple(output["xyz_norm"].shape) == (1, 3, 64, 64)
    assert tuple(output["mask_logit"].shape) == (1, 1, 64, 64)
    assert output["leaf_ids"].min() >= 0
    assert output["leaf_ids"].max() < 4096
    assert output["residual_norm"].max() <= 1.000001
    assert torch.allclose(output["beam_scores"].exp().sum(-1),
                          torch.ones(1, 64, 64), atol=1e-5)


def test_symmetric_subset_branch_is_finite(head):
    feature = torch.zeros(2, 1024, 8, 8)
    classes = torch.tensor([0, 5])
    xyz = torch.full((2, 3, 64, 64), 0.5)
    visible = torch.ones(2, 1, 64, 64)
    losses, stats = head(feature, classes, xyz, visible)
    assert all(torch.isfinite(value) for value in losses.values())
    assert 0 <= stats["selected_symmetry_branch_mean"] <= 0.5


@pytest.mark.parametrize("name", ["train_reused.py", "smoke_reused.py", "smoke_independent.py"])
def test_config_contract(name):
    path = f"configs/gdrn/lmo_pbr/research/exp022_progressive_pcc/{name}"
    cfg = Config.fromfile(path)
    validate_research_run_config(cfg, mode="formal" if name == "train_reused.py" else "smoke",
                                 expected_experiment_id="EXP-20260916-022-progressive-pcc")
    assert cfg.SOLVER.AMP.ENABLED


def test_warmup_transitions_directly_to_cosine():
    cfg = Config.fromfile("configs/gdrn/lmo_pbr/research/exp022_progressive_pcc/train_reused.py")
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=3e-4)
    _scheduler, factor = build_lr_scheduler(cfg, optimizer, total_iters=1000, return_function=True)
    assert factor(0) == pytest.approx(0.001)
    assert factor(40) == pytest.approx(1.0)
    assert factor(41) < 1.0
    assert factor(1000) == pytest.approx(0.01)
