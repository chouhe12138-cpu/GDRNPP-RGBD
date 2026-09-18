"""Dataset order, dynamic object count, and configurable training contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.datasets.lm_pbr import LM_PBR_Dataset, SPLITS_LM_PBR
from core.gdrn_modeling.datasets.lmo_bop_test import LMO_BOP_TEST_Dataset, SPLITS_LMO
from core.gdrn_modeling.models.heads.progressive_pcc_head import (
    ProgressivePCCHead,
)
from research.exp022.dataset_context import resolve_dataset_context
from research.run_contract import validate_research_run_config


SOURCE = ".local/dataset_cache/exp022/reused_v1.npz"
LM_CONFIG = "configs/gdrn/research/exp022_progressive_pcc/train_lm13_pbr.py"


def _synthetic_tree(tmp_path, count):
    with np.load(SOURCE, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]).copy() for name in source.files}
    for name, value in tuple(data.items()):
        if value.ndim and value.shape[0] == 8 and name not in {"source_leaf_indices"}:
            data[name] = np.concatenate([value] * ((count + 7) // 8), axis=0)[:count]
    data["source_leaf_indices"] = np.full((count, 4096), -1, dtype=np.int64)
    data["object_ids"] = np.arange(1, count + 1, dtype=np.int64)
    data["dataset_key"] = np.asarray("synthetic")
    data["mode"] = np.asarray("independent")
    data["generator_version"] = np.asarray(2, dtype=np.int64)
    path = tmp_path / f"n{count}.npz"
    np.savez_compressed(path, **data)
    return path


@pytest.mark.parametrize("count", [8, 13, 30])
def test_dynamic_object_count_forward_backward_inference(tmp_path, count):
    path = _synthetic_tree(tmp_path, count)
    head = ProgressivePCCHead(str(path), token_dim=32, dataset_key="synthetic",
                              expected_object_ids=tuple(range(1, count + 1)))
    backbone = torch.randn(1, 1024, 8, 8)
    classes = torch.tensor([count - 1])
    xyz = torch.full((1, 3, 64, 64), 0.5)
    mask = torch.ones(1, 1, 64, 64)
    losses, _ = head(backbone, classes, xyz, mask)
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert head.input_adapter.weight.grad is not None
    with torch.no_grad():
        output = head(backbone, classes)
    assert output["xyz_norm"].shape == (1, 3, 64, 64)
    with pytest.raises(ValueError, match="roi_class outside"):
        head(backbone, torch.tensor([count]))


def test_metadata_hierarchy_order_mismatch_fails(tmp_path):
    cfg = Config.fromfile(LM_CONFIG)
    with np.load(SOURCE, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]).copy() for name in source.files}
    data["object_ids"] = data["object_ids"][::-1].copy()
    path = tmp_path / "wrong_order.npz"
    np.savez(path, **data)
    cfg.MODEL.POSE_NET.PCC_HEAD.HIERARCHY_PATH = str(path)
    with pytest.raises(ValueError, match="hierarchy object order mismatch"):
        resolve_dataset_context(cfg)


def test_lm13_splits_use_same_noncontiguous_ids_without_xyz_crop():
    train = LM_PBR_Dataset(SPLITS_LM_PBR["lm_pbr_13_online_train"])
    test = LMO_BOP_TEST_Dataset(SPLITS_LMO["lm_bop_test_13"])
    expected = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    assert train.cat_ids == test.cat_ids == expected
    assert not train.require_xyz


def test_lm13_context_rejects_wrong_test_object_order():
    cfg = Config.fromfile(LM_CONFIG)
    context = resolve_dataset_context(cfg)
    assert context.object_ids == (1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15)
    cfg.DATASETS.TEST = ("lmo_bop_test",)
    with pytest.raises(ValueError, match="train/test object order mismatch"):
        resolve_dataset_context(cfg)


def test_new_schedule_is_configurable_but_formal_requires_readiness():
    cfg = Config.fromfile(LM_CONFIG)
    cfg.SOLVER.TOTAL_EPOCHS = 12
    cfg.SOLVER.CHECKPOINT_PERIOD = 3
    cfg.TEST.EVAL_PERIOD = 3
    cfg.SOLVER.OPTIMIZER_CFG.lr = 2e-4
    validate_research_run_config(cfg, mode="prepare")
    with pytest.raises(ValueError, match="not ready"):
        validate_research_run_config(cfg, mode="formal")
    cfg.RESEARCH_PROTOCOL.FORMAL_READY = True
    report = validate_research_run_config(cfg, mode="formal")
    assert (report["total_epochs"], report["evaluation_period"]) == (12, 3)


def test_tless_reservation_parses_without_dataset_files():
    cfg = Config.fromfile("configs/gdrn/research/exp022_progressive_pcc/tless_reserved.py")
    assert cfg.MODEL.POSE_NET.NUM_CLASSES == 30
    assert cfg.DATASET_CONTEXT.CAD_REF_KEY == "tless"
    assert not cfg.RESEARCH_PROTOCOL.FORMAL_READY
