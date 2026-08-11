from pathlib import Path

import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.datasets.lm_pbr import (
    PROJ_ROOT,
    SPLITS_LM_PBR,
    resolve_dataset_cache_root,
)
from core.gdrn_modeling.models.GDRN_double_mask import get_backbone_init_args
from core.utils.solver_utils import (
    accumulation_window_size,
    get_accumulation_steps,
    optimizer_updates_per_training,
    should_optimizer_step,
)
from research.pnp_control.verify_checkpoint_isolation import compare_states, load_state


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAL_CONFIG = (
    PROJECT_ROOT
    / "configs/gdrn/lmo_pbr/convnext_stage3c0_pnp_only_lmo.py"
)
LOCAL_CONFIG = (
    PROJECT_ROOT
    / "configs/gdrn/lmo_pbr/convnext_stage3c0_pnp_only_local_lmo.py"
)


def test_stage3_scenes_are_disjoint_and_complete():
    train = set(SPLITS_LM_PBR["lmo_pbr_stage3_train"]["scene_ids"])
    validation = set(SPLITS_LM_PBR["lmo_pbr_stage3_val"]["scene_ids"])
    assert len(train) == 47
    assert validation == {12, 13, 14}
    assert not train & validation
    assert train | validation == set(range(50))


def test_local_split_is_balanced_and_visibility_filtered():
    local = SPLITS_LM_PBR["lmo_pbr_stage3_local_train"]
    assert local["scene_ids"] == (0, 1, 2)
    assert local["max_instances_per_object"] == 1024
    assert local["min_visible_fraction"] == pytest.approx(0.3)
    assert local["require_xyz"] is False


def test_dataset_cache_supports_an_absolute_machine_local_override():
    assert resolve_dataset_cache_root({}) == str(Path(PROJ_ROOT) / ".cache")
    override = "/home/gdrn/.cache/gdrnpp_datasets"
    assert resolve_dataset_cache_root({"GDRN_DATASET_CACHE_DIR": override}) == override
    with pytest.raises(ValueError, match="absolute path"):
        resolve_dataset_cache_root({"GDRN_DATASET_CACHE_DIR": "relative/cache"})
    assert all(
        split["cache_dir"] == str(Path(PROJ_ROOT) / ".cache")
        for split in SPLITS_LM_PBR.values()
    )


@pytest.mark.parametrize("config_path", [FORMAL_CONFIG, LOCAL_CONFIG])
def test_control_configs_train_only_patch_pnp(config_path):
    cfg = Config.fromfile(str(config_path))
    assert cfg.MODEL.WEIGHTS
    assert cfg.MODEL.POSE_NET.BACKBONE.FREEZE is True
    assert cfg.MODEL.POSE_NET.GEO_HEAD.FREEZE is True
    assert cfg.MODEL.POSE_NET.PNP_NET.FREEZE is False
    assert cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.pretrained is False
    assert cfg.SEED == 20260731
    assert cfg.SOLVER.REFERENCE_BS == 48
    assert cfg.SOLVER.OPTIMIZER_CFG.type == "Ranger"
    assert cfg.SOLVER.OPTIMIZER_CFG.lr == pytest.approx(8e-5)
    assert cfg.SOLVER.OPTIMIZER_CFG.weight_decay == pytest.approx(0.01)


def test_formal_and_local_schedules_are_frozen():
    formal = Config.fromfile(str(FORMAL_CONFIG))
    local = Config.fromfile(str(LOCAL_CONFIG))
    assert tuple(formal.DATASETS.TRAIN) == ("lmo_pbr_train",)
    assert tuple(formal.DATASETS.TEST) == ("lmo_bop_test",)
    assert formal.TEST.EVAL_PERIOD == 5
    assert formal.TEST.TEST_BBOX_TYPE == "gt"
    assert formal.SOLVER.IMS_PER_BATCH == 48
    assert formal.SOLVER.TOTAL_EPOCHS == 40
    assert formal.SOLVER.CHECKPOINT_PERIOD == 5
    assert formal.SOLVER.MAX_TO_KEEP == 3
    assert formal.RUN_ARTIFACTS.STRUCTURED_LAYOUT
    assert formal.RUN_ARTIFACTS.COMPACT_LOG
    assert formal.RUN_ARTIFACTS.TENSORBOARD is False
    assert formal.TRAIN.PRINT_FREQ == 500
    assert tuple(local.DATASETS.TRAIN) == ("lmo_pbr_stage3_local_train",)
    assert tuple(local.DATASETS.TEST) == ()
    assert local.TEST.EVAL_PERIOD == 0
    assert local.SOLVER.IMS_PER_BATCH == 4
    assert local.SOLVER.TOTAL_EPOCHS == 1


def test_full_checkpoint_disables_backbone_network_download_without_mutation():
    cfg = Config.fromfile(str(FORMAL_CONFIG))
    cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.pretrained = True
    backbone_type, init_args = get_backbone_init_args(cfg)
    assert backbone_type.startswith("timm/")
    assert init_args["pretrained"] is False
    assert cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.pretrained is True

    cfg.MODEL.WEIGHTS = ""
    _, init_args = get_backbone_init_args(cfg)
    assert init_args["pretrained"] is True


def test_gradient_accumulation_boundaries_and_update_count():
    accumulation = get_accumulation_steps(48, 4)
    assert accumulation == 12
    assert optimizer_updates_per_training(2048, 1, accumulation) == 171
    assert not should_optimizer_step(10, accumulation, 2048, 2048)
    assert should_optimizer_step(11, accumulation, 2048, 2048)
    assert should_optimizer_step(2047, accumulation, 2048, 2048)
    assert accumulation_window_size(0, accumulation, 2048) == 12
    assert accumulation_window_size(2040, accumulation, 2048) == 8
    assert accumulation_window_size(2047, accumulation, 2048) == 8


def test_gradient_accumulation_restarts_at_each_epoch():
    assert should_optimizer_step(7, 4, 10, 20)
    assert should_optimizer_step(9, 4, 10, 20)
    assert not should_optimizer_step(11, 4, 10, 20)
    assert should_optimizer_step(13, 4, 10, 20)
    assert should_optimizer_step(19, 4, 10, 20)
    assert accumulation_window_size(10, 4, 10) == 4
    assert accumulation_window_size(18, 4, 10) == 2


@pytest.mark.parametrize("reference,batch", [(47, 4), (2, 4), (48, 0)])
def test_invalid_gradient_accumulation_ratio_is_rejected(reference, batch):
    with pytest.raises(ValueError):
        get_accumulation_steps(reference, batch)


def test_checkpoint_isolation_accepts_only_pnp_changes(tmp_path):
    official_path = tmp_path / "official.pth"
    trained_path = tmp_path / "trained.pth"
    torch.save(
        {
            "model": {
                "backbone.weight": torch.tensor([1.0]),
                "pnp_net.weight": torch.tensor([2.0]),
            }
        },
        official_path,
    )
    torch.save(
        {
            "model": {
                "_module.backbone.weight": torch.tensor([1.0]),
                "_module.pnp_net.weight": torch.tensor([3.0]),
            }
        },
        trained_path,
    )
    result = compare_states(load_state(official_path), load_state(trained_path))
    assert result["changed_pnp"] == ["pnp_net.weight"]
    assert result["changed_frozen"] == []


def test_checkpoint_isolation_detects_frozen_change():
    official = {
        "backbone.weight": torch.tensor([1.0]),
        "pnp_net.weight": torch.tensor([2.0]),
    }
    trained = {
        "backbone.weight": torch.tensor([0.0]),
        "pnp_net.weight": torch.tensor([3.0]),
    }
    result = compare_states(official, trained)
    assert result["changed_frozen"] == ["backbone.weight"]
