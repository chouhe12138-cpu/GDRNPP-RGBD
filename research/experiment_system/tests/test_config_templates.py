from pathlib import Path

from mmcv import Config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs/gdrn/lmo_pbr/research"


def test_future_formal_template_resolves_without_changing_legacy_configs():
    cfg = Config.fromfile(str(CONFIG_ROOT / "templates/pose_head/train.py"))
    assert tuple(cfg.DATASETS.TRAIN) == ("lmo_pbr_train",)
    assert tuple(cfg.DATASETS.TEST) == ("lmo_bop_test",)
    assert cfg.TEST.TEST_BBOX_TYPE == "gt"
    assert cfg.RUN_ARTIFACTS.STRUCTURED_LAYOUT
    assert cfg.SOLVER.TOTAL_EPOCHS == 40
    assert cfg.EXPERIMENT_ID == "EXP-REPLACE-BEFORE-USE"


def test_future_smoke_template_is_small_and_has_no_evaluation():
    cfg = Config.fromfile(str(CONFIG_ROOT / "templates/pose_head/smoke.py"))
    assert tuple(cfg.DATASETS.TRAIN) == ("lmo_pbr_stage3_local_train",)
    assert tuple(cfg.DATASETS.TEST) == ()
    assert cfg.SOLVER.TOTAL_EPOCHS == 1
    assert cfg.TEST.EVAL_PERIOD == 0
