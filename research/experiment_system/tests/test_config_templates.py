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


def test_managed_exp005_exp009_and_exp010_configs_use_fixed_seed_42():
    for experiment_dir in (
        "exp005_pnp_control",
        "exp009_cpm_head",
        "exp010_cpm_official_lr_control",
    ):
        formal = Config.fromfile(str(CONFIG_ROOT / experiment_dir / "train.py"))
        smoke = Config.fromfile(str(CONFIG_ROOT / experiment_dir / "smoke.py"))
        audit = Config.fromfile(str(CONFIG_ROOT / experiment_dir / "audit48.py"))
        evaluation = Config.fromfile(str(CONFIG_ROOT / experiment_dir / "eval.py"))
        assert formal.SEED == smoke.SEED == audit.SEED == 42
        assert formal.SOLVER.TOTAL_EPOCHS == 40
        assert smoke.SOLVER.IMS_PER_BATCH == 4
        assert audit.SOLVER.IMS_PER_BATCH == 48
        assert smoke.SOLVER.TOTAL_EPOCHS == audit.SOLVER.TOTAL_EPOCHS == 1
        assert formal.DATALOADER.NUM_WORKERS == 16
        assert audit.DATALOADER.NUM_WORKERS == 16
        assert smoke.DATALOADER.NUM_WORKERS == 2
        assert evaluation.DATALOADER.NUM_WORKERS == 16


def test_exp010_formal_diff_from_exp009_is_identity_output_and_lr_only():
    exp009 = Config.fromfile(str(CONFIG_ROOT / "exp009_cpm_head/train.py"))
    exp010 = Config.fromfile(
        str(CONFIG_ROOT / "exp010_cpm_official_lr_control/train.py")
    )

    assert exp009.EXPERIMENT_ID == "EXP-20260809-009-cpm-head"
    assert exp010.EXPERIMENT_ID == "EXP-20260816-010-cpm-official-lr-control"
    assert exp009.OUTPUT_DIR != exp010.OUTPUT_DIR
    assert exp009.SOLVER.OPTIMIZER_CFG.lr == 8e-5
    assert exp010.SOLVER.OPTIMIZER_CFG.lr == 8e-4

    exp009_normalized = exp009._cfg_dict.to_dict()
    exp010_normalized = exp010._cfg_dict.to_dict()
    for cfg in (exp009_normalized, exp010_normalized):
        cfg.pop("EXPERIMENT_ID")
        cfg.pop("OUTPUT_DIR")
    exp010_normalized["SOLVER"]["OPTIMIZER_CFG"]["lr"] = exp009_normalized[
        "SOLVER"
    ]["OPTIMIZER_CFG"]["lr"]
    assert exp010_normalized == exp009_normalized
