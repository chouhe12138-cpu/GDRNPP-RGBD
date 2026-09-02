from __future__ import annotations

from pathlib import Path

from mmcv import Config


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_CONFIG_ROOT = ROOT / "configs/gdrn/lmo_pbr/research"

ACTIVE_EXPERIMENT_DIRS = (
    "exp012_hierarchical_corr_head",
    "exp013/a_xyz_residual",
    "exp013/b_geometry_attention",
    "exp013/c_rt_decoupled",
    "exp013/d_fulltrain",
    "exp013/e_official_head_random",
    "exp013/f_glm_pose_l",
    "exp017/support_aware_rotation_residual",
)
ACTIVE_EXPERIMENT_FILES = ("train.py", "smoke.py", "audit48.py", "eval.py")

LEGACY_REFERENCES = (
    "convnext_stage3c0_pnp_only",
    "convnext_stage3c1",
    "convnext_stage3c2",
    "experiment_system",
    "managed_runtime",
)


def _load(path: Path):
    assert path.is_file(), f"missing active config: {path}"
    text = path.read_text(encoding="utf-8")
    for legacy in LEGACY_REFERENCES:
        assert legacy not in text, f"{path} still references legacy path {legacy}"
    return Config.fromfile(str(path))


def test_long_term_pnp_control_contract():
    base = RESEARCH_CONFIG_ROOT / "controls/pnp_only"
    train = _load(base / "train.py")
    smoke = _load(base / "smoke.py")
    eval_cfg = _load(base / "eval.py")

    assert train.EXPERIMENT_ID == "EXP-20260731-005-pnp-only-control"
    assert train.SEED == 42
    assert train.SOLVER.TOTAL_EPOCHS == 40
    assert train.SOLVER.IMS_PER_BATCH == 48
    assert train.MODEL.POSE_NET.BACKBONE.FREEZE
    assert train.MODEL.POSE_NET.GEO_HEAD.FREEZE
    assert train.MODEL.POSE_NET.GEO_HEAD.TRAIN_SUPERVISION is False
    assert not train.MODEL.POSE_NET.PNP_NET.FREEZE
    assert train.MODEL.POSE_NET.PNP_NET.INIT_CFG.type == "ConvPnPNet"
    assert train.MODEL.POSE_NET.QUALITY_COVERAGE.ENABLED is False
    assert train.INPUT.get("HEAD_DEPTH", False) is False
    assert train.SOLVER.OPTIMIZER_CFG.type == "Ranger"
    assert train.SOLVER.OPTIMIZER_CFG.lr == 8e-5

    assert tuple(smoke.DATASETS.TRAIN) == ("lmo_pbr_stage3_local_train",)
    assert tuple(smoke.DATASETS.TEST) == ()
    assert smoke.SOLVER.TOTAL_EPOCHS == 1
    assert smoke.SOLVER.IMS_PER_BATCH == 4
    assert smoke.TEST.EVAL_PERIOD == 0

    assert eval_cfg.MODEL.WEIGHTS == "REPLACE-WITH-INDEXED-CHECKPOINT"
    assert tuple(eval_cfg.DATASETS.TEST) == ("lmo_bop_test",)
    assert eval_cfg.TEST.TEST_BBOX_TYPE == "gt"
    assert eval_cfg.TEST.USE_PNP is False


def test_active_research_configs_load_without_legacy_dependencies():
    for relative_dir in ACTIVE_EXPERIMENT_DIRS:
        base = RESEARCH_CONFIG_ROOT / relative_dir
        for name in ACTIVE_EXPERIMENT_FILES:
            _load(base / name)


def test_head_depth_is_opt_in_for_exp013f_only():
    for relative_dir in ACTIVE_EXPERIMENT_DIRS:
        cfg = _load(RESEARCH_CONFIG_ROOT / relative_dir / "train.py")
        expected = relative_dir == "exp013/f_glm_pose_l"
        assert cfg.INPUT.get("HEAD_DEPTH", False) is expected


def test_frozen_geometry_configs_disable_training_renderer_but_keep_bop_renderer():
    for relative_dir in ACTIVE_EXPERIMENT_DIRS:
        cfg = _load(RESEARCH_CONFIG_ROOT / relative_dir / "train.py")
        pose = cfg.MODEL.POSE_NET
        if pose.GEO_HEAD.FREEZE:
            assert pose.GEO_HEAD.TRAIN_SUPERVISION is False
            assert pose.XYZ_RENDERER is None
        else:
            assert pose.GEO_HEAD.TRAIN_SUPERVISION is True
            assert pose.XYZ_RENDERER in ("cpp", "egl")
        assert cfg.VAL.RENDERER_TYPE in ("cpp", "egl")
