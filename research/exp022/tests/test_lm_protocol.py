"""LM real and lm_imgn loaders, and the LM13 GDR-Net protocol contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import ref
from mmcv import Config

from core.gdrn_modeling.datasets.data_loader import background_replace_probability
from core.gdrn_modeling.datasets.lm_dataset_d2 import LM_D2_Dataset, SPLITS_LM
from core.gdrn_modeling.datasets.lm_pbr import LM_13_OBJECTS
from core.gdrn_modeling.datasets.lm_syn_imgn import LM_SYN_IMGN_Dataset, SPLITS_LM_IMGN
from research.exp022.dataset_context import resolve_dataset_context
from research.exp022.eval_manifest import build_manifest, write_manifest
from research.exp022.preflight import check_lm13_gdrn_protocol


LM_ROOT = Path("datasets/BOP_DATASETS/lm")
IMGN_ROOT = Path("datasets/lm_imgn")
CONFIG_DIR = "configs/gdrn/research/exp022_progressive_pcc"
GDRN_CONFIG = f"{CONFIG_DIR}/train_lm13_gdrn.py"
REAL_ONLY_CONFIG = f"{CONFIG_DIR}/train_lm13_real_only.py"
PBR_CONFIG = f"{CONFIG_DIR}/train_lm13_pbr.py"
BOP_EVAL_CONFIG = f"{CONFIG_DIR}/eval_lm13_bop.py"


def _bounded(split_dict, tmp_path, **overrides):
    """A cache-free probe that loads only a handful of records."""

    return dict(split_dict, use_cache=False, cache_dir=str(tmp_path), **overrides)


def test_splits_declare_the_thirteen_lm_objects():
    for name in ("lm_13_train", "lm_13_train_online", "lm_13_test", "lm_13_test_online"):
        assert SPLITS_LM[name]["objs"] == LM_13_OBJECTS
        assert SPLITS_LM[name]["ref_key"] == "lm_full"
    assert SPLITS_LM["lm_13_train"].get("require_xyz", True) is True
    for name in ("lm_13_train_online", "lm_13_train_smoke"):
        assert SPLITS_LM[name]["require_xyz"] is False
    for name in ("lm_imgn_13_train_1k_per_obj", "lm_imgn_13_train_1k_per_obj_online"):
        cfg = SPLITS_LM_IMGN[name]
        assert cfg["objs"] == LM_13_OBJECTS and cfg["n_per_obj"] == 1000
    assert SPLITS_LM_IMGN["lm_imgn_13_train_1k_per_obj"].get("require_xyz", True) is True
    assert SPLITS_LM_IMGN["lm_imgn_13_train_1k_per_obj_online"]["require_xyz"] is False


def _image_set_ids(obj_name, split):
    return {
        int(token)
        for token in (LM_ROOT / "image_set" / f"{obj_name}_{split}.txt").read_text().split()
    }


def test_image_set_selection_resolves_inside_the_test_scenes():
    """The loader reads lm/test with image_set, so every id must exist there."""

    for obj_name in LM_13_OBJECTS:
        obj_id = int(ref.lm_full.obj2id[obj_name])
        available = {
            int(path.stem)
            for path in (LM_ROOT / "test" / f"{obj_id:06d}" / "rgb").glob("*.png")
        }
        train, test = _image_set_ids(obj_name, "train"), _image_set_ids(obj_name, "test")
        assert train <= available and test <= available, obj_name
        assert train.isdisjoint(test), obj_name


def test_the_physical_lm_train_directory_is_a_subset_of_the_official_split():
    """lm/train is a convenience copy, not the authority for the split.

    It is missing two benchvise images (891, 892) that the official
    `benchvise_train.txt` selects, which is why the loader reads lm/test.
    """

    for obj_name in LM_13_OBJECTS:
        obj_id = int(ref.lm_full.obj2id[obj_name])
        physical = {
            int(path.stem)
            for path in (LM_ROOT / "train" / f"{obj_id:06d}" / "rgb").glob("*.png")
        }
        assert physical <= _image_set_ids(obj_name, "train"), obj_name


def test_lm_real_loader_returns_bop_records(tmp_path):
    dicts = LM_D2_Dataset(_bounded(SPLITS_LM["lm_13_train_online"], tmp_path, num_to_load=6))()
    assert len(dicts) == 6
    record = dicts[0]
    assert record["img_type"] == "real"
    assert record["depth_factor"] == 1000.0
    assert record["cam"].shape == (3, 3)
    inst = record["annotations"][0]
    assert inst["category_id"] == 0  # the first LM object is ape
    assert inst["pose"].shape == (3, 4)
    assert {"segmentation", "mask_full", "bbox", "bbox_obj", "visib_fract"} <= set(inst)
    # translations are metres, so a table-top object sits between 0.5 m and 2 m
    assert 0.5 < float(inst["trans"][2]) < 2.0


def test_lm_imgn_loader_uses_the_render_conventions(tmp_path):
    dicts = LM_SYN_IMGN_Dataset(
        _bounded(SPLITS_LM_IMGN["lm_imgn_13_train_1k_per_obj_online"],
                 tmp_path, n_per_obj=4, num_to_load=4)
    )()
    assert len(dicts) == 4
    record = dicts[0]
    assert record["img_type"] == "syn"
    assert record["file_name"].endswith("-color.png")
    assert record["depth_factor"] == 1000.0  # renders store depth in mm
    assert np.array_equal(record["cam"], np.array(ref.lm_full.camera_matrix, dtype=np.float32))
    inst = record["annotations"][0]
    assert inst["pose"].shape == (3, 4)
    assert 0.5 < float(inst["trans"][2]) < 2.0
    assert np.array_equal(inst["bbox"], inst["bbox_obj"])  # no separate visibility


def test_benchviseblue_renders_map_to_the_benchvise_object(tmp_path):
    dicts = LM_SYN_IMGN_Dataset(
        _bounded(SPLITS_LM_IMGN["lm_imgn_13_train_1k_per_obj_online"],
                 tmp_path, objs=["benchvise"], n_per_obj=3, num_to_load=3)
    )()
    assert len(dicts) == 3
    assert all(record["scene_im_id"].startswith("benchviseblue/") for record in dicts)
    inst = dicts[0]["annotations"][0]
    assert inst["category_id"] == 0  # the only probed object
    assert inst["model_info"]["diameter"] > 0


def test_background_policy_is_explicit_per_domain():
    cfg = Config.fromfile(GDRN_CONFIG)
    assert background_replace_probability(cfg, "syn") == 1.0
    assert background_replace_probability(cfg, "real") == 0.5
    assert background_replace_probability(cfg, "syn_pbr") == 0.5
    cfg.INPUT.PBR_CHANGE_BG_PROB = 0.25
    assert background_replace_probability(cfg, "syn_pbr") == 0.25
    with pytest.raises(ValueError, match="Unknown img_type"):
        background_replace_probability(cfg, "unexpected")


def test_gdrn_config_matches_the_protocol():
    cfg = Config.fromfile(GDRN_CONFIG)
    report = check_lm13_gdrn_protocol(cfg)
    assert report["train_splits"] == ["lm_13_train_online", "lm_imgn_13_train_1k_per_obj_online"]
    assert (report["total_epochs"], report["reference_batch_size"]) == (160, 24)
    assert str(cfg.SOLVER.OPTIMIZER_CFG.type) == "Ranger"
    assert float(cfg.SOLVER.OPTIMIZER_CFG.lr) == 1e-4
    assert float(cfg.SOLVER.OPTIMIZER_CFG.weight_decay) == 0.0
    assert cfg.SOLVER.get("WARMUP_RATIO", None) is None
    assert float(cfg.DATALOADER.FILTER_VISIB_THR) == 0.0
    assert cfg.SEED == 42
    assert int(cfg.TEST.EVAL_PERIOD) > 0


def test_real_only_changes_only_the_training_splits():
    main = Config.fromfile(GDRN_CONFIG)
    ablation = Config.fromfile(REAL_ONLY_CONFIG)
    assert ablation.DATASETS.TRAIN == ("lm_13_train_online",)
    assert main.DATASETS.TRAIN == ("lm_13_train_online", "lm_imgn_13_train_1k_per_obj_online")
    for key in ("COLOR_AUG_PROB", "CHANGE_BG_PROB", "PBR_CHANGE_BG_PROB",
                "DZI_PAD_SCALE", "DZI_SCALE_RATIO", "DZI_SHIFT_RATIO"):
        assert ablation.INPUT[key] == main.INPUT[key]
    assert ablation.SOLVER.TOTAL_EPOCHS == main.SOLVER.TOTAL_EPOCHS
    assert ablation.SOLVER.REFERENCE_BS == main.SOLVER.REFERENCE_BS
    assert ablation.SOLVER.OPTIMIZER_CFG.type == main.SOLVER.OPTIMIZER_CFG.type


def test_pbr_config_is_labelled_as_its_own_domain():
    cfg = Config.fromfile(PBR_CONFIG)
    assert cfg.DATASETS.TRAIN == ("lm_pbr_13_online_train",)
    assert cfg.TRAIN_PROTOCOL.NAME == "lm13_pbr"
    assert float(cfg.INPUT.COLOR_AUG_PROB) == 0.8


@pytest.mark.parametrize("key,value", [("COLOR_AUG_PROB", 0.8), ("TRUNCATE_FG", True),
                                       ("DZI_SCALE_RATIO", 0.5), ("CHANGE_BG_PROB", 0.0)])
def test_protocol_check_rejects_input_drift(key, value):
    cfg = Config.fromfile(GDRN_CONFIG)
    cfg.INPUT[key] = value
    with pytest.raises(ValueError, match=key):
        check_lm13_gdrn_protocol(cfg)


def test_protocol_check_rejects_a_warmup_ratio():
    cfg = Config.fromfile(GDRN_CONFIG)
    cfg.SOLVER.WARMUP_RATIO = 0.04  # would silently replace ANNEAL_POINT
    with pytest.raises(ValueError, match="WARMUP_RATIO"):
        check_lm13_gdrn_protocol(cfg)


def test_protocol_check_rejects_a_different_data_domain():
    cfg = Config.fromfile(GDRN_CONFIG)
    cfg.DATASETS.TRAIN = ("lm_13_train_online",)
    with pytest.raises(ValueError, match="training splits"):
        check_lm13_gdrn_protocol(cfg)


def test_protocol_check_rejects_a_non_ranger_baseline():
    cfg = Config.fromfile(GDRN_CONFIG)
    cfg.SOLVER.OPTIMIZER_CFG.type = "AdamW"
    with pytest.raises(ValueError, match="Ranger"):
        check_lm13_gdrn_protocol(cfg)


def test_eval_configs_declare_their_own_protocols():
    legacy = Config.fromfile(GDRN_CONFIG)
    bop = Config.fromfile(BOP_EVAL_CONFIG)
    assert legacy.DATASETS.TEST == ("lm_13_test",)
    assert legacy.VAL.TARGETS_FILENAME == "lm_test_targets_bb8.json"
    assert legacy.VAL.USE_BOP is False
    assert legacy.EVAL_PROTOCOL.NAME == "lm_legacy_diagnostic"
    assert bop.DATASETS.TEST == ("lm_bop_test_13",)
    assert bop.VAL.TARGETS_FILENAME == "test_targets_bop19.json"
    assert bop.VAL.USE_BOP is True
    assert bop.EVAL_PROTOCOL.NAME == "bop_official"
    # both are GT-box runs until the official detector boxes are installed
    assert legacy.EVAL_PROTOCOL.BBOX_SOURCE == bop.EVAL_PROTOCOL.BBOX_SOURCE == "gt"
    # only the evaluation differs; the training definition is shared
    assert bop.DATASETS.TRAIN == legacy.DATASETS.TRAIN
    assert bop.SOLVER.OPTIMIZER_CFG.type == legacy.SOLVER.OPTIMIZER_CFG.type


@pytest.mark.parametrize("config,dataset,test_split,protocol,targets", [
    (GDRN_CONFIG, "lm13", "lm_13_test", "lm_legacy_diagnostic", "lm_test_targets_bb8.json"),
    (BOP_EVAL_CONFIG, "lm13", "lm_bop_test_13", "bop_official", "test_targets_bop19.json"),
])
def test_manifest_records_the_evaluation_provenance(tmp_path, config, dataset, test_split,
                                                    protocol, targets):
    cfg = Config.fromfile(config)
    context = resolve_dataset_context(cfg)
    manifest = build_manifest(cfg, context, checkpoint="checkpoints/model.pth")
    assert manifest["experiment_id"] == cfg.EXPERIMENT_ID
    assert manifest["dataset_context"] == dataset
    assert manifest["dataset_protocol"] == "lm13_gdrn"
    assert manifest["data_domain"] == "real+imgn"
    assert manifest["train_datasets"] == ["lm_13_train_online", "lm_imgn_13_train_1k_per_obj_online"]
    assert manifest["test_dataset"] == test_split
    assert manifest["eval_protocol"] == protocol
    assert manifest["bbox_source"] == "gt"
    assert manifest["targets_filename"] == targets
    assert manifest["pose_solver"] == "ransac_pnp"
    assert manifest["checkpoint"] == "checkpoints/model.pth"
    assert manifest["object_ids"] == [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    path = write_manifest(tmp_path, manifest)
    assert json.loads(path.read_text(encoding="utf-8")) == manifest


def test_pbr_arm_is_not_reported_as_the_gdrn_protocol():
    cfg = Config.fromfile(PBR_CONFIG)
    manifest = build_manifest(cfg, resolve_dataset_context(cfg), checkpoint="model.pth",
                              extra={"eval_protocol": "bop_official"})
    assert manifest["dataset_protocol"] == "lm13_pbr"
    assert manifest["data_domain"] == "syn_pbr"
    assert manifest["train_datasets"] == ["lm_pbr_13_online_train"]


def test_mixed_domain_config_passes_the_context_check():
    cfg = Config.fromfile(GDRN_CONFIG)
    context = resolve_dataset_context(cfg)
    assert context.object_ids == (1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15)
    assert context.test_dataset == "lm_13_test"


def test_context_rejects_a_training_split_with_another_object_order():
    cfg = Config.fromfile(GDRN_CONFIG)
    cfg.DATASETS.TRAIN = ("lm_13_train_online", "lmo_pbr_train")
    with pytest.raises(ValueError, match="training split object order mismatch"):
        resolve_dataset_context(cfg)
