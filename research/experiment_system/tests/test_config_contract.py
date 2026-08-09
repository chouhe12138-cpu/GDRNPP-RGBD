from pathlib import Path

import pytest
from mmcv import Config

from research.experiment_system.config_contract import validate_config_contract


def experiment():
    return {
        "experiment_id": "EXP-20260808-008-example",
        "legacy_import": False,
        "protocol": {
            "seed": 7,
            "train_dataset": "train",
            "test_dataset": "test",
            "bbox_type": "gt",
            "total_epochs": 40,
        },
    }


def config():
    return Config(
        dict(
            EXPERIMENT_ID="EXP-20260808-008-example",
            SEED=7,
            OUTPUT_DIR="placeholder",
            DATASETS=dict(TRAIN=("train",), TEST=("test",)),
            TEST=dict(TEST_BBOX_TYPE="gt"),
            SOLVER=dict(TOTAL_EPOCHS=40),
        )
    )


def test_formal_contract_locks_scientific_fields_and_sets_runtime_identity():
    cfg = config()
    validate_config_contract(
        experiment(), cfg, "formal", 7, "RUN-ID", Path("/tmp/run")
    )
    assert cfg.OUTPUT_DIR == "/tmp/run"
    assert cfg.RUN_ID == "RUN-ID"
    assert cfg.SEED == 7


def test_formal_contract_rejects_dataset_drift():
    cfg = config()
    cfg.DATASETS.TRAIN = ("other",)
    with pytest.raises(ValueError, match="training dataset"):
        validate_config_contract(
            experiment(), cfg, "formal", 7, "RUN-ID", Path("/tmp/run")
        )


def test_contract_rejects_requested_seed_drift_for_legacy_experiment():
    legacy = experiment()
    legacy["legacy_import"] = True
    cfg = config()
    with pytest.raises(ValueError, match="config seed"):
        validate_config_contract(
            legacy, cfg, "formal", 42, "RUN-ID", Path("/tmp/run")
        )


def test_contract_rejects_protocol_seed_drift():
    cfg = config()
    cfg.SEED = 42
    with pytest.raises(ValueError, match="protocol seed"):
        validate_config_contract(
            experiment(), cfg, "formal", 42, "RUN-ID", Path("/tmp/run")
        )
