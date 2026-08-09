"""Validate scientific config identity before creating a new-system run."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_config_contract(
    experiment: dict[str, Any],
    cfg: Any,
    mode: str,
    seed: int,
    run_id: str,
    run_dir: Path,
) -> None:
    """Validate immutable fields, then apply only runtime identity/output fields."""

    protocol = experiment["protocol"]
    configured_seed = cfg.get("SEED")
    if configured_seed is None or int(configured_seed) != int(seed):
        raise ValueError("config seed does not match the requested run seed")
    if "seed" in protocol and int(protocol["seed"]) != int(seed):
        raise ValueError("experiment protocol seed does not match the requested run seed")

    configured_experiment = cfg.get("EXPERIMENT_ID")
    if configured_experiment is not None:
        if configured_experiment != experiment["experiment_id"]:
            raise ValueError(
                "config EXPERIMENT_ID does not match the registered experiment identity"
            )
    elif not experiment["legacy_import"]:
        raise ValueError(
            "config EXPERIMENT_ID does not match the registered experiment identity"
        )

    if mode == "formal":
        datasets = cfg.get("DATASETS", {})
        expected_train = protocol.get("train_dataset")
        if expected_train and tuple(datasets.get("TRAIN", ())) != (expected_train,):
            raise ValueError("formal training dataset differs from the experiment protocol")
        expected_test = protocol.get("test_dataset") or protocol.get("dataset")
        if expected_test and tuple(datasets.get("TEST", ())) != (expected_test,):
            raise ValueError("formal test dataset differs from the experiment protocol")
        expected_bbox = protocol.get("bbox_type")
        if expected_bbox and cfg.get("TEST", {}).get("TEST_BBOX_TYPE") != expected_bbox:
            raise ValueError("formal bbox type differs from the experiment protocol")
        expected_epochs = protocol.get("total_epochs")
        if expected_epochs is not None and int(cfg.SOLVER.TOTAL_EPOCHS) != int(expected_epochs):
            raise ValueError("formal epoch budget differs from the experiment protocol")

    cfg.OUTPUT_DIR = str(run_dir)
    cfg.EXPERIMENT_ID = experiment["experiment_id"]
    cfg.SEED = int(seed)
    cfg.RUN_ID = run_id
