from pathlib import Path

import json

from mmcv import Config
from detectron2.utils.events import EventStorage

from core.gdrn_modeling.engine.artifact_layout import (
    artifact_dir,
    compact_log_enabled,
    evaluation_dir,
    skip_redundant_final_evaluation,
    tensorboard_enabled,
)
from core.utils.my_writer import EpochJSONWriter


def make_config(structured=True):
    return Config(
        dict(
            OUTPUT_DIR="/tmp/run",
            MODEL=dict(WEIGHTS="/tmp/official.pth"),
            SOLVER=dict(TOTAL_EPOCHS=40),
            TEST=dict(EVAL_PERIOD=5),
            RUN_ARTIFACTS=dict(
                STRUCTURED_LAYOUT=structured,
                COMPACT_LOG=True,
                TENSORBOARD=False,
                SKIP_DUPLICATE_FINAL_EVAL=True,
            ),
        )
    )


def test_structured_paths_are_stable_and_epoch_named():
    cfg = make_config()
    assert artifact_dir(cfg, "checkpoints") == "/tmp/run/checkpoints"
    assert artifact_dir(cfg, "train") == "/tmp/run/train"
    assert (
        evaluation_dir(cfg, "lmo_bop_test", epoch=5, iteration=31989)
        == "/tmp/run/evaluations/epoch_005/lmo_bop_test"
    )
    assert compact_log_enabled(cfg)
    assert not tensorboard_enabled(cfg)
    assert skip_redundant_final_evaluation(cfg)


def test_legacy_layout_is_unchanged():
    cfg = make_config(structured=False)
    assert artifact_dir(cfg, "checkpoints") == "/tmp/run"
    assert (
        evaluation_dir(cfg, "lmo_bop_test", epoch=5, iteration=31989)
        == "/tmp/run/inference_epoch_5_iter_31989/lmo_bop_test"
    )


def test_final_evaluation_is_not_skipped_when_period_does_not_cover_final():
    cfg = make_config()
    cfg.TEST.EVAL_PERIOD = 6
    assert not skip_redundant_final_evaluation(cfg)


def test_epoch_writer_emits_exactly_one_record_per_completed_epoch(tmp_path):
    path = tmp_path / "epoch_summary.jsonl"
    writer = EpochJSONWriter(str(path), iters_per_epoch=2)
    with EventStorage(0) as storage:
        for iteration in range(4):
            storage.iter = iteration
            storage.put_scalar("epoch", iteration // 2 + 1, smoothing_hint=False)
            storage.put_scalar("loss_total", float(4 - iteration))
            writer.write()
    writer.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["epoch"] for record in records] == [1, 2]
    assert [record["iteration"] for record in records] == [1, 3]
