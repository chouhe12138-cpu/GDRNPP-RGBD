"""Opt-in structured artifact paths for formal research runs."""

from __future__ import annotations

import os.path as osp


def artifact_options(cfg):
    return cfg.get("RUN_ARTIFACTS", {})


def structured_layout_enabled(cfg) -> bool:
    return bool(artifact_options(cfg).get("STRUCTURED_LAYOUT", False))


def compact_log_enabled(cfg) -> bool:
    return bool(artifact_options(cfg).get("COMPACT_LOG", False))


def tensorboard_enabled(cfg) -> bool:
    return bool(artifact_options(cfg).get("TENSORBOARD", True))


def artifact_dir(cfg, kind: str) -> str:
    if not structured_layout_enabled(cfg):
        return cfg.OUTPUT_DIR
    names = {
        "meta": "meta",
        "train": "train",
        "checkpoints": "checkpoints",
        "evaluations": "evaluations",
        "summary": "summary",
    }
    try:
        name = names[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown artifact kind: {kind}") from exc
    return osp.join(cfg.OUTPUT_DIR, name)


def evaluation_dir(cfg, dataset_name: str, epoch=None, iteration=None) -> str:
    if not structured_layout_enabled(cfg):
        if epoch is not None and iteration is not None:
            root = f"inference_epoch_{epoch}_iter_{iteration}"
        else:
            model_name = osp.basename(cfg.MODEL.WEIGHTS).split(".")[0]
            root = f"inference_{model_name}"
        return osp.join(cfg.OUTPUT_DIR, root, dataset_name)

    if epoch is not None:
        root = f"epoch_{int(epoch):03d}"
    else:
        root = "final"
    return osp.join(artifact_dir(cfg, "evaluations"), root, dataset_name)


def skip_redundant_final_evaluation(cfg) -> bool:
    options = artifact_options(cfg)
    if not options.get("SKIP_DUPLICATE_FINAL_EVAL", False):
        return False
    period = int(cfg.TEST.EVAL_PERIOD)
    total_epochs = int(cfg.SOLVER.TOTAL_EPOCHS)
    return period > 0 and total_epochs % period == 0
