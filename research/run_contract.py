#!/usr/bin/env python3
"""Shared safety contract for research smoke, formal, and evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from mmcv import Config

from core.gdrn_modeling.engine.engine_utils import geometry_supervision_enabled


def _evaluation_renderer(cfg: Config) -> Optional[str]:
    raw = cfg.VAL.get("RENDERER_TYPE", None)
    if raw is None or raw is False:
        return None
    value = str(raw).strip().lower()
    if value in {"", "0", "false", "none", "disabled"}:
        return None
    if value not in {"cpp", "egl"}:
        raise ValueError(f"Unsupported BOP evaluation renderer: {raw!r}")
    return value


def validate_research_run_config(
    cfg: Config,
    *,
    mode: str,
    expected_experiment_id: Optional[str] = None,
) -> dict[str, object]:
    """Validate the effective config before a managed run creates output."""
    if mode not in {"smoke", "formal", "eval", "prepare"}:
        raise ValueError(f"Unsupported run mode: {mode}")
    if expected_experiment_id is not None and cfg.EXPERIMENT_ID != expected_experiment_id:
        raise ValueError(
            f"EXPERIMENT_ID mismatch: config={cfg.EXPERIMENT_ID!r}, "
            f"launcher={expected_experiment_id!r}"
        )

    protocol_name = str(cfg.get('TRAIN_PROTOCOL', {}).get('NAME', ''))
    if protocol_name == 'exp026_lmo' and mode != 'prepare' \
            and not bool(cfg.get('RESEARCH_PROTOCOL', {}).get('SERVER_RELEASE_ALLOWED', False)):
        raise ValueError('EXP026 SERVER_BLOCKED: release not authorized')

    training_supervision = geometry_supervision_enabled(cfg)
    evaluation_renderer = _evaluation_renderer(cfg)

    protocol = cfg.get("RESEARCH_PROTOCOL", {})
    configurable = protocol.get("SCHEDULE") == "configurable"
    if configurable:
        if int(cfg.SEED) < 0:
            raise ValueError("Configurable research runs require a fixed nonnegative seed")
        for key in ("TOTAL_EPOCHS", "IMS_PER_BATCH", "REFERENCE_BS", "CHECKPOINT_PERIOD"):
            if int(cfg.SOLVER[key]) <= 0:
                raise ValueError(f"SOLVER.{key} must be positive")
        if int(cfg.SOLVER.REFERENCE_BS) % int(cfg.SOLVER.IMS_PER_BATCH):
            raise ValueError("REFERENCE_BS must be divisible by IMS_PER_BATCH")
        if int(cfg.TEST.EVAL_PERIOD) < 0:
            raise ValueError("TEST.EVAL_PERIOD must be nonnegative")
        if not bool(cfg.SOLVER.AMP.ENABLED):
            raise ValueError("New formal research configurations require AMP")
    elif int(cfg.SEED) != 42:
        raise ValueError(f"Research runs require seed 42, got {cfg.SEED}")

    if mode == "smoke":
        if int(cfg.SOLVER.TOTAL_EPOCHS) != 1:
            raise ValueError("smoke requires exactly one epoch")
        if int(cfg.SOLVER.IMS_PER_BATCH) > 8:
            raise ValueError("smoke batch size must be at most 8")
        if int(cfg.SOLVER.CHECKPOINT_PERIOD) != 1:
            raise ValueError("smoke requires an epoch-1 checkpoint")
        if int(cfg.TEST.EVAL_PERIOD) != 0 or tuple(cfg.DATASETS.TEST):
            raise ValueError("smoke must disable periodic/formal evaluation")
        if bool(cfg.SOLVER.get("BEST_CHECKPOINT", {}).get("ENABLED", False)):
            raise ValueError("smoke must disable best-checkpoint selection")

    if mode == "formal":
        if configurable:
            if not bool(protocol.get("FORMAL_READY", False)):
                raise ValueError("Configurable formal protocol is not ready")
            if not cfg.DATASETS.TRAIN or not cfg.DATASETS.TEST:
                raise ValueError("Formal protocol requires train and test splits")
            if int(cfg.TEST.EVAL_PERIOD) <= 0:
                raise ValueError("Formal protocol requires periodic evaluation")
            if int(cfg.SOLVER.CHECKPOINT_PERIOD) > int(cfg.SOLVER.TOTAL_EPOCHS):
                raise ValueError("Checkpoint period exceeds total epochs")
            if int(cfg.TEST.EVAL_PERIOD) > int(cfg.SOLVER.TOTAL_EPOCHS):
                raise ValueError("Evaluation period exceeds total epochs")
            if str(cfg.TEST.TEST_BBOX_TYPE).lower() != "gt" or evaluation_renderer is None:
                raise ValueError("Formal protocol requires GT box and evaluation renderer")
            if protocol_name == "exp025_lmo":
                expected = {
                    "SEED": 42,
                    "TOTAL_EPOCHS": 40,
                    "IMS_PER_BATCH": 48,
                    "REFERENCE_BS": 48,
                    "CHECKPOINT_PERIOD": 5,
                    "EVAL_PERIOD": 5,
                }
                actual = {
                    "SEED": int(cfg.SEED),
                    "TOTAL_EPOCHS": int(cfg.SOLVER.TOTAL_EPOCHS),
                    "IMS_PER_BATCH": int(cfg.SOLVER.IMS_PER_BATCH),
                    "REFERENCE_BS": int(cfg.SOLVER.REFERENCE_BS),
                    "CHECKPOINT_PERIOD": int(cfg.SOLVER.CHECKPOINT_PERIOD),
                    "EVAL_PERIOD": int(cfg.TEST.EVAL_PERIOD),
                }
                if actual != expected:
                    raise ValueError(f"EXP025 formal protocol mismatch: expected={expected}, actual={actual}")
                if tuple(cfg.DATASETS.TRAIN) != ("lmo_pbr_train",) or tuple(cfg.DATASETS.TEST) != ("lmo_bop_test",):
                    raise ValueError("EXP025 formal requires LM-PBR train and LM-O test")
                if str(cfg.MODEL.POSE_NET.XYZ_RENDERER).lower() != "egl" or evaluation_renderer != "cpp":
                    raise ValueError("EXP025 formal requires EGL training and CPP evaluation")
                if not bool(cfg.SOLVER.CHECKPOINT_BY_EPOCH):
                    raise ValueError("EXP025 formal checkpoints must be epoch based")
                scale = float(cfg.SOLVER.AMP.get("INIT_SCALE", 0))
                if scale < 1:
                    raise ValueError("EXP025 formal requires the server-gated AMP.INIT_SCALE")
                arm = str(cfg.get("EXP025_ARM", ""))
                if arm not in {"official_frozen", "imagenet_full"}:
                    raise ValueError(f"Unknown EXP025 arm: {arm}")
            elif protocol_name == 'exp026_lmo':
                actual = (int(cfg.SEED), int(cfg.SOLVER.TOTAL_EPOCHS),
                          int(cfg.SOLVER.IMS_PER_BATCH), int(cfg.SOLVER.REFERENCE_BS),
                          int(cfg.SOLVER.CHECKPOINT_PERIOD), int(cfg.TEST.EVAL_PERIOD))
                if actual != (42, 40, 48, 48, 5, 5):
                    raise ValueError(f'EXP026 formal protocol mismatch: {actual}')
                if tuple(cfg.DATASETS.TRAIN) != ('lmo_pbr_train',) or tuple(cfg.DATASETS.TEST) != ('lmo_bop_test',):
                    raise ValueError('EXP026 formal requires LM-O PBR40 and BOP test')
                if str(cfg.MODEL.POSE_NET.XYZ_RENDERER).lower() != 'egl' or evaluation_renderer != 'cpp':
                    raise ValueError('EXP026 formal requires EGL training and CPP evaluation')
                if str(cfg.TEST.TEST_BBOX_TYPE).lower() != 'gt' or not bool(cfg.SOLVER.CHECKPOINT_BY_EPOCH):
                    raise ValueError('EXP026 formal requires GT boxes and epoch checkpoints')
                if (cfg.BACKBONE_INIT, bool(cfg.TRAIN_BACKBONE), float(cfg.BACKBONE_LR_MULT)) != ('imagenet', True, 1.):
                    raise ValueError('EXP026 formal requires ImageNet Full training')
                if cfg.MODEL.WEIGHTS or cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.INIT_CFG.residual_target_mode != 'predicted_route':
                    raise ValueError('EXP026 formal requires fresh predicted-route residual training')
                if str(cfg.EXP026_ARM) not in ('uniform_full', 'adaptive_l1_full'):
                    raise ValueError(f'Unknown EXP026 arm: {cfg.EXP026_ARM}')
                if float(cfg.SOLVER.AMP.get('INIT_SCALE', 0)) != 16384:
                    raise ValueError('EXP026 formal requires locally gated AMP initial scale 16384')
            elif protocol_name == "exp025_lm13":
                expected = {
                    "SEED": 42,
                    "TOTAL_EPOCHS": 160,
                    "IMS_PER_BATCH": 4,
                    "REFERENCE_BS": 24,
                    "CHECKPOINT_PERIOD": 20,
                    "EVAL_PERIOD": 20,
                }
                actual = {
                    "SEED": int(cfg.SEED),
                    "TOTAL_EPOCHS": int(cfg.SOLVER.TOTAL_EPOCHS),
                    "IMS_PER_BATCH": int(cfg.SOLVER.IMS_PER_BATCH),
                    "REFERENCE_BS": int(cfg.SOLVER.REFERENCE_BS),
                    "CHECKPOINT_PERIOD": int(cfg.SOLVER.CHECKPOINT_PERIOD),
                    "EVAL_PERIOD": int(cfg.TEST.EVAL_PERIOD),
                }
                if actual != expected:
                    raise ValueError(f"EXP025 LM13 protocol mismatch: expected={expected}, actual={actual}")
                if tuple(cfg.DATASETS.TRAIN) != (
                    "lm_13_train_online", "lm_imgn_13_train_1k_per_obj_online"
                ) or tuple(cfg.DATASETS.TEST) != ("lm_13_test_online",):
                    raise ValueError("EXP025 LM13 requires the historical real+DeepIM protocol")
                if str(cfg.MODEL.POSE_NET.XYZ_RENDERER).lower() != "egl" or evaluation_renderer != "cpp":
                    raise ValueError("EXP025 LM13 requires EGL training and CPP evaluation")
                optimizer = cfg.SOLVER.OPTIMIZER_CFG
                if (str(optimizer.type) != "Ranger" or float(optimizer.lr) != 1e-4
                        or float(optimizer.weight_decay) != 0):
                    raise ValueError("EXP025 LM13 requires Ranger lr=1e-4 weight_decay=0")
                schedule = {
                    "WARMUP_RATIO": cfg.SOLVER.get("WARMUP_RATIO", None),
                    "WARMUP_ITERS": int(cfg.SOLVER.WARMUP_ITERS),
                    "ANNEAL_METHOD": str(cfg.SOLVER.ANNEAL_METHOD),
                    "ANNEAL_POINT": float(cfg.SOLVER.ANNEAL_POINT),
                    "TARGET_LR_FACTOR": float(cfg.SOLVER.TARGET_LR_FACTOR),
                }
                if schedule != {"WARMUP_RATIO": None, "WARMUP_ITERS": 1000,
                                "ANNEAL_METHOD": "cosine", "ANNEAL_POINT": .72,
                                "TARGET_LR_FACTOR": 0.}:
                    raise ValueError(f"EXP025 LM13 flat-and-anneal mismatch: {schedule}")
                if str(cfg.EXP025_ARM) != "lm13_imagenet_full_prepared":
                    raise ValueError(f"Unknown EXP025 LM13 arm: {cfg.EXP025_ARM}")
                if float(cfg.SOLVER.AMP.get("INIT_SCALE", 0)) < 1:
                    raise ValueError("EXP025 LM13 formal requires a server-gated AMP.INIT_SCALE")
            return _summary(cfg, mode, training_supervision, evaluation_renderer)
        expected = {
            "TOTAL_EPOCHS": 40,
            "IMS_PER_BATCH": 48,
            "CHECKPOINT_PERIOD": 5,
            "EVAL_PERIOD": 5,
        }
        actual = {
            "TOTAL_EPOCHS": int(cfg.SOLVER.TOTAL_EPOCHS),
            "IMS_PER_BATCH": int(cfg.SOLVER.IMS_PER_BATCH),
            "CHECKPOINT_PERIOD": int(cfg.SOLVER.CHECKPOINT_PERIOD),
            "EVAL_PERIOD": int(cfg.TEST.EVAL_PERIOD),
        }
        if actual != expected:
            raise ValueError(f"formal protocol mismatch: expected={expected}, actual={actual}")
        if not bool(cfg.SOLVER.CHECKPOINT_BY_EPOCH):
            raise ValueError("formal checkpoints must be epoch based")
        if tuple(cfg.DATASETS.TRAIN) != ("lmo_pbr_train",):
            raise ValueError("formal requires the LM-PBR training dataset")
        if tuple(cfg.DATASETS.TEST) != ("lmo_bop_test",):
            raise ValueError("formal requires the LM-O BOP19 evaluation dataset")
        if str(cfg.TEST.TEST_BBOX_TYPE).lower() != "gt":
            raise ValueError("formal requires LM-O GT-box evaluation")
        if evaluation_renderer is None:
            raise ValueError("formal BOP evaluation renderer must be cpp or egl")

    if mode == "eval":
        if not tuple(cfg.DATASETS.TEST):
            raise ValueError("independent evaluation requires a test dataset")
        if evaluation_renderer is None:
            raise ValueError("independent BOP evaluation renderer must be cpp or egl")

    return _summary(cfg, mode, training_supervision, evaluation_renderer)


def _summary(cfg, mode, training_supervision, evaluation_renderer):
    return {
        "mode": mode,
        "experiment_id": str(cfg.EXPERIMENT_ID),
        "seed": int(cfg.SEED),
        "total_epochs": int(cfg.SOLVER.TOTAL_EPOCHS),
        "batch_size": int(cfg.SOLVER.IMS_PER_BATCH),
        "checkpoint_period": int(cfg.SOLVER.CHECKPOINT_PERIOD),
        "evaluation_period": int(cfg.TEST.EVAL_PERIOD),
        "training_geometry_supervision": bool(training_supervision),
        "training_renderer": cfg.MODEL.POSE_NET.get("XYZ_RENDERER", None),
        "evaluation_renderer": evaluation_renderer,
        "amp_enabled": bool(cfg.SOLVER.get("AMP", {}).get("ENABLED", False)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "formal", "eval", "prepare"))
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(str(args.config))
    result = validate_research_run_config(
        cfg,
        mode=args.mode,
        expected_experiment_id=args.experiment_id,
    )
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
