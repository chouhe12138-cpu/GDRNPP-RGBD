"""EXP025 runtime compatibility exports and immutable diagnostic metadata."""
from __future__ import annotations

import subprocess

from research.cad_common.runtime import (
    NonFiniteTrainingError, amp_init_scale, amp_step, grad_norm_stats,
    head_telemetry, load_last_good, raw_residual_stats, restore_rng, rng_state,
    save_last_good, save_report, seed_all, tensor_stats, token_norm_stats,
)
from research.cad_common.runtime import real_batch as _shared_real_batch


def metadata(cfg):
    from .configuration import require_hierarchy
    hierarchy = cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH
    return dict(experiment_id=cfg.EXPERIMENT_ID, seed=42, backbone_init=cfg.BACKBONE_INIT,
                train_backbone=bool(cfg.TRAIN_BACKBONE), config=cfg.filename,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                hierarchy=hierarchy, hierarchy_sha256=require_hierarchy(hierarchy, cfg.DATASET_CONTEXT.KEY),
                training_kind='diagnostic', formal=False)


def real_batch(cfg, device, batch_size, renderer_type, load_batch=None, save_batch=None):
    """Preserve historical unlabelled LM-O batches for EXP025 diagnostics only."""
    allow_unlabeled = str(cfg.DATASET_CONTEXT.KEY) == 'lmo'
    return _shared_real_batch(cfg, device, batch_size, renderer_type, load_batch, save_batch,
                              allow_legacy_unlabeled=allow_unlabeled)
