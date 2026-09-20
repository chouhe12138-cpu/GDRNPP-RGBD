"""Dataset and CAD artifact contract owned by EXP025."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from detectron2.data import MetadataCatalog

import ref
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg


@dataclass(frozen=True)
class DatasetContext:
    key: str
    train_dataset: str
    test_dataset: str | None
    data_ref_key: str
    cad_ref_key: str
    object_names: tuple[str, ...]
    object_ids: tuple[int, ...]
    hierarchy_path: Path
    bop_dataset: str
    bop_targets: Path
    cad_model_dir: Path
    cad_vertex_scale: float


def resolve_dataset_context(cfg, *, hierarchy_path=None) -> DatasetContext:
    if not cfg.DATASETS.TRAIN:
        raise ValueError("EXP025 requires one training split")
    settings = cfg.get("DATASET_CONTEXT", {})
    if not settings:
        raise ValueError("EXP025 requires DATASET_CONTEXT")
    register_datasets_in_cfg(cfg)
    train_name = str(cfg.DATASETS.TRAIN[0])
    train_meta = MetadataCatalog.get(train_name)
    data_ref_key = str(train_meta.ref_key)
    data_ref = ref.__dict__[data_ref_key]
    names = tuple(train_meta.objs)
    ids = tuple(int(data_ref.obj2id[name]) for name in names)
    if ids != (1, 5, 6, 8, 9, 10, 11, 12):
        raise ValueError(f"EXP025 requires LM-O object order, got {ids}")
    if len(cfg.DATASETS.TRAIN) != 1 or len(cfg.DATASETS.TEST) > 1:
        raise ValueError("EXP025 requires one train split and at most one test split")
    test_name = str(cfg.DATASETS.TEST[0]) if cfg.DATASETS.TEST else None
    if test_name:
        test_meta = MetadataCatalog.get(test_name)
        test_ref = ref.__dict__[test_meta.ref_key]
        test_ids = tuple(int(test_ref.obj2id[name]) for name in test_meta.objs)
        if test_ids != ids:
            raise ValueError(f"EXP025 train/test object order mismatch: {ids} != {test_ids}")
    if int(cfg.MODEL.POSE_NET.NUM_CLASSES) != len(ids):
        raise ValueError("EXP025 NUM_CLASSES does not match the dataset")
    cad_ref = ref.__dict__[str(settings.CAD_REF_KEY)]
    raw = hierarchy_path or cfg.MODEL.POSE_NET.CAD_ATTENTION_HEAD.HIERARCHY_PATH
    expanded = os.path.expanduser(os.path.expandvars(str(raw)))
    if "$" in expanded:
        raise ValueError(f"Unresolved EXP025 hierarchy path: {raw}")
    hierarchy = Path(expanded)
    if not hierarchy.is_file():
        raise FileNotFoundError(f"EXP025 hierarchy missing: {hierarchy}")
    with np.load(hierarchy, allow_pickle=False) as artifact:
        stored_ids = tuple(int(value) for value in artifact["object_ids"])
        if stored_ids != ids or str(artifact.get("dataset_key", "lmo")) != str(settings.KEY):
            raise ValueError("EXP025 hierarchy dataset/object identity mismatch")
    targets = Path(data_ref.bop_root) / str(settings.BOP_DATASET) / str(settings.BOP_TARGETS_FILENAME)
    return DatasetContext(str(settings.KEY), train_name, test_name, data_ref_key,
                          str(settings.CAD_REF_KEY), names, ids, hierarchy,
                          str(settings.BOP_DATASET), targets, Path(cad_ref.model_dir),
                          float(cad_ref.vertex_scale))
