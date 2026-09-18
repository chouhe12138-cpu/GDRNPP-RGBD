"""Resolve the dataset contract shared by EXP022 training and evaluation."""

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

    @property
    def num_objects(self) -> int:
        return len(self.object_ids)


def hierarchy_path_from_config(cfg) -> Path:
    raw = str(cfg.MODEL.POSE_NET.PCC_HEAD.HIERARCHY_PATH)
    expanded = os.path.expanduser(os.path.expandvars(raw))
    if "$" in expanded:
        raise ValueError(f"Unresolved EXP022 hierarchy path: {raw}")
    return Path(expanded)


def resolve_dataset_context(cfg, *, require_hierarchy: bool = True) -> DatasetContext:
    """Register splits, check their class order, then validate the CAD rows."""
    if not cfg.DATASETS.TRAIN:
        raise ValueError("EXP022 requires at least one training split")
    settings = cfg.get("DATASET_CONTEXT", {})
    if not settings:
        raise ValueError("EXP022 requires DATASET_CONTEXT in its config")
    register_datasets_in_cfg(cfg)
    train_name = str(cfg.DATASETS.TRAIN[0])
    train_meta = MetadataCatalog.get(train_name)
    data_ref_key = str(train_meta.ref_key)
    data_ref = ref.__dict__[data_ref_key]
    names = tuple(train_meta.objs)
    ids = tuple(int(data_ref.obj2id[name]) for name in names)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("EXP022 training object IDs must be nonempty and unique")
    # additional training splits may mix domains, but must describe the same
    # objects in the same order as the first one
    for extra_name in list(cfg.DATASETS.TRAIN)[1:]:
        extra_meta = MetadataCatalog.get(str(extra_name))
        extra_ref = ref.__dict__[extra_meta.ref_key]
        extra_ids = tuple(int(extra_ref.obj2id[name]) for name in extra_meta.objs)
        if extra_ids != ids:
            raise ValueError(
                f"EXP022 training split object order mismatch: {extra_name} -> {extra_ids} != {ids}"
            )
    test_name = str(cfg.DATASETS.TEST[0]) if cfg.DATASETS.TEST else None
    if len(cfg.DATASETS.TEST) > 1:
        raise ValueError("EXP022 supports one test split")
    if test_name:
        test_meta = MetadataCatalog.get(test_name)
        test_ref = ref.__dict__[test_meta.ref_key]
        test_ids = tuple(int(test_ref.obj2id[name]) for name in test_meta.objs)
        if test_ids != ids:
            raise ValueError(f"EXP022 train/test object order mismatch: {ids} != {test_ids}")
    if int(cfg.MODEL.POSE_NET.NUM_CLASSES) != len(ids):
        raise ValueError("EXP022 NUM_CLASSES does not match dataset object count")
    cad_ref_key = str(settings.CAD_REF_KEY)
    cad_ref = ref.__dict__[cad_ref_key]
    for name, obj_id in zip(names, ids):
        if int(cad_ref.obj2id[name]) != obj_id:
            raise ValueError(f"EXP022 CAD/data object ID mismatch for {name}")
    hierarchy = hierarchy_path_from_config(cfg)
    if require_hierarchy:
        if not hierarchy.is_file():
            raise FileNotFoundError(f"EXP022 hierarchy missing: {hierarchy}")
        with np.load(hierarchy, allow_pickle=False) as artifact:
            stored_ids = tuple(int(value) for value in artifact["object_ids"])
            if stored_ids != ids:
                raise ValueError(f"EXP022 hierarchy object order mismatch: {stored_ids} != {ids}")
            if "dataset_key" in artifact:
                stored_key = str(artifact["dataset_key"])
                if stored_key != str(settings.KEY):
                    raise ValueError(f"EXP022 hierarchy dataset mismatch: {stored_key}")
            elif str(settings.KEY) != "lmo":
                raise ValueError("EXP022 non-LM-O hierarchy requires dataset_key")
    bop_dataset = str(settings.BOP_DATASET)
    targets = Path(data_ref.bop_root) / bop_dataset / str(settings.BOP_TARGETS_FILENAME)
    return DatasetContext(
        key=str(settings.KEY), train_dataset=train_name, test_dataset=test_name,
        data_ref_key=data_ref_key, cad_ref_key=cad_ref_key,
        object_names=names, object_ids=ids, hierarchy_path=hierarchy,
        bop_dataset=bop_dataset, bop_targets=targets,
        cad_model_dir=Path(cad_ref.model_dir), cad_vertex_scale=float(cad_ref.vertex_scale),
    )
