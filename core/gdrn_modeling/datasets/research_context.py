"""Generic dataset and CAD-artifact context for retained research models."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from detectron2.data import MetadataCatalog

import ref
from .dataset_factory import register_datasets_in_cfg


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


def resolve_dataset_context(cfg, *, require_hierarchy: bool = True, hierarchy_path=None):
    """Register configured splits and verify their common object/artifact identity."""
    if not cfg.DATASETS.TRAIN:
        raise ValueError("A research dataset context requires a training split")
    settings = cfg.get("DATASET_CONTEXT", {})
    if not settings:
        raise ValueError("A research dataset context requires DATASET_CONTEXT")
    register_datasets_in_cfg(cfg)
    train_name = str(cfg.DATASETS.TRAIN[0])
    train_meta = MetadataCatalog.get(train_name)
    data_ref_key = str(train_meta.ref_key)
    data_ref = ref.__dict__[data_ref_key]
    names = tuple(train_meta.objs)
    ids = tuple(int(data_ref.obj2id[name]) for name in names)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Training object IDs must be nonempty and unique")
    for extra_name in list(cfg.DATASETS.TRAIN)[1:]:
        extra_meta = MetadataCatalog.get(str(extra_name))
        extra_ref = ref.__dict__[extra_meta.ref_key]
        extra_ids = tuple(int(extra_ref.obj2id[name]) for name in extra_meta.objs)
        if extra_ids != ids:
            raise ValueError(f"Training split object order mismatch: {extra_name} -> {extra_ids} != {ids}")
    if len(cfg.DATASETS.TEST) > 1:
        raise ValueError("The research context supports at most one test split")
    test_name = str(cfg.DATASETS.TEST[0]) if cfg.DATASETS.TEST else None
    if test_name:
        test_meta = MetadataCatalog.get(test_name)
        test_ref = ref.__dict__[test_meta.ref_key]
        test_ids = tuple(int(test_ref.obj2id[name]) for name in test_meta.objs)
        if test_ids != ids:
            raise ValueError(f"Train/test object order mismatch: {ids} != {test_ids}")
    if int(cfg.MODEL.POSE_NET.NUM_CLASSES) != len(ids):
        raise ValueError("NUM_CLASSES does not match the dataset")
    cad_ref_key = str(settings.CAD_REF_KEY)
    cad_ref = ref.__dict__[cad_ref_key]
    for name, obj_id in zip(names, ids):
        if int(cad_ref.obj2id[name]) != obj_id:
            raise ValueError(f"CAD/data object ID mismatch for {name}")
    raw = hierarchy_path if hierarchy_path is not None else cfg.MODEL.POSE_NET.PCC_HEAD.HIERARCHY_PATH
    expanded = os.path.expanduser(os.path.expandvars(str(raw)))
    if "$" in expanded:
        raise ValueError(f"Unresolved hierarchy path: {raw}")
    hierarchy = Path(expanded)
    if require_hierarchy:
        if not hierarchy.is_file():
            raise FileNotFoundError(f"Hierarchy artifact missing: {hierarchy}")
        with np.load(hierarchy, allow_pickle=False) as artifact:
            stored_ids = tuple(int(value) for value in artifact["object_ids"])
            if stored_ids != ids:
                raise ValueError(f"Hierarchy object order mismatch: {stored_ids} != {ids}")
            if "dataset_key" in artifact:
                if str(artifact["dataset_key"]) != str(settings.KEY):
                    raise ValueError("Hierarchy dataset identity mismatch")
            elif str(settings.KEY) != "lmo":
                raise ValueError("A non-LM-O hierarchy requires dataset_key metadata")
    bop_dataset = str(settings.BOP_DATASET)
    return DatasetContext(
        key=str(settings.KEY), train_dataset=train_name, test_dataset=test_name,
        data_ref_key=data_ref_key, cad_ref_key=cad_ref_key, object_names=names,
        object_ids=ids, hierarchy_path=hierarchy, bop_dataset=bop_dataset,
        bop_targets=Path(data_ref.bop_root) / bop_dataset / str(settings.BOP_TARGETS_FILENAME),
        cad_model_dir=Path(cad_ref.model_dir), cad_vertex_scale=float(cad_ref.vertex_scale),
    )
