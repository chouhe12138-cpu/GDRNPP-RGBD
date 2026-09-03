from __future__ import annotations

import importlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from detectron2.data import MetadataCatalog
from mmcv import Config

import ref
from core.gdrn_modeling.datasets.data_loader import build_gdrn_train_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import (
    batch_data,
    geometry_supervision_enabled,
    get_renderer,
)
from core.utils.my_checkpoint import MyCheckpointer

from .common import DiagnosticBatch
from .model_access import capture_model_pose_call, decode_raw_pose


@dataclass
class DiagnosticRuntime:
    cfg: Any
    model: torch.nn.Module
    loader: Any
    renderer: Any
    device: torch.device
    checkpoint: str

    def close(self) -> None:
        if self.renderer is not None and hasattr(self.renderer, "close"):
            self.renderer.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_cfg(config_file: str, batch_size: int, num_workers: int, device: str, opts: Optional[Dict[str, Any]] = None):
    cfg = Config.fromfile(config_file)
    if opts:
        cfg.merge_from_dict(opts)
    cfg.MODEL.DEVICE = device
    cfg.DATALOADER.NUM_WORKERS = int(num_workers)
    cfg.DATALOADER.PERSISTENT_WORKERS = bool(num_workers > 0 and cfg.DATALOADER.get("PERSISTENT_WORKERS", False))
    cfg.SOLVER.IMS_PER_BATCH = int(batch_size)
    cfg.SOLVER.REFERENCE_BS = int(batch_size)

    # Mirror the pieces of main_gdrn.setup needed by model construction.
    cfg.SOLVER.pop("STEPS", None)
    cfg.SOLVER.pop("MAX_ITER", None)
    optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
    if isinstance(optim_cfg, str) and optim_cfg:
        optim_cfg = eval(optim_cfg)
        cfg.SOLVER.OPTIMIZER_CFG = optim_cfg
    if isinstance(optim_cfg, dict):
        cfg.SOLVER.OPTIMIZER_NAME = optim_cfg.get("type", cfg.SOLVER.get("OPTIMIZER_NAME", ""))
        cfg.SOLVER.BASE_LR = float(optim_cfg.get("lr", cfg.SOLVER.get("BASE_LR", 1e-4)))
        cfg.SOLVER.MOMENTUM = float(optim_cfg.get("momentum", cfg.SOLVER.get("MOMENTUM", 0.9)))
        cfg.SOLVER.WEIGHT_DECAY = float(optim_cfg.get("weight_decay", cfg.SOLVER.get("WEIGHT_DECAY", 1e-4)))
    register_datasets_in_cfg(cfg)
    return cfg


def build_runtime(
    config_file: str,
    checkpoint: str,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cuda:0",
    seed: int = 42,
    cfg_overrides: Optional[Dict[str, Any]] = None,
) -> DiagnosticRuntime:
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA is required by the current GDRN/online-render training path.")
    set_seed(seed)
    cfg = _prepare_cfg(config_file, batch_size, num_workers, device, cfg_overrides)

    model_module = importlib.import_module(
        f"core.gdrn_modeling.models.{cfg.MODEL.POSE_NET.NAME}"
    )
    model, _optimizer = model_module.build_model_optimizer(cfg, is_test=True)
    MyCheckpointer(model, save_dir=str(Path(checkpoint).parent), prefix_to_remove="_module.").resume_or_load(
        checkpoint, resume=False
    )
    model.eval()

    loader = build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN)
    renderer = None
    if cfg.MODEL.POSE_NET.XYZ_ONLINE and geometry_supervision_enabled(cfg):
        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        data_ref = ref.__dict__[meta.ref_key]
        gpu_id = torch.device(device).index or 0
        renderer = get_renderer(cfg, data_ref, obj_names=meta.objs, gpu_id=gpu_id)

    return DiagnosticRuntime(
        cfg=cfg,
        model=model,
        loader=loader,
        renderer=renderer,
        device=torch.device(device),
        checkpoint=checkpoint,
    )


def prepare_diagnostic_batch(runtime: DiagnosticRuntime, raw_data: List[Dict[str, Any]]) -> DiagnosticBatch:
    batch = batch_data(
        runtime.cfg,
        raw_data,
        renderer=runtime.renderer,
        device=str(runtime.device),
        phase="train",
    )
    with torch.no_grad():
        out, call = capture_model_pose_call(runtime.model, runtime.cfg, batch, do_loss=False)
        pred = decode_raw_pose(runtime.cfg, call.raw_rot, call.raw_t, batch, is_train=False)
    return DiagnosticBatch(
        raw_data=raw_data,
        batch=batch,
        pose_call=call.detached(),
        pred_rot=pred.rot.detach(),
        pred_trans=pred.trans.detach(),
    )


def iter_diagnostic_batches(runtime: DiagnosticRuntime, max_batches: int) -> Iterator[DiagnosticBatch]:
    it = iter(runtime.loader)
    for _ in range(int(max_batches)):
        raw = next(it)
        yield prepare_diagnostic_batch(runtime, raw)
