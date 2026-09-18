#!/usr/bin/env python3
"""Check dataset context, backbone initialization, and a CPU PCC step."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from mmcv import Config

import ref
from core.gdrn_modeling.models.GDRN_PCC import build_model_optimizer
from research.exp013.preflight import PROJECT_ROOT, checkpoint_model_state
from research.run_contract import validate_research_run_config
from research.exp022.dataset_context import resolve_dataset_context


CONFIG_ROOT = PROJECT_ROOT / "configs/gdrn/research/exp022_progressive_pcc"
# The LM-O arm of EXP022 keeps its own tree; bare `--config` names may resolve
# into either one so existing commands keep working.
LMO_CONFIG_ROOT = PROJECT_ROOT / "configs/gdrn/lmo_pbr/research/exp022_progressive_pcc"
EXPERIMENT_ID = "EXP-20260916-022-progressive-pcc"

# GDR-Net's LINEMOD protocol, mirrored by lm13_gdrn_protocol.py.  A formal run
# that drifts from these values is refused rather than silently reinterpreted.
LM13_GDRN_TRAIN = ("lm_13_train_online", "lm_imgn_13_train_1k_per_obj_online")
LM13_GDRN_INPUT = {
    "COLOR_AUG_PROB": 0.0,
    "CHANGE_BG_PROB": 0.5,
    "PBR_CHANGE_BG_PROB": 0.5,
    "DZI_TYPE": "uniform",
    "DZI_PAD_SCALE": 1.5,
    "DZI_SCALE_RATIO": 0.25,
    "DZI_SHIFT_RATIO": 0.25,
    "TRUNCATE_FG": False,
}
LM13_GDRN_SOLVER = {
    "TOTAL_EPOCHS": 160,
    "REFERENCE_BS": 24,
    "WARMUP_ITERS": 1000,
    "WARMUP_METHOD": "linear",
    "ANNEAL_METHOD": "cosine",
    "ANNEAL_POINT": 0.72,
    "TARGET_LR_FACTOR": 0.0,
}


def resolve_config_path(raw: Path) -> Path:
    """Resolve a bare config name against the EXP022 config trees."""
    if raw.is_absolute() or raw.exists():
        return raw
    for root in (CONFIG_ROOT, LMO_CONFIG_ROOT):
        candidate = root / raw
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"EXP022 config not found in {CONFIG_ROOT} or {LMO_CONFIG_ROOT}: {raw}")


def check_lm13_gdrn_protocol(cfg) -> dict:
    """Verify the LM13 GDR-Net protocol before its data or training is trusted."""
    train = tuple(str(name) for name in cfg.DATASETS.TRAIN)
    if train != LM13_GDRN_TRAIN:
        raise ValueError(f"lm13_gdrn training splits must be {LM13_GDRN_TRAIN}, got {train}")
    # solver_utils prefers WARMUP_RATIO over WARMUP_ITERS and would replace
    # ANNEAL_POINT with that ratio, so it has to stay unset
    if cfg.SOLVER.get("WARMUP_RATIO", None) is not None:
        raise ValueError("lm13_gdrn must leave WARMUP_RATIO unset")
    for name, expected in LM13_GDRN_INPUT.items():
        actual = cfg.INPUT.get(name, None)
        if actual != expected:
            raise ValueError(f"lm13_gdrn INPUT.{name} must be {expected}, got {actual}")
    for name, expected in LM13_GDRN_SOLVER.items():
        actual = cfg.SOLVER.get(name, None)
        if actual != expected:
            raise ValueError(f"lm13_gdrn SOLVER.{name} must be {expected}, got {actual}")
    if float(cfg.DATALOADER.FILTER_VISIB_THR) != 0.0:
        raise ValueError("lm13_gdrn keeps every annotated instance (FILTER_VISIB_THR=0)")
    if str(cfg.SOLVER.OPTIMIZER_CFG.type) != "Ranger":
        raise ValueError(f"lm13_gdrn baseline must be Ranger, got {cfg.SOLVER.OPTIMIZER_CFG.type}")
    if float(cfg.SOLVER.OPTIMIZER_CFG.get("lr", 0.0)) != 1e-4:
        raise ValueError("lm13_gdrn baseline learning rate must be 1e-4")
    if float(cfg.SOLVER.OPTIMIZER_CFG.get("weight_decay", 0.0)) != 0.0:
        raise ValueError("lm13_gdrn baseline must not use weight decay")
    return {
        "train_splits": list(train),
        "reference_batch_size": int(cfg.SOLVER.REFERENCE_BS),
        "total_epochs": int(cfg.SOLVER.TOTAL_EPOCHS),
    }


def check_protocol_files(cfg, context) -> dict:
    """Check the files the configured evaluation protocols need."""
    if not Path(context.bop_targets).is_file():
        raise FileNotFoundError(f"BOP targets missing: {context.bop_targets}")
    data_ref = ref.__dict__[context.data_ref_key]
    name = str(cfg.VAL.TARGETS_FILENAME)
    candidates = [Path(data_ref.dataset_root) / name,
                  PROJECT_ROOT / "lib/pysixd/tools/lm" / name,
                  PROJECT_ROOT / name]
    legacy = next((path for path in candidates if path.is_file()), None)
    if legacy is None:
        raise FileNotFoundError(f"legacy LM targets {name} not found in {[str(p) for p in candidates]}")
    detector_files = [str(entry) for entry in cfg.DATASETS.DET_FILES_TEST]
    bbox_type = str(cfg.TEST.TEST_BBOX_TYPE).lower()
    if bbox_type == "est":
        if not detector_files:
            raise ValueError("detector-bbox evaluation needs DATASETS.DET_FILES_TEST")
        for entry in detector_files:
            if not Path(entry).is_file():
                raise FileNotFoundError(f"detector bbox file missing: {entry}")
    return {"bop_targets": str(context.bop_targets), "legacy_targets": str(legacy),
            "bbox_source": bbox_type, "detector_files": detector_files}


def load_official_backbone(model, weights: Path):
    state = checkpoint_model_state(weights)
    current = model.backbone.state_dict()
    source = {name.removeprefix("backbone."): tensor for name, tensor in state.items()
              if name.startswith("backbone.")}
    if current.keys() != source.keys():
        raise RuntimeError(f"Official backbone key mismatch: missing={current.keys()-source.keys()}, "
                           f"unexpected={source.keys()-current.keys()}")
    for name in current:
        if current[name].shape != source[name].shape:
            raise RuntimeError(f"Official backbone shape mismatch: {name}")
    model.backbone.load_state_dict(source, strict=True)
    return len(source)


def verify_imagenet_backbone(model, checkpoint: Path) -> int:
    """Check that all feature tensors match the supplied Facebook ConvNeXt file."""
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    current = model.backbone.state_dict()
    converted = {}
    for name, tensor in raw.items():
        if name.startswith(("head.", "norm.")):
            continue
        name = name.replace("downsample_layers.0.0.", "stem_0.")
        name = name.replace("downsample_layers.0.1.", "stem_1.")
        name = re.sub(r"downsample_layers\.(\d+)\.(\d+)",
                      lambda match: f"stages_{match.group(1)}.downsample.{match.group(2)}", name)
        name = re.sub(r"stages\.(\d+)\.(\d+)",
                      lambda match: f"stages_{match.group(1)}.blocks.{match.group(2)}", name)
        name = name.replace("dwconv", "conv_dw").replace("pwconv", "mlp.fc")
        if name not in current:
            raise RuntimeError(f"Unmapped ImageNet tensor: {name}")
        converted[name] = tensor.reshape(current[name].shape)
    if current.keys() != converted.keys():
        raise RuntimeError(f"ImageNet backbone tensor mismatch: {current.keys()-converted.keys()}")
    if any(not torch.equal(current[name].cpu(), converted[name]) for name in current):
        raise RuntimeError("ImageNet backbone weights were not loaded exactly")
    return len(converted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LMO_CONFIG_ROOT / "train_reused.py")
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config_path = resolve_config_path(args.config)
    cfg = Config.fromfile(str(config_path))
    mode = "smoke" if config_path.stem.startswith("smoke") else (
        "prepare" if cfg.get("RESEARCH_PROTOCOL", {}).get("SCHEDULE") == "configurable" else "formal"
    )
    validate_research_run_config(cfg, mode=mode, expected_experiment_id=cfg.EXPERIMENT_ID)
    context = resolve_dataset_context(cfg)
    protocol = {}
    # The smoke config inherits TRAIN_PROTOCOL from the formal one but swaps in
    # the *_smoke splits, so the formal protocol check would reject its own
    # smoke.  Only a full-size config is held to the formal definition.
    if mode != "smoke" and str(cfg.get("TRAIN_PROTOCOL", {}).get("NAME", "")) == "lm13_gdrn":
        protocol = check_lm13_gdrn_protocol(cfg)
    files = check_protocol_files(cfg, context)
    cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    model, optimizer = build_model_optimizer(cfg)
    if cfg.MODEL.POSE_NET.BACKBONE.FREEZE:
        loaded = load_official_backbone(model, args.weights or Path(cfg.MODEL.WEIGHTS))
    else:
        loaded = verify_imagenet_backbone(
            model, Path(cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.checkpoint_path)
        )
    model.train()
    assert model.backbone.training == (not cfg.MODEL.POSE_NET.BACKBONE.FREEZE)
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    registered = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    allowed = ("pcc_head.",) if cfg.MODEL.POSE_NET.BACKBONE.FREEZE else ("pcc_head.", "backbone.")
    if not trainable or any(not name.startswith(allowed) for name in trainable):
        raise RuntimeError("Unexpected EXP022 trainable tensors")
    if not cfg.MODEL.POSE_NET.BACKBONE.FREEZE and not any(
        name.startswith("backbone.") for name in trainable
    ):
        raise RuntimeError("EXP022 full training has no trainable backbone")
    if registered != {id(parameter) for parameter in trainable.values()}:
        raise RuntimeError("EXP022 optimizer does not exactly match PCC tensors")
    image = torch.randn(1, 3, 256, 256)
    xyz = torch.full((1, 3, 64, 64), 0.5)
    mask = torch.ones(1, 1, 64, 64)
    _, losses = model(image, roi_classes=torch.tensor([0]), gt_xyz=xyz,
                      gt_mask_visib=mask, do_loss=True)
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise RuntimeError("EXP022 non-finite preflight loss")
    total.backward()
    if any(value.grad is None or not torch.isfinite(value.grad).all() for value in trainable.values()):
        raise RuntimeError("EXP022 missing/non-finite trainable gradient")
    optimizer.step()
    with torch.no_grad():
        output = model(image, roi_classes=torch.tensor([0]))
    if any(tuple(output[key].shape) != (1, 1, 64, 64) for key in
           ("coor_x", "coor_y", "coor_z", "mask")):
        raise RuntimeError("EXP022 inference output shape mismatch")
    print(json.dumps({"status": "PASS", "config": str(config_path), "formal_training": False,
                      "dataset": context.key, "num_objects": context.num_objects,
                      "object_ids": context.object_ids,
                      "hierarchy_path": str(context.hierarchy_path),
                      "protocol": protocol, "files": files,
                      "backbone_checkpoint": str(args.weights or cfg.MODEL.WEIGHTS or
                                                 cfg.MODEL.POSE_NET.BACKBONE.INIT_CFG.get("checkpoint_path", "")),
                      "backbone_tensors": loaded,
                      "trainable_parameters": sum(p.numel() for p in trainable.values()),
                      "symmetry_counts": model.pcc_head.symmetry_counts.tolist(),
                      "max_symmetry_count": int(model.pcc_head.symmetry_counts.max().item()),
                      "losses": {name: float(value.detach()) for name, value in losses.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
