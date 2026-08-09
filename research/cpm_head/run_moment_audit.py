#!/usr/bin/env python3
"""Run a frozen aggregate-only PBR audit of CPM moments and soft support."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from mmcv import Config

from core.gdrn_modeling.datasets.data_loader import build_gdrn_train_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from core.gdrn_modeling.models.heads.cpm_pnp_net import compute_effective_support_qc
from research.cpm_head.preflight import (
    DEFAULT_CONFIG,
    DEFAULT_WEIGHTS,
    EXPECTED_WEIGHT_SHA256,
    checkpoint_model_state,
    load_official_shared_state,
    sha256,
    validate_config,
)
from research.cpm_head.qc_utils import (
    COVERAGE_EDGES,
    EFFECTIVE_SAMPLE_EDGES,
    MOMENT_GROUP_SLICES,
    bin_indices,
    derive_moment_scales,
    moment_group_summaries,
    scalar_summary,
)


class CapturedPnPInput(RuntimeError):
    """Stop a full forward immediately after frozen geometry outputs are ready."""


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def stack_batch(data: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "roi_img": torch.stack([item["roi_img"] for item in data]).to(device),
        "roi_cls": torch.as_tensor(
            [item["roi_cls"] for item in data], dtype=torch.long, device=device
        ),
        "roi_coord_2d": torch.stack([item["roi_coord_2d"] for item in data]).to(device),
        "roi_extent": torch.stack([item["roi_extent"] for item in data]).to(device),
    }


def scale_descriptors(raw: np.ndarray, scales: dict[str, float]) -> np.ndarray:
    scaled = np.asarray(raw, dtype=np.float32).copy()
    for group, group_slice in MOMENT_GROUP_SLICES.items():
        scaled[..., group_slice] /= float(scales[group])
    return scaled


def grouped_summary(
    raw: np.ndarray,
    valid: np.ndarray,
    object_ids: np.ndarray,
    object_names: tuple[str, ...],
    total_mass: np.ndarray,
) -> dict[str, object]:
    quartile_edges = np.quantile(total_mass, [0.0, 0.25, 0.5, 0.75, 1.0])
    # Repeated quantiles are harmless; digitize still gives deterministic bins.
    quartiles = np.clip(np.digitize(total_mass, quartile_edges[1:-1], right=True), 0, 3)
    return {
        "all_regions": moment_group_summaries(raw, np.ones_like(valid, dtype=bool)),
        "valid_regions": moment_group_summaries(raw, valid),
        "empty_regions": moment_group_summaries(raw, ~valid),
        "per_object": {
            name: {
                "samples": int(np.sum(object_ids == index)),
                "moments": moment_group_summaries(raw, valid & (object_ids[:, None] == index)),
            }
            for index, name in enumerate(object_names)
        },
        "support_quartiles": {
            f"q{index + 1}": {
                "samples": int(np.sum(quartiles == index)),
                "moments": moment_group_summaries(raw, valid & (quartiles[:, None] == index)),
            }
            for index in range(4)
        },
        "support_quartile_edges": [float(value) for value in quartile_edges],
    }


def joint_support_summary(
    raw: np.ndarray,
    coverage: np.ndarray,
    effective_sample_size: np.ndarray,
    max_weight: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    coverage_bins = bin_indices(coverage, COVERAGE_EDGES)
    effective_bins = bin_indices(effective_sample_size, EFFECTIVE_SAMPLE_EDGES)
    cells: dict[str, object] = {}
    for coverage_index in range(len(COVERAGE_EDGES) - 1):
        for effective_index in range(len(EFFECTIVE_SAMPLE_EDGES) - 1):
            selector = (
                valid
                & (coverage_bins == coverage_index)
                & (effective_bins == effective_index)
            )
            key = f"coverage_{coverage_index}__n_eff_{effective_index}"
            cells[key] = {
                "coverage_interval": [
                    COVERAGE_EDGES[coverage_index],
                    COVERAGE_EDGES[coverage_index + 1],
                ],
                "n_eff_interval": [
                    EFFECTIVE_SAMPLE_EDGES[effective_index],
                    EFFECTIVE_SAMPLE_EDGES[effective_index + 1],
                ],
                "regions": int(selector.sum()),
                "region_fraction": float(selector.sum() / max(int(valid.sum()), 1)),
                "coverage": scalar_summary(coverage[selector]),
                "n_eff": scalar_summary(effective_sample_size[selector]),
                "max_normalized_weight": scalar_summary(max_weight[selector]),
                "moments": moment_group_summaries(raw, selector),
            }
    low_concentrated = valid & (coverage <= 1e-3) & (effective_sample_size < 2.0)
    low_diffuse = valid & (coverage <= 1e-3) & (effective_sample_size >= 32.0)
    return {
        "coverage_edges": list(COVERAGE_EDGES),
        "effective_sample_size_edges": list(EFFECTIVE_SAMPLE_EDGES),
        "cells": cells,
        "risk_summary": {
            "valid_regions": int(valid.sum()),
            "low_coverage_concentrated_regions": int(low_concentrated.sum()),
            "low_coverage_concentrated_fraction": float(
                low_concentrated.sum() / max(int(valid.sum()), 1)
            ),
            "low_coverage_diffuse_regions": int(low_diffuse.sum()),
            "low_coverage_diffuse_fraction": float(
                low_diffuse.sum() / max(int(valid.sum()), 1)
            ),
            "low_coverage_concentrated_moments": moment_group_summaries(
                raw, low_concentrated
            ),
            "low_coverage_diffuse_moments": moment_group_summaries(raw, low_diffuse),
        },
    }


def main() -> int:
    args = parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA audit requested but CUDA is unavailable")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    weights = args.weights.resolve()
    weight_hash = sha256(weights)
    if weight_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"Unexpected official checkpoint hash: {weight_hash}")
    cfg = Config.fromfile(str(args.config.resolve()))
    validate_config(cfg)
    cfg.MODEL.DEVICE = args.device
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATASETS.TRAIN = ("lmo_pbr_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    register_datasets_in_cfg(cfg)
    metadata = MetadataCatalog.get("lmo_pbr_train")
    object_names = tuple(metadata.objs)

    model, optimizer = build_model_optimizer(cfg, is_test=True)
    if optimizer is not None:
        raise RuntimeError("Frozen audit unexpectedly created an optimizer")
    migration = load_official_shared_state(model, checkpoint_model_state(weights))
    model.eval()
    loader = build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN)
    iterator = iter(loader)
    captured: dict[str, torch.Tensor] = {}

    def capture_input(_module, hook_inputs, hook_kwargs):
        captured["coor"] = hook_inputs[0]
        captured["region"] = hook_kwargs["region"]
        captured["extents"] = hook_kwargs["extents"]
        captured["mask"] = hook_kwargs["mask_attention"]
        raise CapturedPnPInput

    handle = model.pnp_net.register_forward_pre_hook(capture_input, with_kwargs=True)
    descriptors: list[np.ndarray] = []
    coverages: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    effective_sizes: list[np.ndarray] = []
    max_weights: list[np.ndarray] = []
    valid_regions: list[np.ndarray] = []
    object_ids: list[np.ndarray] = []
    total_masses: list[np.ndarray] = []
    processed = 0
    try:
        with torch.no_grad():
            while processed < args.samples:
                data = next(iterator)
                batch = stack_batch(data, torch.device(args.device))
                captured.clear()
                try:
                    model(
                        batch["roi_img"],
                        roi_classes=batch["roi_cls"],
                        roi_coord_2d=batch["roi_coord_2d"],
                        roi_extents=batch["roi_extent"],
                        do_loss=False,
                    )
                except CapturedPnPInput:
                    pass
                if set(captured) != {"coor", "region", "extents", "mask"}:
                    raise RuntimeError("Failed to capture effective CPM inputs")
                encoding = model.pnp_net.encode_moments(
                    captured["coor"],
                    region=captured["region"],
                    extents=captured["extents"],
                    mask_attention=captured["mask"],
                )
                qc = compute_effective_support_qc(encoding.weighting)
                remaining = args.samples - processed
                take = min(batch["roi_cls"].shape[0], remaining)
                descriptors.append(encoding.raw_descriptor[:take].float().cpu().numpy())
                coverages.append(encoding.weighting.coverage[:take].float().cpu().numpy())
                masses.append(encoding.weighting.region_mass[:take].float().cpu().numpy())
                effective_sizes.append(qc.effective_sample_size[:take].cpu().numpy())
                max_weights.append(qc.max_normalized_weight[:take].cpu().numpy())
                valid_regions.append(encoding.weighting.valid[:take].cpu().numpy())
                object_ids.append(batch["roi_cls"][:take].cpu().numpy())
                total_masses.append(
                    encoding.weighting.total_effective_mass[:take, 0].float().cpu().numpy()
                )
                processed += take
                if processed % 1024 == 0 or processed == args.samples:
                    print(json.dumps({"processed": processed, "target": args.samples}), flush=True)
    finally:
        handle.remove()

    raw = np.concatenate(descriptors, axis=0)
    coverage = np.concatenate(coverages, axis=0)
    mass = np.concatenate(masses, axis=0)
    effective_sample_size = np.concatenate(effective_sizes, axis=0)
    max_weight = np.concatenate(max_weights, axis=0)
    valid = np.concatenate(valid_regions, axis=0).astype(bool)
    object_id = np.concatenate(object_ids, axis=0)
    total_mass = np.concatenate(total_masses, axis=0)
    if raw.shape != (args.samples, 64, 21):
        raise RuntimeError(f"Unexpected aggregate descriptor shape: {raw.shape}")

    scaling = derive_moment_scales(raw, valid)
    if scaling["status"] != "PASS":
        scaled_summary = {"status": "BLOCKED", "reason": "raw_scaling_failed"}
    else:
        scaled = scale_descriptors(raw, scaling["scales"])
        scaled_summary = grouped_summary(
            scaled, valid, object_id, object_names, total_mass
        )

    moment_qc = {
        "status": "PASS" if bool(np.isfinite(raw).all()) else "BLOCKED",
        "samples": args.samples,
        "regions_per_sample": 64,
        "descriptor_dim": 21,
        "raw": grouped_summary(raw, valid, object_id, object_names, total_mass),
        "scaled": scaled_summary,
        "coverage": scalar_summary(coverage),
        "region_mass": scalar_summary(mass),
        "total_effective_mass": scalar_summary(total_mass),
        "effective_sample_size": scalar_summary(effective_sample_size),
        "max_normalized_weight": scalar_summary(max_weight),
        "valid_region_fraction": float(valid.mean()),
    }
    low_support_qc = joint_support_summary(
        raw, coverage, effective_sample_size, max_weight, valid
    )
    protocol = {
        "status": "COMPLETE",
        "config": str(args.config.resolve()),
        "weights": str(weights),
        "official_weight_sha256": weight_hash,
        "dataset": "lmo_pbr_train",
        "samples": args.samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "dtype": "float32",
        "updates": 0,
        "optimizer_created": False,
        "per_instance_moments_persisted": False,
        **migration,
    }
    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "moment_qc.json", moment_qc)
    write_json(output_dir / "low_support_qc.json", low_support_qc)
    write_json(output_dir / "moment_scaling.json", scaling)
    write_json(
        output_dir / "run_state.json",
        {
            "status": "COMPLETE"
            if moment_qc["status"] == "PASS" and scaling["status"] == "PASS"
            else "BLOCKED",
            "processed": args.samples,
        },
    )
    print(
        json.dumps(
            {
                "status": moment_qc["status"],
                "processed": args.samples,
                "valid_region_fraction": moment_qc["valid_region_fraction"],
                "scaling": scaling,
                "risk_summary": low_support_qc["risk_summary"],
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
