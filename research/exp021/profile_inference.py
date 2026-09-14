#!/usr/bin/env python3
"""Batch-1 CUDA latency and peak-memory profiler for EXP021 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import inference_context

from core.gdrn_modeling.datasets.data_loader import build_gdrn_test_loader
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data
from research.exp020.matched_pnp_eval import _load_model
from research.exp021.matched_pnp_eval import CONFIG_ROOT, DEFAULT_OFFICIAL, _configured


def _profile_forward(model, batch: dict, beam_ks: tuple[int, ...]):
    kwargs = dict(
        roi_classes=batch["roi_cls"],
        roi_cams=batch["roi_cam"],
        roi_whs=batch["roi_wh"],
        roi_centers=batch["roi_center"],
        resize_ratios=batch["resize_ratio"],
        roi_coord_2d=batch.get("roi_coord_2d"),
        roi_coord_2d_rel=batch.get("roi_coord_2d_rel"),
        roi_extents=batch["roi_extent"],
        return_dense_only=True,
        return_cad_debug=False,
    )
    if model.cad_head is not None:
        kwargs["cad_beam_ks"] = beam_ks
    with inference_context(model), torch.no_grad():
        return model(batch["roi_img"], **kwargs)


def profile(name: str, config: Path, checkpoint: Path, device: str, beam_ks: tuple[int, ...]) -> dict:
    cfg = _configured(config, checkpoint, device)
    register_datasets_in_cfg(cfg)
    model = _load_model(cfg, checkpoint, device)
    metadata = MetadataCatalog.get("lmo_bop_test")
    raw = next(
        iter(
            build_gdrn_test_loader(
                cfg, "lmo_bop_test", train_objs=metadata.objs, batch_size=1
            )
        )
    )
    batch = batch_data(cfg, raw if isinstance(raw, list) else [raw], device=device, phase="test")
    for _ in range(50):
        _profile_forward(model, batch, beam_ks)
    torch.cuda.synchronize()
    samples = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(200):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        _profile_forward(model, batch, beam_ks)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "name": name,
        "checkpoint": str(checkpoint),
        "beam_ks": list(beam_ks),
        "latency_median_ms": float(np.median(samples)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "total_parameters": sum(value.numel() for value in model.parameters()),
        "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
    }


def resource_gate(reports: list[dict]) -> dict | None:
    """Apply the preregistered C-vs-A deployment budget when C is present."""

    by_name = {item["name"]: item for item in reports}
    if "c_k4" not in by_name:
        return None
    checks = {
        "c_latency_increase_at_most_25pct": by_name["c_k4"][
            "latency_increase_vs_a"
        ]
        <= 0.25,
        "c_memory_increase_at_most_20pct": by_name["c_k4"][
            "memory_increase_vs_a"
        ]
        <= 0.20,
    }
    return {"checks": checks, "pass": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--checkpoint-c", type=Path)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EXP021 profiling requires CUDA")
    reports = [
        profile("a", CONFIG_ROOT / "a_official_eval.py", args.official_checkpoint, args.device, ()),
        profile("b_k4", CONFIG_ROOT / "b_hierarchical.py", args.checkpoint_b, args.device, (4,)),
    ]
    if args.checkpoint_c:
        reports.append(profile("c_k4", CONFIG_ROOT / "c_global.py", args.checkpoint_c, args.device, (4,)))
    by_name = {item["name"]: item for item in reports}
    for name in ("b_k4", "c_k4"):
        if name in by_name:
            by_name[name]["latency_increase_vs_a"] = by_name[name]["latency_median_ms"] / by_name["a"]["latency_median_ms"] - 1
            by_name[name]["memory_increase_vs_a"] = by_name[name]["peak_allocated_bytes"] / by_name["a"]["peak_allocated_bytes"] - 1
    text = json.dumps(
        {"profiles": reports, "resource_gate": resource_gate(reports)}, indent=2
    )
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
