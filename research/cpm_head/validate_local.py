#!/usr/bin/env python3
"""Validate the CPM one-epoch local training and checkpoint chain."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import torch
from mmcv import Config

from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from research.cpm_head.preflight import run_full_forward


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/gdrn/lmo_pbr/convnext_cpm_head_local_lmo.py"
DEFAULT_OFFICIAL = PROJECT_ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth"
DEFAULT_RUN_DIR = PROJECT_ROOT / "output/cpm_head/local_integration"


def model_state(checkpoint: dict[str, object]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("checkpoint model state is not a mapping")
    return {
        key.removeprefix("_module."): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }


def metric_value(value: object) -> float:
    if isinstance(value, list):
        value = value[0]
    return float(value)


def load_metrics(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_path = run_dir / "model_0002047.pth"
    metrics_path = run_dir / "metrics.json"
    logs = sorted(run_dir.glob("log_*.txt"))
    if len(logs) != 1:
        raise RuntimeError(f"expected one training log, found {len(logs)}")

    official_checkpoint = torch.load(args.official.resolve(), map_location="cpu")
    final_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    official = model_state(official_checkpoint)
    final = model_state(final_checkpoint)
    official_shared = {
        key: value for key, value in official.items() if not key.startswith("pnp_net.")
    }
    final_shared = {
        key: value for key, value in final.items() if not key.startswith("pnp_net.")
    }
    shared_keys_match = official_shared.keys() == final_shared.keys()
    changed_shared = sorted(
        key
        for key in official_shared.keys() & final_shared.keys()
        if not torch.equal(official_shared[key], final_shared[key])
    )

    cfg = Config.fromfile(str(args.config.resolve()))
    cfg.MODEL.DEVICE = args.device
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    model, optimizer = build_model_optimizer(cfg, is_test=True)
    if optimizer is not None:
        raise RuntimeError("test-mode model unexpectedly created an optimizer")
    expected_pnp = {f"pnp_net.{key}" for key in model.pnp_net.state_dict()}
    actual_pnp = {key for key in final if key.startswith("pnp_net.")}
    incompatible = model.load_state_dict(final, strict=True)
    strict_reload = not incompatible.missing_keys and not incompatible.unexpected_keys
    device = torch.device(args.device)
    model.to(device)
    run_full_forward(model, device)

    metrics = load_metrics(metrics_path)
    losses = [
        metric_value(record["total_loss"])
        for record in metrics
        if "total_loss" in record
    ]
    rotation_losses = [
        metric_value(record["loss_PM_R"])
        for record in metrics
        if "loss_PM_R" in record
    ]
    quarter = max(1, len(losses) // 4)
    first_loss = statistics.median(losses[:quarter])
    last_loss = statistics.median(losses[-quarter:])
    first_rotation = statistics.median(rotation_losses[:quarter])
    last_rotation = statistics.median(rotation_losses[-quarter:])

    optimizer_state = final_checkpoint.get("optimizer", {})
    optimizer_entries = optimizer_state.get("state", {}) if isinstance(optimizer_state, dict) else {}
    optimizer_steps = []
    optimizer_moments_finite = True
    optimizer_moments_nonzero = True
    for state in optimizer_entries.values():
        optimizer_steps.append(int(state.get("step", 0)))
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = state.get(name)
            if not torch.is_tensor(tensor):
                optimizer_moments_finite = False
                optimizer_moments_nonzero = False
                continue
            optimizer_moments_finite &= bool(torch.isfinite(tensor).all())
            optimizer_moments_nonzero &= bool(torch.count_nonzero(tensor))

    log_text = logs[0].read_text(encoding="utf-8", errors="replace")
    peak_values = [int(value) for value in re.findall(r"max_mem:\s*(\d+)M", log_text)]
    checks = {
        "checkpoint_iteration_2047": int(final_checkpoint.get("iteration", -1)) == 2047,
        "checkpoint_epoch_1": int(final_checkpoint.get("epoch", -1)) == 1,
        "metrics_finish_at_2047": bool(metrics) and int(metrics[-1]["iteration"]) == 2047,
        "all_losses_finite": bool(losses) and all(math.isfinite(value) for value in losses),
        "official_shared_keys_match": shared_keys_match,
        "official_shared_tensors_bit_identical": not changed_shared,
        "cpm_state_keys_exact": actual_pnp == expected_pnp,
        "strict_checkpoint_reload": strict_reload,
        "full_model_inference_finite": True,
        "eight_trainable_optimizer_states": len(optimizer_entries) == 8,
        "optimizer_steps_positive": bool(optimizer_steps) and min(optimizer_steps) > 0,
        "optimizer_moments_finite": optimizer_moments_finite,
        "optimizer_moments_nonzero": optimizer_moments_nonzero,
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "official_shared_tensors": len(official_shared),
        "changed_official_shared_tensors": changed_shared,
        "cpm_state_tensors": len(actual_pnp),
        "optimizer_steps": sorted(set(optimizer_steps)),
        "metrics_records": len(metrics),
        "total_loss_first_quarter_median": first_loss,
        "total_loss_last_quarter_median": last_loss,
        "rotation_loss_first_quarter_median": first_rotation,
        "rotation_loss_last_quarter_median": last_rotation,
        "peak_gpu_memory_mib": max(peak_values) if peak_values else None,
        "device": args.device,
    }
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
