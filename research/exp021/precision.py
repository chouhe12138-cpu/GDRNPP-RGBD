"""Precision helpers shared by EXP021 CUDA diagnostics."""

from __future__ import annotations

from contextlib import AbstractContextManager

import torch
from mmcv import Config


PRECISION_CHOICES = ("config", "fp32", "amp-fp16")


def resolve_precision(cfg: Config, requested: str) -> str:
    if requested not in PRECISION_CHOICES:
        raise ValueError(f"Unsupported precision: {requested}")
    if requested == "config":
        enabled = bool(cfg.SOLVER.get("AMP", {}).get("ENABLED", False))
        return "amp-fp16" if enabled else "fp32"
    return requested


def autocast_context(
    precision: str, device: torch.device
) -> AbstractContextManager:
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=precision == "amp-fp16",
    )


def grad_scaler(precision: str) -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=precision == "amp-fp16")


def gradients_are_finite(parameters) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )
