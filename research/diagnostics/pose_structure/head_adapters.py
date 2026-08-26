from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .common import CapturedPoseCall, cosine_scalar
from .model_access import call_head, linear_input_transform, temporary_scalar


def head_family(head: nn.Module) -> str:
    name = type(head).__name__
    if name == "RTDecoupledGeometryPnPNet":
        return "exp013c"
    if hasattr(head, "geometry_attention"):
        return "exp013b"
    if hasattr(head, "geometry_projection") and hasattr(head, "_encode_geometry"):
        return "exp013a"
    if hasattr(head, "pose_fc1") and hasattr(head, "coarse_grid_size"):
        return "exp012_like"
    return name


def supported_branch_variants(head: nn.Module) -> List[str]:
    out = ["normal"]
    if getattr(head, "use_region_aux", False):
        out.append("region_zero")
    if hasattr(head, "geometry_scale"):
        out.extend(["main_only", "geometry_only"])
    if hasattr(head, "attention_scale"):
        out.append("attention_zero")
    return out


def run_branch_variant(head: nn.Module, call: CapturedPoseCall, variant: str):
    if variant == "normal":
        return call_head(head, call)
    if variant == "region_zero":
        if call.region is None:
            raise RuntimeError("region_zero requires Region input")
        return call_head(head, call, region=torch.zeros_like(call.region))
    if variant == "main_only":
        if not hasattr(head, "geometry_scale"):
            raise RuntimeError("main_only is unsupported by this head")
        with temporary_scalar(head.geometry_scale, 0.0):
            return call_head(head, call)
    if variant == "attention_zero":
        if not hasattr(head, "attention_scale"):
            raise RuntimeError("attention_zero is unsupported by this head")
        with temporary_scalar(head.attention_scale, 0.0):
            return call_head(head, call)
    if variant == "geometry_only":
        if not all(hasattr(head, x) for x in ("_validate_and_mask_inputs", "_encode_geometry", "geometry_scale")):
            raise RuntimeError("geometry_only is unsupported by this head")
        metric, _masked_region, support = head._validate_and_mask_inputs(
            call.coor_feat, call.region, call.extents, call.mask_attention
        )
        geometry_latent, _ = head._encode_geometry(metric, support)
        latent = head.geometry_scale.to(geometry_latent.dtype) * geometry_latent
        if not hasattr(head, "pose_rotation") or not hasattr(head, "pose_translation"):
            raise RuntimeError("geometry_only requires shared EXP012/A/B output heads")
        return head.pose_rotation(latent), head.pose_translation(latent)
    raise ValueError(f"Unknown branch variant: {variant}")


def _attention_entropy(head: nn.Module) -> Optional[float]:
    if not hasattr(head, "geometry_attention"):
        return None
    weights = getattr(head.geometry_attention, "last_weights", None)
    if weights is None:
        return None
    w = weights.detach().float().clamp_min(1.0e-12)
    ent = -(w * w.log()).sum(dim=1)
    valid = weights.detach().sum(dim=1) > 0
    if bool(valid.any()):
        return float(ent[valid].mean().cpu())
    return None


def trace_head_features(head: nn.Module, call: CapturedPoseCall) -> Dict[str, Any]:
    trace: Dict[str, Any] = {"family": head_family(head)}
    with torch.no_grad():
        if hasattr(head, "region_scale"):
            trace["region_scale"] = float(head.region_scale.detach().cpu().flatten()[0])
        if hasattr(head, "geometry_scale"):
            trace["geometry_scale"] = float(head.geometry_scale.detach().cpu().flatten()[0])
        if hasattr(head, "attention_scale"):
            trace["attention_scale"] = float(head.attention_scale.detach().cpu().flatten()[0])

        if all(hasattr(head, x) for x in ("_validate_and_mask_inputs", "_encode_main", "_encode_geometry")):
            metric, masked_region, support = head._validate_and_mask_inputs(
                call.coor_feat, call.region, call.extents, call.mask_attention
            )
            main_latent, _fine, _mid, _high = head._encode_main(metric, masked_region)
            geometry_latent, geometry_grid = head._encode_geometry(metric, support)
            trace["main_latent_rms"] = float(main_latent.float().pow(2).mean().sqrt().cpu())
            trace["geometry_latent_rms"] = float(geometry_latent.float().pow(2).mean().sqrt().cpu())
            trace["main_geometry_cosine"] = cosine_scalar(main_latent, geometry_latent)
            trace["geometry_grid_rms"] = float(geometry_grid.float().pow(2).mean().sqrt().cpu())
            if hasattr(head, "geometry_scale"):
                scaled = head.geometry_scale.to(geometry_latent.dtype) * geometry_latent
                trace["scaled_geometry_latent_rms"] = float(scaled.float().pow(2).mean().sqrt().cpu())
        trace["attention_entropy"] = _attention_entropy(head)
    return trace


def _fixed_spatial_permutation(size: int, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randperm(size, generator=g).to(device=device)


def _permute_grid_segment(x: torch.Tensor, start: int, channels: int, grid: int, seed: int) -> torch.Tensor:
    end = start + channels * grid * grid
    if end > x.shape[1]:
        raise RuntimeError(f"Descriptor too short: need {end}, got {x.shape[1]}")
    out = x.clone()
    seg = out[:, start:end].reshape(out.shape[0], channels, grid * grid)
    perm = _fixed_spatial_permutation(grid * grid, seg.device, seed)
    out[:, start:end] = seg[:, :, perm].reshape(out.shape[0], -1)
    return out


@contextlib.contextmanager
def spatial_intervention(head: nn.Module, variant: str, seed: int = 1234):
    """Shuffle spatial cell order while preserving all per-channel values."""
    stack = contextlib.ExitStack()
    try:
        if variant == "normal":
            yield
            return

        family = head_family(head)
        want_main = variant in ("main_grid_shuffle", "both_grid_shuffle")
        want_geo = variant in ("geometry_grid_shuffle", "both_grid_shuffle")

        if family in ("exp012_like", "exp013a", "exp013b"):
            if want_main:
                base_c = head.input_projection.conv.out_channels
                mid_c = head.downsample_mid.conv.out_channels
                high_c = head.downsample_high.conv.out_channels
                grid = int(head.coarse_grid_size)
                start = base_c + mid_c
                stack.enter_context(
                    linear_input_transform(
                        head.pose_fc1,
                        lambda x, s=start, c=high_c, g=grid: _permute_grid_segment(x, s, c, g, seed),
                    )
                )
            if want_geo:
                if not hasattr(head, "geometry_projection"):
                    raise RuntimeError("geometry_grid_shuffle requires EXP013 A/B geometry projection")
                geo_c = head.geometry_downsample_high.conv.out_channels
                geo_g = int(head.geometry_grid_size)
                stack.enter_context(
                    linear_input_transform(
                        head.geometry_projection,
                        lambda x, c=geo_c, g=geo_g: _permute_grid_segment(x, 0, c, g, seed + 1),
                    )
                )
        elif family == "exp013c":
            main_c = head.downsample_high.conv.out_channels
            main_g = int(head.coarse_grid_size)
            geo_c = head.geometry_downsample_high.conv.out_channels
            geo_g = int(head.geometry_grid_size)
            main_len = main_c * main_g * main_g
            if want_main or want_geo:
                def transform(x):
                    y = x
                    if want_main:
                        y = _permute_grid_segment(y, 0, main_c, main_g, seed)
                    if want_geo:
                        y = _permute_grid_segment(y, main_len, geo_c, geo_g, seed + 1)
                    return y
                stack.enter_context(linear_input_transform(head.rotation_fc1, transform))
        else:
            raise RuntimeError(f"Spatial diagnostics unsupported for {family}")

        yield
    finally:
        stack.close()
