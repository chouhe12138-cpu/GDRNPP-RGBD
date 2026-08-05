"""Pure utilities used by the Patch-PnP information-flow diagnostic."""

from __future__ import annotations

import hashlib
from typing import Dict, Mapping

import numpy as np
import torch


CONDITIONS = (
    "baseline",
    "gt_x",
    "gt_y",
    "gt_z",
    "gt_xy",
    "gt_xyz",
    "permute_xyz",
    "permute_roi_2d",
    "permute_region",
    "mean_region",
    "gt_xyz_boundary",
    "gt_xyz_interior_matched",
    "gt_xyz_high_error",
    "gt_xyz_random_matched",
)


def erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary 3x3 erosion implemented without an image-processing dependency."""

    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(iterations)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighborhoods = [
            padded[y : y + result.shape[0], x : x + result.shape[1]]
            for y in range(3)
            for x in range(3)
        ]
        result = np.logical_and.reduce(neighborhoods)
    return result


def matched_spatial_masks(
    xyz: np.ndarray,
    gt_xyz: np.ndarray,
    support: np.ndarray,
    seed: int,
    boundary_width: int = 2,
    error_fraction: float = 0.25,
) -> Dict[str, np.ndarray]:
    """Build deterministic, cardinality-matched spatial correction masks."""

    mask = np.asarray(support, dtype=bool)
    predicted = np.asarray(xyz, dtype=np.float64)
    ground_truth = np.asarray(gt_xyz, dtype=np.float64)
    if predicted.shape != ground_truth.shape or predicted.shape[:2] != mask.shape:
        raise ValueError("XYZ and support shapes differ")
    rng = np.random.default_rng(int(seed))

    interior_candidates = erode_mask(mask, iterations=boundary_width)
    boundary_candidates = mask & ~interior_candidates
    boundary_indices = np.flatnonzero(boundary_candidates)
    interior_indices = np.flatnonzero(interior_candidates)
    boundary_count = min(len(boundary_indices), len(interior_indices))
    if len(boundary_indices) > boundary_count:
        boundary_indices = np.sort(rng.choice(boundary_indices, boundary_count, replace=False))
    if len(interior_indices) > boundary_count:
        interior_indices = np.sort(rng.choice(interior_indices, boundary_count, replace=False))

    support_indices = np.flatnonzero(mask)
    high_count = int(np.ceil(len(support_indices) * float(error_fraction)))
    if high_count:
        errors = np.linalg.norm(predicted - ground_truth, axis=2).reshape(-1)
        order = support_indices[np.argsort(errors[support_indices], kind="stable")]
        high_indices = np.sort(order[-high_count:])
        random_indices = np.sort(rng.choice(support_indices, high_count, replace=False))
    else:
        high_indices = np.empty(0, dtype=np.int64)
        random_indices = np.empty(0, dtype=np.int64)

    def from_indices(indices: np.ndarray) -> np.ndarray:
        result = np.zeros(mask.size, dtype=bool)
        result[indices] = True
        return result.reshape(mask.shape)

    return {
        "gt_xyz_boundary": from_indices(boundary_indices),
        "gt_xyz_interior_matched": from_indices(interior_indices),
        "gt_xyz_high_error": from_indices(high_indices),
        "gt_xyz_random_matched": from_indices(random_indices),
    }


def apply_intervention(
    xyz: np.ndarray,
    gt_xyz: np.ndarray,
    roi_2d: np.ndarray,
    region: np.ndarray,
    support: np.ndarray,
    condition: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return copies with one controlled intervention on a fixed support."""

    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    xyz_out = np.asarray(xyz).copy()
    gt = np.asarray(gt_xyz)
    roi_out = np.asarray(roi_2d).copy()
    region_out = np.asarray(region).copy()
    mask = np.asarray(support, dtype=bool)
    if xyz_out.shape != gt.shape or xyz_out.ndim != 3 or xyz_out.shape[-1] != 3:
        raise ValueError("xyz and gt_xyz must both have shape HxWx3")
    if roi_out.shape[:2] != mask.shape or region_out.shape[:2] != mask.shape:
        raise ValueError("all inputs must share the support spatial shape")
    indices = np.flatnonzero(mask.reshape(-1))
    rng = np.random.default_rng(int(seed))

    spatial_masks = matched_spatial_masks(xyz_out, gt, mask, seed)

    if condition in {"gt_x", "gt_y", "gt_z", "gt_xy", "gt_xyz"}:
        channels = {
            "gt_x": (0,),
            "gt_y": (1,),
            "gt_z": (2,),
            "gt_xy": (0, 1),
            "gt_xyz": (0, 1, 2),
        }[condition]
        for channel in channels:
            xyz_out[..., channel][mask] = gt[..., channel][mask]
    elif condition in spatial_masks:
        spatial_mask = spatial_masks[condition]
        xyz_out[spatial_mask] = gt[spatial_mask]
    elif condition.startswith("permute_") and len(indices) > 1:
        permutation = rng.permutation(indices)
        if condition == "permute_xyz":
            flat = xyz_out.reshape(-1, 3)
        elif condition == "permute_roi_2d":
            flat = roi_out.reshape(-1, roi_out.shape[-1])
        else:
            flat = region_out.reshape(-1, region_out.shape[-1])
        source = flat.copy()
        flat[indices] = source[permutation]
    elif condition == "mean_region" and len(indices):
        flat = region_out.reshape(-1, region_out.shape[-1])
        flat[indices] = np.mean(flat[indices], axis=0, keepdims=True)
    return xyz_out, roi_out, region_out


def response_metrics(reference: torch.Tensor, changed: torch.Tensor) -> Dict[str, float]:
    """Summarize a layer response relative to the baseline activation."""

    ref = reference.detach().double().reshape(-1)
    cur = changed.detach().double().reshape(-1)
    if ref.shape != cur.shape:
        raise ValueError(f"activation shapes differ: {tuple(ref.shape)} vs {tuple(cur.shape)}")
    finite = torch.isfinite(ref) & torch.isfinite(cur)
    finite_count = int(finite.sum().item())
    if not finite_count:
        return {
            "relative_l2": float("nan"),
            "cosine_distance": float("nan"),
            "mean_absolute": float("nan"),
            "finite_count": 0,
        }
    ref = ref[finite]
    cur = cur[finite]
    delta = cur - ref
    denominator = max(float(torch.linalg.vector_norm(ref).item()), 1e-12)
    relative_l2 = float(torch.linalg.vector_norm(delta).item()) / denominator
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    cur_norm = float(torch.linalg.vector_norm(cur).item())
    cosine = (
        float(torch.dot(ref, cur).item()) / (ref_norm * cur_norm)
        if ref_norm > 1e-12 and cur_norm > 1e-12
        else (1.0 if ref_norm <= 1e-12 and cur_norm <= 1e-12 else 0.0)
    )
    return {
        "relative_l2": relative_l2,
        "cosine_distance": float(1.0 - np.clip(cosine, -1.0, 1.0)),
        "mean_absolute": float(torch.mean(torch.abs(delta)).item()),
        "finite_count": finite_count,
    }


def rotation_geodesic_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = np.asarray(rotation_a, dtype=np.float64) @ np.asarray(
        rotation_b, dtype=np.float64
    ).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()
