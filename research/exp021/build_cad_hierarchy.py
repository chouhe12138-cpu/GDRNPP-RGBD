#!/usr/bin/env python3
"""Build the deterministic external CAD hierarchy used by EXP021."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mmcv
import numpy as np
from scipy.spatial.distance import cdist

import ref
from lib.pysixd import inout, misc


ROOT = Path(__file__).resolve().parents[2]
LMO_OBJECT_IDS = (1, 5, 6, 8, 9, 10, 11, 12)
GENERATOR_VERSION = 1
DEFAULT_SAMPLES = 200_000
DEFAULT_SEED = 20260914


def _normalize(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    normalized = values / np.maximum(norm, 1e-12)
    if fallback is not None:
        invalid = norm[..., 0] <= 1e-12
        normalized[invalid] = fallback[invalid]
    return normalized.astype(np.float32)


def _sample_surface(model: dict, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(model["pts"], dtype=np.float64)
    faces = np.asarray(model["faces"], dtype=np.int64)
    triangles = points[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-15
    if not np.any(valid):
        raise ValueError("CAD mesh has no non-degenerate triangle")
    faces = faces[valid]
    triangles = triangles[valid]
    cross = cross[valid]
    probability = double_area[valid] / double_area[valid].sum()
    selected = rng.choice(len(triangles), size=count, replace=True, p=probability)
    tri = triangles[selected]
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    sampled = (
        (1.0 - u)[:, None] * tri[:, 0]
        + (u * (1.0 - v))[:, None] * tri[:, 1]
        + (u * v)[:, None] * tri[:, 2]
    )
    face_normals = _normalize(cross)[selected]
    vertex_normals = np.asarray(model.get("normals", np.zeros_like(points)), dtype=np.float64)
    vertex_normals = _normalize(vertex_normals, fallback=np.tile(np.array([0, 0, 1]), (len(points), 1)))
    return (
        np.concatenate([sampled.astype(np.float32), points.astype(np.float32)], axis=0),
        np.concatenate([face_normals, vertex_normals], axis=0),
    )


def _nearest(points: np.ndarray, anchors: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    labels = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        labels[start:stop] = cdist(points[start:stop], anchors).argmin(axis=1)
    return labels


def _fps(points: np.ndarray, count: int, first_point: np.ndarray) -> np.ndarray:
    if len(points) < count:
        raise ValueError(f"Need at least {count} surface candidates, got {len(points)}")
    selected = np.empty(count, dtype=np.int64)
    selected[0] = np.linalg.norm(points - first_point[None], axis=1).argmin()
    min_distance = np.linalg.norm(points - points[selected[0]][None], axis=1)
    for index in range(1, count):
        selected[index] = int(min_distance.argmax())
        min_distance = np.minimum(
            min_distance, np.linalg.norm(points - points[selected[index]][None], axis=1)
        )
    return selected


def _region_stats(
    points: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
    anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    region_normals = np.zeros_like(anchors, dtype=np.float32)
    radii = np.zeros(len(anchors), dtype=np.float32)
    for index in range(len(anchors)):
        chosen = labels == index
        if not np.any(chosen):
            raise ValueError(f"Empty CAD region {index}")
        mean_normal = normals[chosen].mean(axis=0, keepdims=True)
        region_normals[index] = _normalize(mean_normal)[0]
        radii[index] = np.linalg.norm(points[chosen] - anchors[index], axis=1).max() * 1.05
    return region_normals, radii


def build_hierarchy(sample_count: int = DEFAULT_SAMPLES, seed: int = DEFAULT_SEED) -> dict[str, np.ndarray]:
    model_dir = Path(ref.lm_full.model_dir)
    fps = mmcv.load(model_dir / "fps_points.pkl")
    models_info = ref.lm_full.get_models_info()
    output: dict[str, list[np.ndarray] | np.ndarray] = {
        "object_ids": np.asarray(LMO_OBJECT_IDS, dtype=np.int64),
        "extents": [],
        "diameters": [],
        "coarse_anchors": [],
        "coarse_normals": [],
        "coarse_radii": [],
        "fine_anchors": [],
        "fine_normals": [],
        "fine_radii": [],
    }
    symmetries = []
    for obj_id in LMO_OBJECT_IDS:
        model = inout.load_ply(
            str(model_dir / f"obj_{obj_id:06d}.ply"), vertex_scale=ref.lm_full.vertex_scale
        )
        points, normals = _sample_surface(
            model, sample_count, np.random.default_rng(seed + obj_id)
        )
        anchors = np.asarray(fps[str(obj_id)]["fps64_and_center"][:-1], dtype=np.float32)
        coarse_labels = _nearest(points, anchors)
        coarse_normals, coarse_radii = _region_stats(points, normals, coarse_labels, anchors)
        object_fine_anchors = np.empty((64, 64, 3), dtype=np.float32)
        object_fine_normals = np.empty_like(object_fine_anchors)
        object_fine_radii = np.empty((64, 64), dtype=np.float32)
        for parent in range(64):
            chosen = coarse_labels == parent
            parent_points = points[chosen]
            parent_normals = normals[chosen]
            indices = _fps(parent_points, 64, anchors[parent])
            leaves = parent_points[indices]
            leaf_labels = _nearest(parent_points, leaves)
            leaf_normals, leaf_radii = _region_stats(
                parent_points, parent_normals, leaf_labels, leaves
            )
            object_fine_anchors[parent] = leaves
            object_fine_normals[parent] = leaf_normals
            object_fine_radii[parent] = leaf_radii
        extent = np.ptp(np.asarray(model["pts"], dtype=np.float32), axis=0)
        output["extents"].append(extent)
        output["diameters"].append(np.float32(models_info[str(obj_id)]["diameter"] / 1000.0))
        output["coarse_anchors"].append(anchors)
        output["coarse_normals"].append(coarse_normals)
        output["coarse_radii"].append(coarse_radii)
        output["fine_anchors"].append(object_fine_anchors)
        output["fine_normals"].append(object_fine_normals)
        output["fine_radii"].append(object_fine_radii)
        transformations = misc.get_symmetry_transformations(
            models_info[str(obj_id)], max_sym_disc_step=0.01
        )
        matrices = []
        for transformation in transformations:
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, :3] = transformation["R"]
            matrix[:3, 3] = np.asarray(transformation["t"]).reshape(3) / 1000.0
            matrices.append(matrix)
        symmetries.append(np.stack(matrices))
    for name in tuple(output):
        if isinstance(output[name], list):
            output[name] = np.stack(output[name]).astype(np.float32)
    max_symmetries = max(len(item) for item in symmetries)
    symmetry_transforms = np.tile(np.eye(4, dtype=np.float32), (len(symmetries), max_symmetries, 1, 1))
    symmetry_counts = np.empty(len(symmetries), dtype=np.int64)
    for index, item in enumerate(symmetries):
        symmetry_transforms[index, : len(item)] = item
        symmetry_counts[index] = len(item)
    output["symmetry_transforms"] = symmetry_transforms
    output["symmetry_counts"] = symmetry_counts
    output["generator_version"] = np.asarray(GENERATOR_VERSION, dtype=np.int64)
    output["sample_count"] = np.asarray(sample_count, dtype=np.int64)
    output["seed"] = np.asarray(seed, dtype=np.int64)
    return output  # type: ignore[return-value]


def save_hierarchy(output: Path, arrays: dict[str, np.ndarray]) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def validate_hierarchy(arrays: dict[str, np.ndarray]) -> dict:
    fine_radii = arrays["fine_radii"]
    summary = {
        "status": "PASS",
        "generator_version": int(arrays["generator_version"]),
        "sample_count": int(arrays["sample_count"]),
        "seed": int(arrays["seed"]),
        "object_ids": arrays["object_ids"].tolist(),
        "coarse_shape": list(arrays["coarse_anchors"].shape),
        "fine_shape": list(arrays["fine_anchors"].shape),
        "min_leaf_radius_m": float(fine_radii.min()),
        "max_leaf_radius_m": float(fine_radii.max()),
        "symmetry_counts": arrays["symmetry_counts"].tolist(),
    }
    if summary["coarse_shape"] != [8, 64, 3] or summary["fine_shape"] != [8, 64, 64, 3]:
        raise ValueError(summary)
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Non-finite values in hierarchy array {key}")
    if np.any(fine_radii <= 0):
        raise ValueError("All EXP021 leaf radii must be positive")
    return summary


def main() -> int:
    cache_root = Path(os.environ.get("GDRN_DATASET_CACHE_DIR", ROOT / ".local" / "dataset_cache"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=cache_root / "exp021" / "hierarchy_v1.npz")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    arrays = build_hierarchy(args.sample_count, args.seed)
    summary = validate_hierarchy(arrays)
    save_hierarchy(args.output, arrays)
    summary["output"] = str(args.output.expanduser().resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
