"""Build deterministic 8^4 CAD trees without storing mesh data in Git."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import ref
from lib.pysixd import inout
from research.exp021.build_cad_hierarchy import (
    DEFAULT_SAMPLES,
    _fps,
    _nearest,
    _normalize,
    _region_stats,
    _sample_surface,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_CACHE = Path(os.environ.get("GDRN_DATASET_CACHE_DIR", ROOT / ".local/dataset_cache"))
SOURCE_DEFAULT = DATASET_CACHE / "exp021/hierarchy_v1.npz"
CACHE_DEFAULT = DATASET_CACHE / "exp022"
OBJECT_IDS = (1, 5, 6, 8, 9, 10, 11, 12)
GENERATOR_VERSION = 1
INDEPENDENT_GENERATOR_VERSION = 2
SEED = 20260916


def _balanced_groups(anchors: np.ndarray) -> np.ndarray:
    """Partition 64 child anchors into eight spatially local, equal-size groups."""
    centers = anchors[_fps(anchors, 8, anchors.mean(axis=0))]
    cost = ((anchors[:, None] - np.repeat(centers, 8, axis=0)[None]) ** 2).sum(-1)
    rows, slots = linear_sum_assignment(cost)
    groups = np.empty((8, 8), dtype=np.int64)
    groups[slots // 8, slots % 8] = rows
    return groups


def _parent_stats(child_xyz: np.ndarray, child_normal: np.ndarray, child_radius: np.ndarray):
    xyz = child_xyz.mean(axis=-2)
    normal = _normalize(child_normal.mean(axis=-2))
    radius = (np.linalg.norm(child_xyz - xyz[..., None, :], axis=-1) + child_radius).max(-1) * 1.05
    return xyz.astype(np.float32), normal.astype(np.float32), radius.astype(np.float32)


def _reused_object(src: dict, index: int):
    coarse = src["coarse_anchors"][index]
    coarse_normals = src["coarse_normals"][index]
    coarse_radii = src["coarse_radii"][index]
    leaves = src["fine_anchors"][index]
    leaf_normals = src["fine_normals"][index]
    leaf_radii = src["fine_radii"][index]
    coarse_groups = _balanced_groups(coarse)
    levels = [None] * 4
    source_leaf_indices = np.empty((8, 8, 8, 8), dtype=np.int64)
    level_2 = coarse[coarse_groups]
    normal_2 = coarse_normals[coarse_groups]
    radius_2 = coarse_radii[coarse_groups]
    level_1, normal_1, radius_1 = _parent_stats(level_2, normal_2, radius_2)
    level_3 = np.empty((8, 8, 8, 3), np.float32)
    normal_3 = np.empty_like(level_3)
    radius_3 = np.empty((8, 8, 8), np.float32)
    level_4 = np.empty((8, 8, 8, 8, 3), np.float32)
    normal_4 = np.empty_like(level_4)
    radius_4 = np.empty((8, 8, 8, 8), np.float32)
    for i in range(8):
        for j in range(8):
            old_parent = coarse_groups[i, j]
            groups = _balanced_groups(leaves[old_parent])
            level_4[i, j] = leaves[old_parent, groups]
            normal_4[i, j] = leaf_normals[old_parent, groups]
            radius_4[i, j] = leaf_radii[old_parent, groups]
            source_leaf_indices[i, j] = old_parent * 64 + groups
            level_3[i, j], normal_3[i, j], radius_3[i, j] = _parent_stats(
                level_4[i, j], normal_4[i, j], radius_4[i, j]
            )
    levels[:] = [
        (level_1, normal_1, radius_1),
        (level_2.reshape(64, 3), normal_2.reshape(64, 3), radius_2.reshape(64)),
        (level_3.reshape(512, 3), normal_3.reshape(512, 3), radius_3.reshape(512)),
        (level_4.reshape(4096, 3), normal_4.reshape(4096, 3), radius_4.reshape(4096)),
    ]
    return levels, source_leaf_indices.reshape(4096)


def _independent_object(obj_id: int, seed: int, sample_count: int):
    if sample_count < 4096:
        raise ValueError("Independent 8^4 tree needs at least 4096 surface samples")
    model = inout.load_ply(
        str(Path(ref.lm_full.model_dir) / f"obj_{obj_id:06d}.ply"),
        vertex_scale=ref.lm_full.vertex_scale,
    )
    points, normals = _sample_surface(model, sample_count, np.random.default_rng(seed + obj_id))
    groups = [(points, normals)]
    levels = []
    for depth in range(4):
        anchors_out, normals_out, radii_out, children = [], [], [], []
        for parent_points, parent_normals in groups:
            anchors = parent_points[_fps(parent_points, 8, parent_points.mean(axis=0))]
            labels = _nearest(parent_points, anchors)
            if depth < 3:
                # Reserve enough samples for every descendant before recursing.
                minimum = 8 ** (3 - depth)
                counts = np.bincount(labels, minlength=8)
                distances = ((parent_points[:, None] - anchors[None]) ** 2).sum(-1)
                for child in range(8):
                    deficit = minimum - counts[child]
                    if deficit <= 0:
                        continue
                    donors = np.flatnonzero(counts[labels] > minimum)
                    ranked = donors[np.argsort(distances[donors, child] - distances[donors, labels[donors]])]
                    for point in ranked:
                        if counts[child] >= minimum:
                            break
                        if counts[labels[point]] <= minimum:
                            continue
                        counts[labels[point]] -= 1
                        labels[point] = child
                        counts[child] += 1
                    if counts[child] < minimum:
                        raise ValueError(f"Cannot balance independent CAD tree at level {depth + 1}")
            node_normals, node_radii = _region_stats(parent_points, parent_normals, labels, anchors)
            # Singleton surface cells otherwise have a zero-radius residual domain.
            node_radii = np.maximum(node_radii, 1e-5)
            anchors_out.append(anchors)
            normals_out.append(node_normals)
            radii_out.append(node_radii)
            if depth < 3:
                for child in range(8):
                    selected = labels == child
                    children.append((parent_points[selected], parent_normals[selected]))
        levels.append((
            np.concatenate(anchors_out).astype(np.float32),
            np.concatenate(normals_out).astype(np.float32),
            np.concatenate(radii_out).astype(np.float32),
        ))
        groups = children
    return levels, np.full(4096, -1, dtype=np.int64)


def build(mode: str, source: Path = SOURCE_DEFAULT, seed: int = SEED, sample_count: int = DEFAULT_SAMPLES):
    if mode not in {"reused", "independent"}:
        raise ValueError(f"Unknown EXP022 hierarchy mode: {mode}")
    with np.load(source, allow_pickle=False) as original:
        src = {name: np.asarray(original[name]).copy() for name in original.files}
    if tuple(src["object_ids"].tolist()) != OBJECT_IDS:
        raise ValueError("EXP021 object order does not match EXP022")
    output = {name: src[name] for name in ("object_ids", "extents", "diameters", "symmetry_counts", "symmetry_transforms")}
    per_level = [[], [], [], []]
    source_indices = []
    for index, obj_id in enumerate(OBJECT_IDS):
        levels, indices = (
            _reused_object(src, index) if mode == "reused"
            else _independent_object(obj_id, seed, sample_count)
        )
        for depth, level in enumerate(levels):
            per_level[depth].append(level)
        source_indices.append(indices)
    for depth, objects in enumerate(per_level, start=1):
        for field, component in (("anchors", 0), ("normals", 1), ("radii", 2)):
            output[f"level{depth}_{field}"] = np.stack([obj[component] for obj in objects])
    output["source_leaf_indices"] = np.stack(source_indices)
    output["generator_version"] = np.asarray(
        GENERATOR_VERSION if mode == "reused" else INDEPENDENT_GENERATOR_VERSION, dtype=np.int64
    )
    output["radius_floor_m"] = np.asarray(0.0 if mode == "reused" else 1e-5, dtype=np.float32)
    output["seed"] = np.asarray(seed, dtype=np.int64)
    output["sample_count"] = np.asarray(sample_count, dtype=np.int64)
    output["mode"] = np.asarray(mode)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reused", "independent"), required=True)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args()
    filename = "reused_v1.npz" if args.mode == "reused" else "independent_v2.npz"
    output = args.output or CACHE_DEFAULT / filename
    if output.exists():
        raise FileExistsError(output)
    hierarchy = build(args.mode, args.source, sample_count=args.sample_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **hierarchy)
    print(f"EXP022 hierarchy {args.mode}: {output}")


if __name__ == "__main__":
    main()
