"""Load complete, fixed-branch CAD trees without experiment-specific policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import torch


@dataclass(frozen=True)
class HierarchyLevel:
    anchors: torch.Tensor
    normals: torch.Tensor
    radii: torch.Tensor


@dataclass(frozen=True)
class CADHierarchy:
    object_ids: torch.Tensor
    extents: torch.Tensor
    diameters: torch.Tensor
    levels: tuple[HierarchyLevel, ...]
    symmetry_counts: torch.Tensor
    symmetry_transforms: torch.Tensor
    metadata: dict
    source_leaf_indices: torch.Tensor | None = None

    @property
    def depth(self):
        return len(self.levels)

    @property
    def branch_factor(self):
        return self.level_counts[0]

    @property
    def level_counts(self):
        return tuple(level.anchors.shape[1] for level in self.levels)

    def level(self, depth):
        if not 1 <= depth <= self.depth:
            raise ValueError(f"Invalid hierarchy depth: {depth}")
        return self.levels[depth - 1]

    def numpy_levels(self):
        """CPU float64 views/copies for geometric diagnostics; depths are one-based."""
        return {i: {field: getattr(level, field).numpy().astype(np.float64)
                    for field in ('anchors', 'normals', 'radii')}
                for i, level in enumerate(self.levels, 1)}

    def tensor_arrays(self):
        """Compatibility representation retaining stored dtypes and buffer names."""
        arrays = {name: getattr(self, name) for name in (
            'object_ids', 'extents', 'diameters', 'symmetry_counts', 'symmetry_transforms')}
        for depth, level in enumerate(self.levels, 1):
            arrays.update({f'level{depth}_{field}': getattr(level, field)
                           for field in ('anchors', 'normals', 'radii')})
        if self.source_leaf_indices is not None:
            arrays['source_leaf_indices'] = self.source_leaf_indices
        return arrays


def load_cad_hierarchy(path, *, expected_object_ids=None, dataset_key=None,
                       allow_missing_dataset=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CAD hierarchy missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]).copy() for key in archive.files}
    required = {'object_ids', 'extents', 'diameters', 'symmetry_counts', 'symmetry_transforms'}
    depths = sorted({int(match.group(1)) for key in data
                     if (match := re.fullmatch(r'level(\d+)_(anchors|normals|radii)', key))})
    if not depths or depths != list(range(1, len(depths) + 1)):
        raise ValueError('CAD hierarchy levels must be contiguous from level1')
    required.update(f'level{d}_{f}' for d in depths for f in ('anchors', 'normals', 'radii'))
    if required - data.keys():
        raise ValueError(f'CAD hierarchy missing arrays: {sorted(required - data.keys())}')
    ids = data['object_ids']
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids) or len(set(ids.tolist())) != len(ids):
        raise ValueError('CAD object IDs must be nonempty unique integers')
    if expected_object_ids is not None and tuple(ids) != tuple(expected_object_ids):
        raise ValueError('CAD object order mismatch')
    stored_dataset = data.get('dataset_key')
    if stored_dataset is not None and stored_dataset.ndim != 0:
        raise ValueError('dataset_key must be scalar')
    if dataset_key is not None and (stored_dataset is None or str(stored_dataset) != dataset_key):
        if not (stored_dataset is None and allow_missing_dataset):
            raise ValueError('CAD hierarchy dataset mismatch')
    n = len(ids)

    def check(name, shape, positive=False, integer=False):
        value = data[name]
        if value.shape != shape or value.dtype.kind not in ('iu' if integer else 'fiu'):
            raise ValueError(f'Invalid {name} shape or dtype: {value.shape}, {value.dtype}')
        if not np.isfinite(value).all() or (positive and np.any(value <= 0)):
            raise ValueError(f'Invalid finite/positive values in {name}')

    check('extents', (n, 3), positive=True)
    check('diameters', (n,), positive=True)
    check('symmetry_counts', (n,), positive=True, integer=True)
    transforms = data['symmetry_transforms']
    if transforms.ndim != 4:
        raise ValueError('symmetry_transforms must have shape [N,S,4,4]')
    check('symmetry_transforms', (n, transforms.shape[1], 4, 4))
    if np.any(data['symmetry_counts'] > transforms.shape[1]):
        raise ValueError('symmetry_counts are outside stored transform bounds')
    root = data['level1_anchors']
    if root.ndim != 3 or root.shape[1] < 2:
        raise ValueError('CAD root must have at least two children')
    branch = root.shape[1]
    counts = tuple(branch ** d for d in depths)
    for d, count in zip(depths, counts):
        for field, shape in (('anchors', (n, count, 3)), ('normals', (n, count, 3)),
                             ('radii', (n, count))):
            check(f'level{d}_{field}', shape, positive=field == 'radii')
    if 'source_leaf_indices' in data:
        check('source_leaf_indices', (n, counts[-1]), integer=True)
    metadata = {key: value.tolist() for key, value in data.items()
                if key not in required and key != 'source_leaf_indices'}
    for key, expected in (('branch_factor', branch), ('depth', len(depths)),
                          ('level_counts', list(counts))):
        if key in metadata and metadata[key] != expected:
            raise ValueError(f'CAD hierarchy metadata mismatch: {key}')
    for key in ('mode', 'builder', 'dataset_key', 'cad_ref_key'):
        if key in metadata and not isinstance(metadata[key], str):
            raise ValueError(f'{key} must be a scalar string')
    for key in ('generator_version', 'schema_version'):
        if key in metadata and (type(metadata[key]) is not int or metadata[key] < 1):
            raise ValueError(f'{key} must be a positive integer')
    tensor = lambda name: torch.from_numpy(data[name])
    return CADHierarchy(
        tensor('object_ids'), tensor('extents'), tensor('diameters'),
        tuple(HierarchyLevel(*(tensor(f'level{d}_{f}') for f in ('anchors', 'normals', 'radii')))
              for d in depths), tensor('symmetry_counts'), tensor('symmetry_transforms'), metadata,
        tensor('source_leaf_indices') if 'source_leaf_indices' in data else None)
