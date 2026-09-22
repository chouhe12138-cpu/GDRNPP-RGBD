"""Exact artifact identity checks for experiments using the shared CAD head."""
from __future__ import annotations

import hashlib
from pathlib import Path

from core.gdrn_modeling.cad.hierarchy import load_cad_hierarchy


def require_exact_hierarchy(path, contract, *, dataset_key, object_ids):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'CAD hierarchy missing: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    expected = str(contract['SHA256'])
    if actual != expected:
        raise ValueError(f'CAD hierarchy SHA mismatch: expected {expected}, got {actual}')
    hierarchy = load_cad_hierarchy(path, expected_object_ids=object_ids, dataset_key=dataset_key)
    metadata = hierarchy.metadata
    if 'DATASET_KEY' in contract and str(contract['DATASET_KEY']) != str(dataset_key):
        raise ValueError('CAD hierarchy contract dataset key mismatch')
    checks = dict(mode=contract['MODE'], generator_version=int(contract['GENERATOR_VERSION']))
    if 'VARIANT' in contract:
        checks['variant'] = contract['VARIANT']
    if 'LAMBDA_GEO' in contract:
        checks['lambda_geo'] = float(contract['LAMBDA_GEO'])
    for field, value in checks.items():
        if metadata.get(field) != value:
            raise ValueError(f'CAD hierarchy {field} mismatch: expected {value}, got {metadata.get(field)}')
    if hierarchy.depth != int(contract['DEPTH']) or hierarchy.level_counts != tuple(contract['LEVEL_COUNTS']):
        raise ValueError(f'CAD hierarchy structure mismatch: {hierarchy.depth}, {hierarchy.level_counts}')
    return actual


def require_configured_hierarchy(cfg, context):
    """Apply cfg identity while preserving immutable formal experiment pins."""
    contract = cfg.get('CAD_HIERARCHY_CONTRACT', None)
    if not contract:
        raise ValueError('CAD_HIERARCHY_CONTRACT is required')
    digest = require_exact_hierarchy(context.hierarchy_path, contract,
                                     dataset_key=context.key, object_ids=context.object_ids)
    experiment = str(cfg.get('EXPERIMENT_ID', ''))
    if experiment == 'EXP-20260920-025-hierarchical-cad-attention':
        from research.exp025.configuration import require_hierarchy
        require_hierarchy(context.hierarchy_path, context.key)
    elif experiment == 'EXP-20260922-026-residual-aligned-sampling-ablation':
        from research.exp026.configuration import require_arm_hierarchy
        require_arm_hierarchy(cfg, context)
    return digest
