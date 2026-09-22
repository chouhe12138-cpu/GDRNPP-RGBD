"""Immutable EXP026 arm-to-artifact identities."""
from __future__ import annotations

from research.cad_hierarchy.contracts import require_exact_hierarchy

EXPERIMENT_ID = 'EXP-20260922-026-residual-aligned-sampling-ablation'
ARM_ARTIFACTS = {
    'uniform_full': ('uniform_512', 0.0,
                     '35287beb3dc67b5f3cd376cc445f7727857aefa888dd11d639ab65b9193235c1'),
    'adaptive_l1_full': ('adaptive_512_l1', 1.0,
                         '7ac75daa756ed7ae59463614737ccc46cd746173452cd1943a4a024d0817c631'),
}


def require_arm_hierarchy(cfg, context):
    if str(cfg.EXPERIMENT_ID) != EXPERIMENT_ID:
        raise ValueError(f'EXP026 experiment ID mismatch: {cfg.EXPERIMENT_ID}')
    arm = str(cfg.EXP026_ARM)
    if arm not in ARM_ARTIFACTS:
        raise ValueError(f'Unknown EXP026 arm: {arm}')
    variant, lambda_geo, sha = ARM_ARTIFACTS[arm]
    contract = cfg.CAD_HIERARCHY_CONTRACT
    if (str(contract.VARIANT), float(contract.LAMBDA_GEO), str(contract.SHA256)) != (variant, lambda_geo, sha):
        raise ValueError(f'EXP026 {arm} artifact identity mismatch')
    if (str(contract.MODE), int(contract.GENERATOR_VERSION), int(contract.DEPTH),
            tuple(contract.LEVEL_COUNTS)) != ('geometry_adaptive', 1, 3, (8, 64, 512)):
        raise ValueError(f'EXP026 {arm} artifact structure mismatch')
    if tuple(context.object_ids) != (1, 5, 6, 8, 9, 10, 11, 12) or context.key != 'lmo':
        raise ValueError('EXP026 requires LM-O eight-object ordering')
    return require_exact_hierarchy(context.hierarchy_path, contract,
                                   dataset_key=context.key, object_ids=context.object_ids)
