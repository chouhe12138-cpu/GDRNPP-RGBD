from dataclasses import dataclass
from typing import Tuple


EXPERIMENT_ID = "EXP-20260907-019-epro-geometry-utilization"
EXPECTED_OFFICIAL_WEIGHT_SHA256 = (
    "bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c"
)
EXPECTED_LMO_TARGETS = 1445


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str = EXPERIMENT_ID
    alphas: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    seed: int = 20260730

    epro_mc_samples: int = 512
    epro_num_iter: int = 4
    epro_lm_num_iter: int = 3
    epro_relative_huber_delta: float = 0.1
    epro_z_min_m: float = 0.01
    epro_fast_mode: bool = True
    weight_mode: str = "uniform"
    uniform_weight: float = 1.0

    max_correspondences: int = 0
    min_correspondences: int = 6
    gate_gt_xyz_bop: float = 0.95
    gate_gt_xyz_add: float = 0.95
    gate_spearman: float = 0.90
    gate_recovery_ratio: float = 0.50

    historical_patch_alpha0_add: float = 0.50242
    historical_patch_alpha0_bop: float = 0.69021
    historical_ransac_alpha0_add: float = 0.53841
    historical_ransac_alpha0_bop: float = 0.69255
    historical_ransac_alpha1_add: float = 0.99377
    historical_ransac_alpha1_bop: float = 0.99377

