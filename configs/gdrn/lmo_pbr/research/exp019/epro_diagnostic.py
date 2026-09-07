EXPERIMENT_ID = "EXP-20260907-019-epro-geometry-utilization"

DIAGNOSTIC = dict(
    DATASET="lmo_bop_test",
    TARGETS_FILENAME="test_targets_bop19.json",
    BBOX_TYPE="gt",
    NUM_TARGETS=1445,
    SEED=20260730,
    ALPHAS=(0.00, 0.25, 0.50, 0.75, 1.00),
    CONSUMERS=("patch", "ransac", "epro"),
    FIXED_SUPPORT="EXP004_EXACT_SHARED_SUPPORT",
    EPRO_WEIGHT_MODE="uniform",
    EPRO=dict(
        VERSION="v2",
        MC_SAMPLES=512,
        NUM_ITER=4,
        LM_NUM_ITER=3,
        RELATIVE_HUBER_DELTA=0.1,
        Z_MIN_M=0.01,
        FAST_MODE=True,
    ),
    GATES=dict(
        GT_XYZ_BOP_MIN=0.95,
        GT_XYZ_ADD_MIN=0.95,
        SPEARMAN_MIN=0.90,
        RECOVERY_RATIO_MIN=0.50,
    ),
)

TRAINING = False
LEARNED_RELIABILITY = False
POSE_REFINEMENT = False
EXP018_CORRECTION = False
BPNP = False
