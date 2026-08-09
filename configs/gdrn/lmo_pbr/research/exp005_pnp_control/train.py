_base_ = ["../../convnext_stage3c0_pnp_only_lmo.py"]

# The legacy Stage 3C0/B config remains frozen.  New managed runs inherit its
# scientific protocol and override only execution identity fields here.
EXPERIMENT_ID = "EXP-20260731-005-pnp-only-control"
OUTPUT_DIR = (
    "output/experiments/EXP-20260731-005-pnp-only-control/RUN-SET-BY-LAUNCHER"
)
SEED = 42

DATALOADER = dict(NUM_WORKERS=16)
