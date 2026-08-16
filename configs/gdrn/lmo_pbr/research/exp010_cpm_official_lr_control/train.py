_base_ = ["../exp009_cpm_head/train.py"]

# Matched optimization control for EXP009.  The CPM architecture, data,
# initialization, losses, schedule, and evaluation protocol are inherited
# unchanged.  Only the fresh pose-head learning rate is changed.
EXPERIMENT_ID = "EXP-20260816-010-cpm-official-lr-control"
OUTPUT_DIR = (
    "output/experiments/EXP-20260816-010-cpm-official-lr-control/"
    "RUN-SET-BY-LAUNCHER"
)

SOLVER = dict(
    OPTIMIZER_CFG=dict(
        _delete_=True,
        type="Ranger",
        lr=8e-4,
        weight_decay=0.01,
    ),
)
