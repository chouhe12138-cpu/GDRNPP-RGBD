"""Editable runtime defaults for new research experiments, independent of dataset."""

_base_ = ["../../../_base_/gdrn_base.py"]

SEED = 42
DATALOADER = dict(NUM_WORKERS=16, FILTER_VISIB_THR=0.3)
SOLVER = dict(
    IMS_PER_BATCH=4, REFERENCE_BS=48, TOTAL_EPOCHS=40,
    CHECKPOINT_PERIOD=5, CHECKPOINT_BY_EPOCH=True, MAX_TO_KEEP=10,
    OPTIMIZER_CFG=dict(_delete_=True, type="AdamW", lr=3e-4,
                       weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8),
    AMP=dict(ENABLED=True), LR_SCHEDULER_NAME="flat_and_anneal",
    WARMUP_RATIO=0.04, WARMUP_FACTOR=0.001, WARMUP_METHOD="linear",
    ANNEAL_METHOD="cosine", TARGET_LR_FACTOR=0.01,
    BEST_CHECKPOINT=dict(ENABLED=False),
)
TEST = dict(EVAL_PERIOD=5, TEST_BBOX_TYPE="gt", USE_PNP=True,
            PNP_TYPE="ransac_pnp")
VAL = dict(RENDERER_TYPE="cpp", USE_BOP=True,
           ERROR_TYPES="mspd,mssd,vsd,ad,reS,teS", SPLIT="test", SPLIT_TYPE="",
           N_TOP=1, TARGETS_FILENAME="test_targets_bop19.json")
RUN_ARTIFACTS = dict(STRUCTURED_LAYOUT=True, COMPACT_LOG=True,
                     TENSORBOARD=False, SKIP_DUPLICATE_FINAL_EVAL=True)
TRAIN = dict(PRINT_FREQ=500)

# New experiments can change SOLVER and TEST in their own config. Formal runs
# remain disabled until their protocol and resource check are recorded.
RESEARCH_PROTOCOL = dict(SCHEDULE="configurable", FORMAL_READY=False)
