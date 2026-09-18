"""LM13 data ablation: real LM images only, everything else as the main protocol."""

_base_ = ["./train_lm13_gdrn.py"]

DATASETS = dict(TRAIN=("lm_13_train_online",))
TRAIN_PROTOCOL = dict(NAME="lm13_real_only", DATA_DOMAIN="real")
