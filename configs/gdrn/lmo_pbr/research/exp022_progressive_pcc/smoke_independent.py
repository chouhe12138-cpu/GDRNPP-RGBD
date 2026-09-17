import os

_base_ = ["./smoke_reused.py"]

MODEL = dict(POSE_NET=dict(PCC_HEAD=dict(HIERARCHY_PATH=os.path.join(
    os.environ.get("GDRN_DATASET_CACHE_DIR", ".local/dataset_cache"),
    "exp022", "independent_v2.npz",
), INIT_CFG=dict(
    token_dim=256, beam_k=2, num_heads=8,
    stage_attention=("global", "global", "window", "window"),
    window_size=8, shift_size=4, attention_dropout=0.0,
))))
