"""The single source of EXP022 PCC architecture and loss settings."""

PCC_INIT_CFG = dict(
    token_dim=256, beam_k=2, num_heads=8,
    stage_attention=("global", "global", "window", "window"),
    window_size=8, shift_size=4, attention_dropout=0.0,
    route_weight=1.0, residual_weight=1.0, mask_weight=1.0,
    residual_beta=0.1,
)
