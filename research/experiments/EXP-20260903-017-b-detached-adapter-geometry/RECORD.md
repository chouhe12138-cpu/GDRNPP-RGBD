# EXP017-B — Detached Adapter Geometry

## 状态与协议

- 状态：`IMPLEMENTED / LOCAL_PREFLIGHT_PASS / NOT_AUTHORIZED_TO_RUN`
- experiment ID：`EXP-20260903-017-b-detached-adapter-geometry`
- 母实验：EXP017。
- 唯一变量：adapter 输入由 `geometry_grid` 改为 `geometry_grid.detach()`。
- EXP013A 自身 geometry latent、translation、loss、optimizer、数据、40 epoch、pooling、
  position 全部不变；不从 EXP017 checkpoint warm-start。
- training renderer：disabled；BOP evaluation renderer：cpp。

## 假设

EXP017 的 rotation residual loss 会经 adapter 输入反传到 EXP013A 的共享 geometry
encoder。只切断这条新增梯度边，可以保留 adapter 的 rotation 表达力，同时减少 rotation
优化对后续共享 translation 表示的额外扰动。

## Gate

本实验尚未获 smoke、screening 或 formal 授权。最小下一步仅为本地 tests/preflight；若
之后单独授权，先做同状态 forward/gradient smoke，再决定是否请求 matched formal。

## 本地门禁

- EXP017-B + EXP017 + pose_structure tests：`26 passed`。
- CPU preflight：`PASS`；参数 `13,000`，normal/detached forward value-exact，translation
  bitwise equal，adapter rotation 梯度保留，translation→adapter 梯度为零，
  adapter→共享 geometry encoder 梯度已切断。
- 未启动 smoke、screening 或 formal。
