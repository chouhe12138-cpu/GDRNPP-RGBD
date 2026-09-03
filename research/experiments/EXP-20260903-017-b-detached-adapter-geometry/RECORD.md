# EXP017-B — Detached Adapter Geometry

## 状态与协议

- 状态：`SERVER_SMOKE_PASS / FORMAL_AUTHORIZED / E40_PENDING`
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

本实验已获 matched formal 授权；E40 是唯一正式决策点。

## 本地门禁

- EXP017-B + EXP017 + pose_structure tests：`26 passed`。
- CPU preflight：`PASS`；参数 `13,000`，normal/detached forward value-exact，translation
  bitwise equal，adapter rotation 梯度保留，translation→adapter 梯度为零，
  adapter→共享 geometry encoder 梯度已切断。

## 服务器 smoke

- run：`RUN-20260903-083348-smoke-s42-a01`。
- source：`f02dadebfcb89d2cbe297d7409c10abfb5fe7013`；machine：lab1。
- config：`configs/gdrn/lmo_pbr/research/exp017/b_detached_adapter_geometry/smoke.py`。
- 用户确认训练正常完成、`exit_code=0`；该 run 只验证链路，不进入性能结论。

## Formal

- 用户已明确授权在 lab1 使用 source `f02dade` 与 matched `train.py` 启动 formal。
- formal run ID 尚待服务器启动结果确认；不得用后续本地整理 commit 替换运行中的只读
  release。
