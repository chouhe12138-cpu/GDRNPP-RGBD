# EXP017-B 决策

## 结论：GO（仅准备候选，不授权运行）

耦合真实存在。EXP017 的 `_encode_geometry()` 产生同一个 `geometry_grid`：一条路径经
`geometry_projection → latent_a` 同时供 `raw_r_A/raw_t_A` 使用，另一条路径不经 detach
进入 rotation adapter。因此 rotation loss 除了 EXP013A 原有的 rotation→geometry
梯度，还会经 adapter 向共享 geometry encoder 注入一项新增梯度。`raw_t = raw_t_A`
只保证当前 forward 不直接加 residual，不能保证训练期间 translation 所依赖的共享参数
不被 adapter 的 rotation 梯度更新。

## E10 只读证据

checkpoint `model_epoch_010.pth` 上，以 4 个 LM-PBR batch、8 个样本运行无 optimizer
step 的 autograd 诊断：

- normal 与 `geometry_grid.detach()` 的 raw R、raw T、decoded R/T 全部 value-exact；
  detach 只改变反传图。
- adapter OFF 与 normal 的 raw T / decoded T RMS 均为 `0`；raw R residual RMS 均值
  `0.074650`，该小子集 rotation error 均值由 OFF 的 `7.9220°` 降为 `6.6681°`。
- EXP017 normal 的共享 geometry encoder rotation-proxy 梯度范数均值 `2.0347`；只切断
  adapter 输入后为 `1.5303`。
- 两者梯度差（adapter 新增注入）范数均值 `0.7367`，相对 A rotation 路径梯度范数
  比值均值 `0.5793`、中位数 `0.3073`；不是数值可忽略的边。
- 新增注入梯度与 translation-proxy 梯度 cosine 均值 `+0.1919`、中位数
  `+0.2990`，当前小样本没有显示稳定的瞬时梯度冲突；这限制了“它直接导致 teS 下降”
  的解释强度。
- adapter 参数在 normal/detached 下的梯度完全一致，说明 detach 没有切断 adapter
  自身学习。

这些子集 re/te 和 rotation proxy 只用于机制诊断，不是正式 BOP/ADD。尤其，E10 时把
adapter OFF 不能撤销 E1–E10 已经写入共享 encoder 的优化历史；同一 E10 state 上
normal/detached forward 相同也不能预测重新训练后的最终指标。

## 对 E5→E10 的解释边界

现象与耦合方向一致：AR_reS `0.351557→0.419608`，同时 AR_teS
`0.781084→0.770473`、ADD(-S) `0.501730→0.491349`。真实且非微小的新增梯度使其成为
合理机制候选，但两个时间点不能证明因果，且小样本瞬时 cosine 并未显示稳定冲突；学习
动态、逐物体组成和正常评估波动仍可能解释下降。GO 的主要理由是消除 EXP017 新增的
非预期共享优化边、恢复严格变量隔离，并以 matched run 检验其长期效应；当前证据不支持
宣称 EXP017-B 必然改善 translation。

## 唯一变化

```python
delta_r = rotation_adapter(geometry_grid.detach(), support)
```

EXP013A 自身 geometry latent 路径继续训练；不冻结 A，不改变 translation、loss、
optimizer、数据、周期、pooling、position、参数量或 renderer。EXP017-B 使用独立 ID
`EXP-20260903-017-b-detached-adapter-geometry`；training renderer disabled，BOP
evaluation renderer cpp。

## 后续最小验证

1. 当前只保留代码、单测与 CPU preflight；不运行 EXP017-B，也不干预正在运行的 EXP017。
2. 若之后另行授权，先做一次本地真实小批次 smoke：验证 normal/detached forward exact、
   adapter 可更新、共享 encoder 不接收 adapter 增量梯度、translation-only 隔离和 checkpoint
   round-trip。
3. smoke 通过后才请求 matched formal；只以预注册 E40 与 EXP017/EXP013A 比较，E5/E10
   仅作趋势检查，不作提前决策。
