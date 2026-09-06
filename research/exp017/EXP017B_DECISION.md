# EXP017-B 决策

## 启动前结论：GO（历史决策）

EXP017 存在真实的共享优化耦合：同一个 `geometry_grid` 一方面经 EXP013A 的
`geometry_projection` 进入 rotation/translation 共用 latent，另一方面直接进入新增
rotation adapter。因而 rotation loss 会通过 adapter 额外更新 translation 所依赖的共享
geometry encoder；`raw_t = raw_t_A` 只能保证直接输出路径不变，不能隔离训练梯度。

E10 checkpoint 的一次性只读 autograd 检查确认该额外梯度非零且不可忽略，但小样本中
没有观察到稳定的瞬时 R/t 梯度冲突。因此它可能参与 E5→E10 的 teS/ADD 下降，却不足以
证明因果。开展 B 的目的，是用 matched 实验消除这条非预期优化边，而不是预言 B 必然
提高 translation。

EXP017-B 相对 EXP017 的唯一变化是：

```python
delta_r = rotation_adapter(geometry_grid.detach(), support)
```

EXP013A 自身 geometry latent 仍正常训练；translation、loss、optimizer、数据、周期、
pooling、position、参数量与 renderer 协议均不变。B 使用独立 experiment/output ID。

本地 tests/preflight 与服务器 smoke 通过后，已按授权运行 matched formal，E40
是唯一正式决策点；中间 checkpoint 只作趋势记录。实验现已完成，最终指标与结论见
[EXP017-B RECORD](../experiments/EXP-20260903-017-b-detached-adapter-geometry/RECORD.md)。
