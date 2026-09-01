# EXP010 — CPM Learning-Rate Control

## 协议与运行

- 状态：`FAILED / USER_NO_RETRY`
- 问题：EXP009 的失败是否主要来自随机初始化新头使用过低学习率？
- 唯一变量：Ranger lr 从 EXP009 的 `8e-5` 改为 `8e-4`
- CPM、初始化、数据、loss、warmup、batch 48、40 epoch、seed 42 与评估协议不变
- formal run：`RUN-20260816-081032-formal-s42-a01`
- source commit：`29580f65abfeb7625bab252011c19399325b0fa2`
- 约 E27 / iteration 166999 崩溃；没有 E40 checkpoint

## 已完成评估

| Epoch | BOP AR | ADD(-S) target-micro |
|---:|---:|---:|
| 5 | 0.496205 | 0.235986 |
| 10 | 0.540775 | 0.328028 |
| 15 | **0.567045** | **0.333564** |
| 20 | 0.513548 | 0.287889 |
| 25 | 0.533396 | 0.303806 |

E15 是已观察峰值，E20 回落；E21 起 loss_PM_R 均值由约 `0.0148` 升到
`0.0153+`，total loss 由 `0.167` 升到 `0.187`，出现高学习率不稳迹象。EXP009
在 E30 已达到 BOP `0.599463`，高于 EXP010 的全部已观察点。

由于没有固定 E40，预注册的 matched learning-rate gate 没有完成，不能写成正式
E40 对照。但现有方向性证据不支持“EXP009 主要因学习率过低而失败”。用户于
2026-08-30 决定不重试，E5–E25 指标和崩溃事实作为失败证据保留。
