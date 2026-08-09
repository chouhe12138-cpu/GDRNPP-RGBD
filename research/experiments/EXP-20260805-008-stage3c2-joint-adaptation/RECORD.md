# EXP-20260805-008 — C2 联合适应

状态：`RUNNING — CURRENTLY_BELOW_GATE`。这不是最终失败结论。

C2 从官方 LM-O checkpoint 独立初始化，冻结 backbone 和 geometry head，同时
训练 Patch-PnP 与质量/覆盖模块。固定协议仍为 40 epoch、seed `20260731`、
LM-O GT-box 每 5 epoch 评估；当前服务器运行链保持不变。

## 当前周期评估

| Epoch | BOP AR (%) | ADD(-S) macro-object (%) | ADD(-S) micro-target (%) |
|---:|---:|---:|---:|
| 5 | 69.1306 | 51.5069 | 51.4187 |
| 10 | 69.1052 | 51.6128 | 51.4187 |
| 15 | 68.8088 | 50.4537 | 50.2422 |
| 20 | 68.9209 | 49.2216 | 49.0657 |
| 25 | **69.2738** | 未取得 | 49.6194 |
| 30 | 69.1532 | 未取得 | 50.3806 |

按预注册主指标 BOP AR，当前最佳是 Epoch 25，而不是 Epoch 5。训练日志在
Epoch 25 评估后重新保存了 `model_best.pth`。现有下载日志已继续到 Epoch 33，
但所提供文件中最后一次完整评估是 Epoch 30。

官方 BOP 基线为约 69.0415%。Epoch 25 当前提高约 `+0.2323 pp`，仍低于
`+0.50 pp` 门槛，因此截至目前没有 checkpoint 通过全部筛选门槛。Epoch 35/40
尚未评估，正式结论等待训练自然结束。

## 指标边界

- `macro-object` 是八个物体 recall 的平均值，是预注册 ADD(-S)主指标。
- `micro-target` 是 1,445 个 target 上的成功比例。
- Epoch 25/30 的现有日志只有 micro recall，没有宏平均与逐物体结果；记录为
  缺失，不用 micro值替代 macro值。
- 自动 `EVAL_SUMMARY` 的 ADD字段为 `null` 是历史 evaluator 目录查找问题，
  不代表日志中的 target-level ADD没有计算。

## 证据与后续

完整日志和六份评估日志继续只读保存在 `E:\6D姿态估计\C2`，不复制进 Git。
机器可读数值、来源哈希和当前 best 身份统一记录在 `RESULTS.json`。

训练结束后追加 Epoch 35/40，重新确定最终 best；仅外置保存最终
`model_best.pth`、SHA-256、严格加载和参数隔离结果。若最终仍未过 gate，再将
状态更新为 `COMPLETE — C2_SCREEN_FAIL`。

C2 的主要因果比较原本是 `C2-B`。若 B 最终取消，只能报告 C2 相对官方基线
未通过筛选，不能声称已经测得质量/覆盖模块相对 PnP-only 适应的额外贡献。
