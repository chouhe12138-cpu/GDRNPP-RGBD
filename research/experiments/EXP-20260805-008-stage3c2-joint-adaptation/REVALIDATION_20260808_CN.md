# C2 运行中追加验收快照（2026-08-08）

状态：`RUNNING_SNAPSHOT`，不是最终验收结论。

本次只读检查了 `E:\6D姿态估计\C2_handoff_20260808`。实验身份为 commit
`24a1d7aa7159cfc8113f4cfe8c11bbd2ce3ba290`、固定 seed `20260731`、同一 Docker
image。该 commit 是当前仓库 HEAD 的祖先；正在运行的容器和代码未被触碰。

交接快照最后位置为 Epoch 24、`148999/255920`（58.2%），最近 checkpoint 为
Epoch 20。交接包没有复制任何 checkpoint，因此目前不能进行最终权重加载与
隔离验收。

| Epoch | BOP AR | ADD(-S)@0.1d 宏平均 | ADD(-S) target recall |
|---:|---:|---:|---:|
| 5 | 0.6913057 | 0.5150687 | 0.5141869 |
| 10 | 0.6910519 | 0.5161275 | 0.5141869 |
| 15 | 0.6880877 | 0.5045371 | 0.5024221 |
| 20 | 0.6892088 | 0.4922156 | 0.4906574 |

这些只是周期评估，不用于提前替代固定 Epoch 40 结论。自动 `EVAL_SUMMARY`
中的 ADD(-S) `null` 属于历史冒号/等号目录匹配问题；交接包中的独立结构化
ADD(-S) JSON 有效。

Epoch 40 后仍需补：最终 checkpoint 与 SHA-256、严格加载和参数隔离、最终
BOP/ADD(-S)、正式日志结束状态以及同协议 diagnostic。B formal 继续不启动。

机器可读详情见 `REVALIDATION_20260808.json`；原 `ACCEPTANCE.json` 保持只读。
