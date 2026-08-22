# EXP013C 运行记录

- 当前状态：`PLANNED / BLOCKED_UNTIL_A_AND_B_FORMAL_GATES_PASS`
- 条件满足后的目标服务器：`lab0`
- 本地只验证工程可执行性；本地通过不构成服务器正式训练授权。
- 本地验证实现 commit：`5b5de4216443cfc3f5110fa0e62ad8b8ec2f3bd9`
- 本地工程 gate：CPU/CUDA full-model preflight、1 epoch 真实数据 smoke、strict checkpoint roundtrip 和 checkpoint isolation 全部通过。

C 的服务器入口继续依赖 `EXPERIMENT.json` 的 `AUTHORIZED` 状态；当前必须拒绝启动。只有 A、B 的服务器 E40 正式 gate 均通过且 B 相对 A 有效，才可另行授权。A 或 B 任一失败时停止 C。
