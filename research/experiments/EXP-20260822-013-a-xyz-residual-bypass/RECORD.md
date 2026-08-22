# EXP013A 运行记录

- 当前状态：`AUTHORIZED / LOCAL_GATE_PASS / FORMAL_NOT_STARTED`
- 目标服务器：`lab0`
- 正式训练：40 epoch，seed 42，每 5 epoch 保存并评估，E40 为主比较点。
- EXP012 E40 冻结基准：BOP AR `0.678800`，ADD(-S) `0.494118`，AR_reS `0.491349`，AR_teS `0.791926`。
- 本地验证实现 commit：`5b5de4216443cfc3f5110fa0e62ad8b8ec2f3bd9`
- 本地 gate：CPU/CUDA full-model preflight、1 epoch 真实数据 smoke、strict checkpoint roundtrip 和 checkpoint isolation 全部通过。

本地 gate 已通过，允许在 lab0 按固定协议启动正式流程；正式运行尚未开始。
