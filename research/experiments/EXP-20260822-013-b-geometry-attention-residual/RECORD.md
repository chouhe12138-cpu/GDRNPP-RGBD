# EXP013B 运行记录

- 当前状态：`AUTHORIZED / LOCAL_GATE_PASS / FORMAL_NOT_STARTED`
- 目标服务器：`lab1`
- 正式训练：40 epoch，seed 42，每 5 epoch 保存并评估，E40 为主比较点。
- 与 A 的唯一差异：16×16 几何特征上的 masked 3×3 局部注意力。
- 本地验证实现 commit：`5b5de4216443cfc3f5110fa0e62ad8b8ec2f3bd9`
- 本地 gate：CPU/CUDA full-model preflight、1 epoch 真实数据 smoke、strict checkpoint roundtrip 和 checkpoint isolation 全部通过。

本地 gate 已通过，允许在 lab1 按固定协议启动正式流程；正式运行尚未开始。

## 2026-08-22 lab1 服务器进度

- 离线 source bundle：`GDRNPP-RGBD-e316b9f3.bundle`
- detached source commit：`e316b9f33abf51cc8d80550e9dfd52ebb0420088`
- environment image ID：`sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee`
- `access` 与 `gate`：`PASS`
- smoke run：`RUN-20260822-110127-smoke-s42-a01`，已启动；完成状态待服务器 `status` 核验。

smoke 尚未记为 `COMPLETE`，因此 audit48 和 formal 仍不得视为已通过或已开始。
