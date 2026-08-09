# EXP007 C1 diagnostic smoke 追加验收（2026-08-08）

状态：`C1_SMOKE_PASS_FULL_PENDING`。

使用 C1 formal Epoch 40 checkpoint 在本地 CPU/FP32 跑通 statistical diagnostic
smoke：8/8 targets、14/14 条件，质量门通过；模型状态未变化，无非有限标量，
baseline re-entry 最大误差均小于 `1e-6`。

该运行没有 optimizer、没有参数更新，也没有写 instance-level feature。输出仅放在
`/tmp/gdrnpp-exp007-c1-smoke-20260808`，不作为新的正式历史产物保留。

本次结果证明 C1 同协议 diagnostic 链路仍可执行，但不替代 1,445-target full。
EXP007 总状态仍是 cross-model comparison pending；B/C2 必须等待各自正式实验
状态确定后再处理。
