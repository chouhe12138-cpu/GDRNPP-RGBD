# EXP010 — CPM Learning-Rate Control

- 状态：`FAILED / USER_NO_RETRY`
- formal run：`RUN-20260816-081032-formal-s42-a01`
- seed：42；Ranger lr `8e-4`
- 运行停止：约 epoch 27，无 E40 checkpoint
- 已评估 BOP AR：E5 `0.496205`、E10 `0.540775`、E15 `0.567045`、
  E20 `0.513548`、E25 `0.533396`

较高学习率没有超过 EXP009 的平台，并在 E21 后出现 loss 回升。因为没有固定
E40，不能作完整 matched gate；用户于 2026-08-30 决定不重试。
