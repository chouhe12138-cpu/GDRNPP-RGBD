# EXP005 — Patch-PnP-Only Control

- 状态：`COMPLETE / EPOCH_040`
- formal run：`RUN-20260811-063606-formal-s42-a01`
- seed：42
- checkpoint：`model_epoch_040.pth`，epoch 40 / iteration 255919
- 结果：BOP AR `0.691912`；ADD(-S) macro-object `0.506574`

该实验只适应原始 Patch-PnP，作为后续结构实验的 matched control。早期 managed
smoke 是无效基础设施 run，不计入科学结果。历史专用配置和执行框架已从当前树
移除，可由 Git 历史恢复。
