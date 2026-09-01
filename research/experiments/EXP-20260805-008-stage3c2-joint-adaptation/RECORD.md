# EXP008 — Stage 3C2 Joint Adaptation

- 状态：`COMPLETE / C2_SCREEN_FAIL`
- 协议：冻结 backbone/geometry，同时训练 Patch-PnP 与 quality/coverage 模块
- seed：20260731；40 epoch
- 固定 E40 BOP AR：`0.693006`
- 最高观测 BOP AR：E35 `0.693520`

两者都没有达到相对官方基线 +0.50 pp 的门槛。E35/E40 的正式 ADD(-S)
macro-object 未生成，保持缺失，不用其他 BOP 指标代替。固定 E40 checkpoint
外置于 `E:\6D姿态估计\C2\model_epoch_040.pth`。
