# EXP000 — 官方 GDRNPP 基线

- 状态：`COMPLETE`
- 模型：GDRNPP ConvNeXt-Base，LM-O class-aware 官方 checkpoint
- 协议：LM-O BOP19、GT bbox、RGB-only、官方 Patch-PnP
- 性质：历史官方 checkpoint 验证，不代表本项目重新训练的模型

完整 1,445-target 推理在后续诊断中按同一模型与测试集复现。不同执行链曾报告
BOP AR `0.69021`（EXP001/002/004）和 `0.690415`（Stage 3C/EXP007 校准）；
这些数值来自不同结果汇总链，不能在没有完整 metric ID 时视为逐位相同的基线。

ADD(-S) 同样必须区分 1,445 个 target 的 micro recall 与 8 个物体等权的
macro-object recall。后续训练实验均在各自 RECORD 中绑定其实际使用的父基线、
聚合口径和增量，不用本页的历史基线替代 matched comparison。
