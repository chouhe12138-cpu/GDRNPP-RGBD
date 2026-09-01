# EXP007 — Patch-PnP Information Flow

- 状态：`COMPLETE`
- 协议：官方 checkpoint 固定权重，LM-O 1,445 targets，14 种输入/层级干预
- 结论：Region 是官方姿态头的主导输入；ROI2D 空间置换影响很小；改善 XYZ
  会引起内部响应，但没有稳定改善最终姿态。

该诊断只解释信息流，不产生新模型。原计划的全部跨模型 full 对比没有继续执行；
现有结论边界以已完成 official full 为限。
