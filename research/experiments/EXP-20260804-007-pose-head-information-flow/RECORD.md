# EXP-20260804-007 — Patch-PnP 姿态头信息流诊断

状态：`OFFICIAL_FULL_COMPLETE — CROSS_MODEL_COMPARISON_PENDING`

本实验不训练、不更新模型参数，使用固定支持上的十四种干预，观察 XYZ、ROI 2D
和 Region 信息在 Patch-PnP 输入、卷积、全连接、旋转/平移分支和最终姿态中的
响应。

## 已完成

- official smoke：8/8 targets，质量门通过；
- official audit80：80/80 targets，质量门通过；
- official full：1,445/1,445 targets，14/14 BOP 条件，质量门通过；
- 3,571 个正式产物文件的哈希已复算通过；
- C1 固定 Epoch 40 权重已经转移并核验 SHA-256。

official 全量结果确认 Region 是姿态头主导输入；ROI 2D 空间置换几乎不影响
最终姿态；完整 GT-XYZ 产生明显内部响应，但没有稳定改善 BOP 或 ADD(-S)。

## 尚未完成

- C1 固定 Epoch 40 同协议 full；
- B、C2 正式训练及固定 Epoch 40 同协议 full；
- official/C1/B/C2 横向信息消费比较。

完整协议、数值、质量控制和复现命令见：
`research/POSE_HEAD_DIAGNOSTIC_HANDOFF_CN.md`。

raw output 保持在 Git 忽略目录：
`output/EXP-20260804-007-pose-head-information-flow/`。
