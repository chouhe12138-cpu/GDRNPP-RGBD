# EXP003 — PBR Validation Calibration

- 状态：`COMPLETE / CALIBRATION_MISMATCH`
- 数据：LM-PBR calibration subset，300 images、2,087 instances
- 结论：PBR calibration 与 LM-O 正式分布不匹配，不能用于正式模型选择。

官方 Patch-PnP ADD(-S) 为 81.696%，GT XYZ + GT visible 为 99.952%。GT XYZ
仍是最大因素，但跨域幅度与冻结门槛不一致；可靠性排序在 PBR 和 LM-O 上也不
稳定。历史 split 与执行代码已退出当前树，结论保留用于避免重复采用该协议。
