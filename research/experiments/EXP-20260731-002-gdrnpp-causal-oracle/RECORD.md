# EXP002 — Causal Oracle Diagnostic

- 状态：`COMPLETE / PASS_XYZ_GEOMETRY`
- 协议：官方 checkpoint、LM-O BOP19 1,445 targets、GT bbox、无训练
- 结论：预测 XYZ geometry 是主要可恢复瓶颈。

在固定 support 下用 GT XYZ 替换预测 XYZ，BOP AR 从约 0.6926 提高到
0.9938，ADD(-S) 从约 0.5384 提高到 0.9938，8/8 物体非负。GT mask、简单
pixel reliability 和 double-mask 变化没有形成同等级稳定增益。Oracle 数值只用于
因果诊断，不是可部署性能。
