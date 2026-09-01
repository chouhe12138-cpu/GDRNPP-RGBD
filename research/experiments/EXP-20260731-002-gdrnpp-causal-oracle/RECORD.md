# EXP002 — Causal Oracle Diagnostic

## 协议

- 状态：`COMPLETE / PASS_XYZ_GEOMETRY`
- 官方 checkpoint；LM-O BOP19 1,445 targets；GT bbox；无训练
- GT、depth 和 oracle selection 只用于因果诊断，不是可部署性能
- 问题：主要可恢复误差来自 support、XYZ、reliability 还是 pose aggregation？

full 完成 1,445 targets × 12 conditions；GT-XYZ 最大重投影误差 `0.4219 px`。

## 正式因果结果

| 比较 | BOP AR | ADD(-S) | ΔBOP | ΔADD | 非负物体 | oracle gap closure |
|---|---:|---:|---:|---:|---:|---:|
| Pred XYZ + GT visible | 69.142% | 54.256% | — | — | — | — |
| GT XYZ + GT visible | 100.000% | 100.000% | +30.858 pp | +45.744 pp | 8/8 | 97.49% |
| Pred XYZ + fixed shared support | 69.255% | 53.841% | — | — | — | — |
| GT XYZ + fixed shared support | 99.377% | 99.377% | +30.122 pp | +45.536 pp | 8/8 | 97.05% |

第一组 ADD 增量的 paired 95% interval 为 `[+43.577,+47.933] pp`。两种 support
控制下结论一致：预测 XYZ geometry 是主要可恢复瓶颈。

被排除为主要解释的结果：

- GT visible support 只增加 `1.176 pp` ADD，BOP 下降 `0.452 pp`，区间跨零。
- predicted full-mask 相对 predicted visible 的 BOP/ADD 分别下降
  `0.348/0.554 pp`。
- 按真实 XYZ error 选择 Top-50% 仍使 BOP/ADD 下降 `0.435/1.799 pp`；仅有
  点级准确性排序不足，空间覆盖和联合几何同样重要。
- 当前 mask × region score 与真实 XYZ error 的平均 Spearman 为 `-0.062`。

axis-wise oracle selection 可将 BOP/ADD 提高 `3.020/8.028 pp`，8/8 物体非负，
但只关闭 `17.11%` 的 XYZ oracle gap，低于 30% 主因素门槛，只作为次级互补信号。

描述性诊断：visible/full mask IoU 中位数为 `0.803/0.863`；instance XYZ 中位误差
`7.70 mm`（物体直径的 `5.01%`）。这些量描述难度，不改变 oracle 的因果边界。
