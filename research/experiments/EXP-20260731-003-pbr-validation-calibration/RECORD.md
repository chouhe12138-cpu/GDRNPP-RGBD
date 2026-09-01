# EXP003 — PBR Validation Calibration

## 协议

- 状态：`COMPLETE / CALIBRATION_MISMATCH`
- LM-PBR calibration subset：300 images、2,087 valid LM-O instances
- 10 pose conditions，共 20,870 instance rows；2,000 次 image-cluster bootstrap
- 性质：pipeline calibration，不是 held-out formal validation

## 结果

| 方法 | ADD(-S)@0.1d |
|---|---:|
| Official Patch-PnP | 81.696% |
| Pred XYZ + Pred visible | 80.307% |
| Pred XYZ + GT visible | 80.498% |
| GT XYZ + GT visible | 99.952% |
| True-XYZ-error Top-50% | 84.236% |
| Oracle best Patch/RANSAC | 85.673% |
| Oracle best R/t axes | 85.721% |

GT XYZ 增益 `+19.454 pp`，区间 `[+17.437,+21.422]` 且 8/8 物体非负，但比冻结
的 20 pp pattern threshold 少 `0.546 pp`。GT visible 仅增 `0.192 pp` 且区间跨零；
axis-wise oracle 增 `4.025 pp`，比 5 pp 门槛少 `0.975 pp`。门槛没有在观察后放宽。

PBR 与 LM-O 的关键域差异：visible-mask IoU 中位数 `0.946 vs 0.803`，归一化
XYZ error 中位数 `0.929% vs 5.010%`；Top-50% error selection 在 PBR 为
`+3.833 pp`，在 LM-O 却为 `-1.799 pp`。可靠性/误差 Spearman 为
`-0.177 vs -0.062`。

因此 PBR 上的 precision/coverage 排序不能校准 LM-O 正式模型选择。官方 checkpoint
还可能见过 calibration scenes，所有数值只用于管线和域差异诊断。
