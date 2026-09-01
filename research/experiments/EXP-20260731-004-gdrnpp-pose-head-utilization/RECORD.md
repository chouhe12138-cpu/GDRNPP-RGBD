# EXP004 — Pose-Head XYZ Utilization

## 协议

- 状态：`COMPLETE / PATCH_PNP_UNDERUTILIZATION`
- 官方 checkpoint、LM-O BOP19 1,445 targets、GT bbox、无训练
- 固定 support，将 predicted XYZ 按 alpha 插值到 GT XYZ
- 同时比较冻结的官方 Patch-PnP 与 RANSAC；alpha=0 精确复现父诊断

## Alpha sweep

| Alpha | Patch ADD(-S) | Patch BOP AR | RANSAC ADD(-S) | RANSAC BOP AR |
|---:|---:|---:|---:|---:|
| 0.00 | 50.242% | 69.021% | 53.841% | 69.255% |
| 0.25 | 49.827% | 68.971% | 61.592% | 71.769% |
| 0.50 | 49.550% | 68.950% | 73.910% | 78.356% |
| 0.75 | 49.204% | 68.958% | 85.329% | 85.392% |
| 1.00 | 49.550% | 68.324% | 99.377% | 99.377% |

alpha 0→1 的逐物体 ADD(-S) 变化：

| 物体 | Patch-PnP Δ(pp) | RANSAC Δ(pp) |
|---|---:|---:|
| ape | -1.143 | +48.571 |
| can | -4.020 | +20.101 |
| cat | -4.678 | +52.632 |
| driller | 0.000 | +14.000 |
| duck | 0.000 | +71.667 |
| eggbox | +1.667 | +61.111 |
| glue | +1.429 | +22.143 |
| holepuncher | +1.500 | +72.500 |

RANSAC 的 ADD/BOP 分别单调改善 `+45.536/+30.122 pp`，8/8 物体非负；官方
Patch-PnP 却下降 `0.692/0.697 pp`，只在 5/8 物体非负。结论是 dense XYZ 含有
可用几何信息，但冻结的直接头没有稳定消费逐步改善的 XYZ。RANSAC 仍只作诊断，
不作为最终部署方法。
