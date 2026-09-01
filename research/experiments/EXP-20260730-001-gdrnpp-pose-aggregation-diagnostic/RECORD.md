# EXP001 — Pose Aggregation Diagnostic

## 协议

- 状态：`COMPLETE / FAIL`
- 官方 ConvNeXt-Base LM-O checkpoint；LM-O BOP19 1,445 targets；GT bbox
- seed `20260730`；无训练、无 depth；失败姿态按 ADD(-S)=0 计
- 问题：固定网络输出后，显式几何聚合能否稳定优于官方 Patch-PnP？

## 正式结果

表中数值为百分数；rotation 使用 symmetry-aware 口径。

| 方法 | BOP AR | VSD AR | MSSD AR | MSPD AR | ADD(-S) | Median R (°) | Median t (mm) | 失败率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Patch-PnP | 69.021 | 51.671 | 66.713 | 88.678 | 50.242 | 4.64 | 17.06 | 0.000% |
| EPnP all | 68.671 | 52.399 | 67.052 | 86.561 | 52.872 | 4.19 | 15.34 | 0.069% |
| RANSAC-EPnP | **69.594** | **53.080** | **67.799** | 87.903 | **53.080** | **4.21** | **15.34** | 0.069% |
| Reliable RANSAC | 68.719 | 52.095 | 66.713 | 87.349 | 51.696 | 4.27 | 16.16 | 0.069% |
| Geometric R + network t | 68.447 | 51.340 | 66.055 | 87.945 | 49.827 | 4.27 | 17.05 | 0.069% |
| Network R + geometric t | 68.955 | 52.338 | 66.900 | 87.626 | 51.834 | 4.63 | 16.06 | 0.346% |

RANSAC-EPnP 相对 Patch-PnP 的 BOP AR 提高 `+0.573 pp`，ADD(-S) 提高
`+2.838 pp`，但逐物体只有 `5/8` 非负，未达到预注册的 `6/8` 稳定性门槛。
它的增益来自 VSD/MSSD，MSPD 反而降低 `0.775 pp`。

固定 reliability proxy 也未通过验证：Top-50% filtering 将 RANSAC BOP AR 从
`69.594` 降到 `68.719`。极低可见率的 eggbox/glue 上还出现少量极端 translation
错误。因此结论为：存在物体依赖的几何信号，但没有找到可稳定替代官方头的固定
聚合方案，RANSAC 不进入部署路径。

原始逐实例/逐物体结果位于历史外置目录 `output/EXP-20260730-001/full/`。
