# EXP013A — XYZ–ROI2D Residual Bypass

- 状态：`COMPLETE / PASS_E40_FORMAL_GATE`
- formal run：`RUN-20260822-112714-formal-s42-a01`
- seed：42；checkpoint：`model_epoch_040.pth`
- 配置：`configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual/train.py`

| 指标 | EXP012 | A E40 | 增量 | 结论 |
|---|---:|---:|---:|---|
| BOP AR | 0.678800 | 0.683956 | +0.005156 | PASS |
| ADD(-S) | 0.494118 | 0.510727 | +0.016609 | PASS |
| 逐物体非负 | — | 5/8 | — | PASS |

AR_reS `0.498039`，AR_teS `0.797693`。结构诊断显示 geometry residual、
Region 和空间布局都被实际使用。结果支持独立几何残差路径，不支持删除 Region。
