# EXP013B — Geometry Attention Residual

- 状态：`COMPLETE / FAIL_STRICT_BOP_GATE / ATTENTION_EFFECTIVE`
- formal run：`RUN-20260822-112724-formal-s42-a01`
- seed：42；checkpoint：`model_epoch_040.pth`
- 配置：`configs/gdrn/lmo_pbr/research/exp013/b_geometry_attention/train.py`

| 指标 | EXP012 | B E40 | 增量 | 结论 |
|---|---:|---:|---:|---|
| BOP AR | 0.678800 | 0.683691 | +0.004891 | FAIL |
| ADD(-S) | 0.494118 | 0.514187 | +0.020069 | PASS |
| 逐物体非负 | — | 5/8 | — | PASS |

BOP 门槛差 `0.000109`，不能事后放宽。B 与 A 的 BOP 基本持平而 ADD 高
`0.003460`，因此 attention 按 B−A 规则有效，但整体正式 gate 仍失败。
AR_reS `0.498039`，AR_teS `0.801153`。
