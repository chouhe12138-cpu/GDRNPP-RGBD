# EXP013E — Official Head Random Initialization

- 状态：`COMPLETE / DIAGNOSTIC_PARTIAL_SUPPORT_M2_M3`
- formal run：`RUN-20260829-080742-formal-s42-a01`
- seed：42；checkpoint：`model_epoch_040.pth`
- 配置：`configs/gdrn/lmo_pbr/research/exp013/e_official_head_random/train.py`

E40：BOP AR `0.688581`、ADD(-S) target-micro `0.510727`、
macro-object `0.512940`、AR_reS `0.535409`、AR_teS `0.801153`。
reS 落在预注册 0.52–0.54 部分支撑带，说明官方头结构可读出与预训练继承都重要。
早期派生权重 smoke 无效，不计入结果。外置证据：
`E:\6D姿态估计\EXP-013\实验E\`。
