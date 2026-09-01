# EXP013C — R/t Decoupled Fusion

- 状态：`COMPLETE / SCREEN_FAIL / ROTATION_SUPPORTED`
- revision 2：从 A 而非未过门槛的 B 继承，不包含 B attention
- formal run：`RUN-20260826-124748-formal-s42-a01`
- seed：42；checkpoint：`model_epoch_040.pth`
- 配置：`configs/gdrn/lmo_pbr/research/exp013/c_rt_decoupled/train.py`

E40：BOP AR `0.684646`、ADD(-S) target-micro `0.496886`、
macro-object `0.498841`、AR_reS `0.525029`、AR_teS `0.794233`。
相对 A，rotation 提高且 translation drop 在门槛内，但 ADD 下降
`0.013841`，因此只支持 R/t 专用表示改善 rotation，不支持整体更优。
外置证据：`E:\6D姿态估计\EXP-013\实验C\`。
