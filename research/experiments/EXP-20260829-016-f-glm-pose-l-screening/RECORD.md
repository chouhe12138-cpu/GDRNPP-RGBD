# EXP013F — GLM-Pose-L Screening

- 状态：`COMPLETE / SCREEN_FAIL_2_OF_4_GATES`
- formal run：`RUN-20260829-103858-formal-s42-a01`
- seed：42；checkpoint：`model_epoch_040.pth`
- 配置：`configs/gdrn/lmo_pbr/research/exp013/f_glm_pose_l/train.py`

E40：BOP AR `0.684129`、ADD(-S) target-micro `0.504498`、
macro-object `0.506465`、AR_reS `0.515802`、AR_teS `0.799308`。
reS 与 BOP 通过；teS 差 `0.0035`、ADD 差 `0.0062`，四项只过 2 项，
正式结论为边缘 `SCREEN_FAIL`。attention pooling 的 rotation 读出有效，
深度统计没有形成优于 B 的 translation 优势。外置证据：
`E:\6D姿态估计\EXP-013\实验F\`。
