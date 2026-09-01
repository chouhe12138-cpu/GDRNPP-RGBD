# EXP009 — Correspondence-Aware Moment Pose Head

- 状态：`COMPLETE / CPM_SCREEN_FAIL`
- formal run：`RUN-20260811-063626-formal-s42-a01`
- seed：42
- checkpoint：`model_epoch_040.pth`，epoch 40 / iteration 255919
- E40：BOP AR `0.598392`；ADD(-S) target-micro `0.380623`；
  macro-object `0.376867`

Region-conditioned low-order moments 没有形成有效 pose head，三项正式门槛均
失败。该结果只否定当前 CPM 压缩方式，不否定 dense correspondence 本身。
