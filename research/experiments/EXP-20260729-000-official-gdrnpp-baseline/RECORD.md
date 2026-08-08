# EXP-20260729-000 — 官方 GDRNPP 基线

本实验身份补录自历史官方 checkpoint 验证，不代表一次新的训练。冻结协议为
LM-O BOP19 test、GT bbox、官方 ConvNeXt-Base checkpoint、Patch-PnP 直接姿态。

完整 1,445-target 基线推理已在 EXP-20260804-007 的 official/full 中按同协议
复现，BOP AR 与 ADD(-S) 均通过其质量门。该共享产物保持原位，不复制到新的
EXP000 output。

ADD(-S) 的 micro-target 与历史 C1 记录中的 macro-object 数值不是同一聚合
口径，后续比较必须携带完整指标 ID。
