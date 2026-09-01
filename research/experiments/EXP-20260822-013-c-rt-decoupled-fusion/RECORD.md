# EXP013C 运行记录

- 最终状态：`COMPLETE / SCREEN_FAIL / ROTATION_SUPPORTED`。
- 原始 B-based C 未运行。B E40 严格 gate 失败后，用户提供的新设计在正式运行前明确改为继承 A；该变化记录为 protocol revision 2。
- 当前本地实现不包含 B attention；R/t 使用独立聚合、latent、输出和 `geometry_scale_r/t`。
- 工程优化：冻结 Geometry Head 时保留预测 Geometry forward 与 online mapper，但跳过 renderer 和不会参与训练的 GT geometry supervision。

## 2026-08-26 本地验证

- 压缩包 SHA-256：`a5bae9653f3eeb5668b5104471617913b461adc89fad2b980b9c0949b42dff14`。
- 源码基线：`6b272bdfa5996ad8e09c7fe4bd5a48e375baac91`，测试时工作树包含未提交的 E40 诊断记录和本次候选修改。
- 静态验证、配置 merge 和 `git diff --check`：PASS。
- 单元与相关回归：27 tests PASS。
- C CPU preflight：官方共享权重迁移、full forward/backward/optimizer step、strict checkpoint roundtrip PASS。
- C CUDA FP32 preflight：full forward/backward/optimizer step PASS；仅 `pnp_net.*` 可训练；trainable params `1,549,724`；head FLOPs `221,671,424`；batch1 latency `2.9973 ms`；peak allocated VRAM `563,791,872 bytes`。
- A/B CPU preflight 均 PASS，证明旧配置默认 supervision/renderer 语义未被改成 no-render。
- 真实数据 smoke：`lmo_pbr_stage3_local_train`，batch 4，2 workers，2048/2048 iterations，独立输出目录；未创建 CPP/EGL renderer，forward/backward/optimizer 全部完成。
- smoke E1：final total loss `2.074373`，running mean `2.099`，checkpoint epoch 1 / iteration 2047。
- smoke checkpoint SHA-256：`d7334c3c70b6f851a62ef517cb0263bafd328b7f5d2c10cddf6f2568efa71c02`。
- checkpoint isolation：PASS，非 PnP tensor 逐值不变；`geometry_scale_r=0.101193`、`geometry_scale_t=0.099384`，均从 0.1 获得有限更新。

2026-08-26 用户在审阅本地 gate 后明确要求提交代码、生成 bundle，并准备在 lab0 训练，因此 revised C 已授权进入服务器完整 `access→create→gate→smoke→audit48→launch` 流程。授权不改变 protocol revision 2，也不把它记作原始 B-based C。

## Formal E40 完成与判决

- formal run：`RUN-20260826-124748-formal-s42-a01`，lab0，seed 42，source commit
  `d702030c65347175aead005d046edc4f5e8bdd83`。
- environment image：`sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee`；
  `MANAGED_RUN_FINISH status=PASS`（2026-08-28T21:30:42Z）。
- 固定 E40：BOP AR `0.6846459054`、ADD(-S) target-micro `0.4968858131`、
  macro-object `0.4988410279`、AR_reS `0.5250288351`、AR_teS `0.7942329873`。
- 相对 A E40，reS 提高 `0.026990`，translation 下降 `0.003460`，均通过 C 的
  rotation 与 translation-drop 条件；但 ADD(-S) target-micro 下降 `0.013841`，
  因此总体筛选未通过。结论为 `SCREEN_FAIL / ROTATION_SUPPORTED`，支持 R/t
  专用表示改善旋转读出，不支持其已形成整体更优的 pose head。
- E40 checkpoint：`399495556` bytes，epoch 40 / iteration `255919`，466 个模型
  张量，含 optimizer 与 scheduler；SHA-256
  `21ae8b074051fbfa7512684e2b623cb29529faa0c29f2e682171fb6fa3f6adf0`。
- console SHA-256：`2b6e0a2e66ab7f6ca3b8e217b42aded9ce6e8aa6bbe0001ede07c9441bd83f6e`；
  E40 score SHA-256：`4fd692ed6e41939a4e45205a02346f8e8e82cc162cf8367d9aadd3f39303384d`。
- 外置证据位于 `E:\6D姿态估计\EXP-013\实验C\`，不进入 Git。
