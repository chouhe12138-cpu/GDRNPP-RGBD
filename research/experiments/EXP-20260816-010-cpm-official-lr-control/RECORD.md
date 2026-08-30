# EXP-20260816-010 — CPM 官方量级学习率受控实验

状态：`FAILED — formal run 于 ~E27 崩溃，用户决定不重试（2026-08-30）`

## 研究问题

EXP009 将一个随机初始化的 CPM 姿态头接到冻结的官方 backbone 与 geometry
head 上，但沿用了 B 对已存在 Patch-PnP 做适应训练时的 `8e-5` 学习率。官方
新姿态头训练使用的量级为 `8e-4`。因此，EXP009 的结果可能同时混合了“CPM
信息组织是否有效”和“新头是否被充分优化”两个因素。

EXP010 只回答一个可证伪问题：

> 在完全相同的 CPM 结构、初始化、数据、损失、训练长度和评估协议下，把学习率
> 从 `8e-5` 改为 `8e-4`，是否能显著改善固定 Epoch 40 的性能与信息流行为？

## 唯一变量

```text
EXP009: Ranger lr = 8e-5
EXP010: Ranger lr = 8e-4
```

允许的 resolved-config 差异只有：

```text
EXPERIMENT_ID
OUTPUT_DIR
SOLVER.OPTIMIZER_CFG.lr
```

其余 CPM 网络、固定 moment scales、official loss、warmup 200、batch 48、
40 epochs、seed 42、数据、评估和诊断协议均保持一致。EXP010 从官方 checkpoint
重新初始化，禁止从 EXP009 checkpoint resume。

## 当前证据与结论边界

- EXP009 固定 Epoch 40 checkpoint 完整且可读，SHA-256 为
  `d447569bf7a1034bb57f38c90ef25bbaac8f1bb7ef3b9d74ef9db75eb32f040d`；
  固定 E40 BOP AR 为 `0.5983921569`，未通过 BOP gate。
- Epoch 30 高覆盖预览诊断表明 CPM 确实使用 XYZ、ROI、Region 与 `CXU`，且
  coverage-only 退化为零；但更准确 XYZ 没有被转化为更好 pose，反而随 alpha
  增大持续下降。
- EXP009 固定 E40 的失败 BOP 结果不能单独证明学习率就是原因。EXP010 才是
  对优化不足解释的匹配控制。
- EXP009 固定 E40 ADD(-S) target-micro 为 `0.3806228374`，由逐物体值派生的
  macro-object 为 `0.3768665461`，逐物体非负 `2/8`；EXP011 固定 E40 机制
  诊断已完成，结论 `MISMATCH_IMPORTANT`。

EXP009 的固定 E40 BOP 结果已具备，EXP010 启动不再等待 EXP009。最终匹配
比较仍等待 EXP010 固定 Epoch 40，并需补齐同协议指标与机制诊断。

## 启动授权

EXP005/B 已完成固定 Epoch 40，用户已明确授权 EXP010 在 lab0 启动。EXP010
config diff 必须继续只包含身份、输出目录和学习率。新 run 创建确定 Git release，
复用稳定 environment image，依次通过
access、create、gate、smoke、audit48，再启动 formal。不得修改或复用 EXP009
的运行目录。

已回传的服务器 access 证据为：

```text
machine:              lab0
physical GPU:         0
source_git_commit:    29580f65abfeb7625bab252011c19399325b0fa2
environment_image_id: sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee
access:               PASS
```

后经 2026-08-30 核对，`console.log`（本地 `E:\6D姿态估计\EXP-010\`）确认存在
formal run `RUN-20260816-081032-formal-s42-a01`（2026-08-16 启动，commit
`29580f65...`、稳定环境镜像、seed 42、lr 8e-4），本节早先"无 formal run 证据"
的表述已过时，见下方"2026-08-30 崩溃记录"。

## 成功、失败与后续

- 若 EXP010 相对 EXP009 同时明显改善固定 Epoch 40 性能和预注册机制响应，说明
  EXP009 存在优化不足，后续 CPM 判断应以匹配学习率结果为准。
- 若只降低训练 loss 而最终性能/机制不改善，则不支持“低学习率是主要原因”。
- 若出现发散、非有限值或明显不稳定，记录为高学习率不适配，不通过事后改变
  warmup、loss 或网络结构修补本实验。
- 无论结果如何，EXP010 都不能证明或否定完整 2D–3D correspondence 路线；它
  只控制当前 CPM 低阶联合矩实现的优化条件。

## 2026-08-30 崩溃记录与关闭决定（保留证据）

- formal run：`RUN-20260816-081032-formal-s42-a01`，lab0，seed 42，lr 8e-4，
  2026-08-16 启动。`console.log` 末尾为 epoch 27 / iteration
  `166999/255920`（65.3%，2026-08-18），此后崩溃，没有固定 Epoch 40
  checkpoint 或最终评估。
- 已完成的正式评估（EVAL_SUMMARY）：

| Epoch | BOP AR | add_s_0.1d |
|---:|---:|---:|
| 5 | 0.496205 | 0.235986 |
| 10 | 0.540775 | 0.328028 |
| 15 | 0.567045 | 0.333564 |
| 20 | 0.513548 | 0.287889 |
| 25 | 0.533396 | 0.303806 |

- 训练轨迹：E15 为已观测峰值（BOP AR `0.567045`），E20 回落，E21 起 loss 回升
  （loss_PM_R 均值由 `0.0148` 升至 `0.0153+`、total 由 `0.167` 升至 `0.187`），
  呈现高学习率不稳迹象。
- 方向性解读（不是 gate 结论）：EXP009（lr 8e-5）在 E30 已到
  `0.5994625144`，高于 EXP010 全程已观测峰值 `0.567045`；10 倍学习率既没有
  让 CPM 更快超过 EXP009 的 E30–E40 平台，还伴随中期评估回落与 loss 回升。
  这与"CPM 失败主要来自优化不足（学习率过低）"的解释不一致。
- 由于没有固定 Epoch 40 结果，预注册的匹配学习率比较在形式上未完成，只能保留
  上述方向性证据。
- 用户决定（2026-08-30）：实验已崩溃，不再重试、不继续消耗算力。EXP010 关闭，
  状态 `FAILED`；E5/E10/E15/E20/E25 评估与崩溃日志保留为证据。
