# EXP-20260816-010 — CPM 官方量级学习率受控实验

状态：`AUTHORIZED — 等待新 Git release 在 lab0 执行`

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

- EXP009 Epoch 30 checkpoint 完整且可读，SHA-256 为
  `d5fabd8ad3f2be5ecf3fcc52a18386d151732f7593a0daa2ca22181c0add5ce0`。
- Epoch 30 高覆盖预览诊断表明 CPM 确实使用 XYZ、ROI、Region 与 `CXU`，且
  coverage-only 退化为零；但更准确 XYZ 没有被转化为更好 pose，反而随 alpha
  增大持续下降。
- 这能说明 EXP009 当前表现不理想，不能单独证明学习率就是原因。EXP010 才是
  对优化不足解释的匹配控制。
- EXP009 的完整 Epoch 35 权重已完成预览诊断；它仍不能替代固定 Epoch 40。

EXP009 的中间预览结果不是最终结论，但也不是 EXP010 启动的依赖。EXP009 与
EXP010 可以分别在 lab1/GPU 1 和 lab0/GPU 0 并行；最终比较仍等待两者固定
Epoch 40。

## 启动授权

EXP005/B 已完成固定 Epoch 40，用户已明确授权 EXP010 在 lab0 启动。EXP010
config diff 必须继续只包含身份、输出目录和学习率。新 run 创建确定 Git release，
复用稳定 environment image，依次通过
access、create、gate、smoke、audit48，再启动 formal。不得修改或复用 EXP009
的运行目录。

## 成功、失败与后续

- 若 EXP010 相对 EXP009 同时明显改善固定 Epoch 40 性能和预注册机制响应，说明
  EXP009 存在优化不足，后续 CPM 判断应以匹配学习率结果为准。
- 若只降低训练 loss 而最终性能/机制不改善，则不支持“低学习率是主要原因”。
- 若出现发散、非有限值或明显不稳定，记录为高学习率不适配，不通过事后改变
  warmup、loss 或网络结构修补本实验。
- 无论结果如何，EXP010 都不能证明或否定完整 2D–3D correspondence 路线；它
  只控制当前 CPM 低阶联合矩实现的优化条件。
