# EXP-20260809-009 — CPM-Head

状态：`FORMAL RUNNING — MANAGED GATE, SMOKE AND AUDIT PASS`

## 研究问题

在 backbone 和 geometry head 不变的前提下，在任何可学习压缩之前显式编码
可见支持上的 2D–3D 低阶联合统计，是否能让 direct pose head 更稳定地利用
XYZ 与 ROI 2D 的配对关系，并改善固定 Epoch 40 的姿态性能？

这是可证伪假设，不是已确认结论。CPM 只测试 Region-conditioned low-order
joint moments 是否足以改善几何消费；即使 CPM 失败，也不能推出 2D–3D
correspondence 本身不重要。

## 冻结协议

- 训练集 `lmo_pbr_train`，评估集 `lmo_bop_test`，GT bbox；
- 新 managed run 固定 seed `42`，40 epoch，正式比较只使用固定 Epoch 40；
- official checkpoint 初始化，冻结 backbone 与 geometry head，只训练 CPM；
- 复用 official `allo_rot6d`、`centroid_z`、pose conversion 与 loss；
- optimizer、schedule、augmentation 与 mandatory B 对照匹配；
- 不根据 LM-O test 中间 checkpoint 选择模型；
- 自动 best-checkpoint 保存关闭；中间评估只作为诊断曲线；
- 主指标为 BOP19 AR、ADD(-S)@0.1d macro-object 和 per-object；
- 机制验收复用 XYZ、ROI、Region、coverage-only、CXU-null 诊断。

## 本地执行证据

2026-08-09 已完成非正式 local chain：8192-sample moment audit、模型构建、
official 权重迁移、forward/backward、2048 micro-batch 短训、checkpoint 保存与
严格重载、完整 LM-O evaluator、BOP/ADD(-S) 索引及 8-target × 19-condition
diagnostic smoke（含 XYZ α、coverage-only、CXU-null）。
所有工程 gate 通过。local Epoch 1 的 BOP/ADD(-S) 均为 0，只证明链路可执行，
不作为方法性能证据，也不用于修改结构或超参数。

详细哈希和验收结果见 `research/cpm_head/LOCAL_CHAIN_20260809.json`。

## 服务器启动条件

1. C2 已完成 Epoch 40、最小归档和旧容器清理，该条件已满足；
2. mandatory B matched-training-protocol control 与 CPM 均使用 seed `42`、同一
   source commit 和同一 environment image，可分别在 lab0/lab1 启动；
3. formal 必须使用干净、已推送的 detached Git release，并绑定已验证的稳定
   environment image ID；两者分别记录，不要求 build-source commit 相等；
4. 新建 `lab1_chx` 后仍须依次通过 gate、smoke、batch-48 audit，才能启动 formal。

本地 checkpoint 只作为工程验收，不能当作正式初始结果或性能结果。

## 2026-08-11 managed server smoke（无效基础设施运行）

第一次服务器 smoke 的冻结身份为：

```text
run_id:                   RUN-20260811-052906-smoke-s42-a01
source_git_commit:        b39f68092de2609b7ee1726811c9ee965e606328
environment_image_id:     sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee
seed:                     42
```

release snapshot、17 个 native artifacts、CUDA/L40 环境、experiment registry
和 CPM preflight 均通过；CPM CUDA full forward 通过，参数隔离仍为预期的
`822,281` 个可训练参数。随后训练阶段因 dataset cache 指向只读 source release
而异常，且旧训练入口吞掉异常，使 postprocess 最终只看到缺失的
`model_epoch_001.pth`。

该 run 没有 checkpoint、指标或机制诊断结果，不构成 CPM 科学失败。原目录
保持不覆盖；修复可写 cache/home 和异常退出传播后创建新的唯一 run 重试。

## 2026-08-11 有效 managed 验收与正式启动

最终服务器身份为：

```text
source_git_commit:        652d7fd9d38f8ea5cea0c5a98cc9477b66623180
environment_image_id:     sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee
environment_build_source: 35313ae3d4139a559a97c01b2d3ee007dc16604c
seed:                     42
machine/container/GPU:    lab1 / lab1_chx / physical GPU 1
```

source snapshot、17 个 native artifacts、CUDA/L40、registry、CPM full-forward
与参数隔离 gate 全部 PASS。有效且非覆盖的验收 run 为：

```text
smoke: RUN-20260811-061226-smoke-s42-a01 — COMPLETE, exit 0
audit: RUN-20260811-062736-audit-s42-a01 — COMPLETE（用户回传）
```

smoke 完成 1 epoch / 2047 iterations，生成并记录
`checkpoints/model_epoch_001.pth`。audit 使用 batch-48 对应配置；两者只证明
训练与产物链可执行，不证明 CPM 的核心假设成立。

用户随后确认 EXP009 formal 已在同一冻结 release、镜像和容器中启动。当前交接
尚无 formal `run_id`、后台 PID 或当前 epoch，因此实验 metadata 只记录
`RUNNING`，精确身份必须在后续从只读服务器状态补齐。

## Epoch 30 高覆盖预览诊断（非正式结论）

外部 Epoch 30 checkpoint 完整可读，大小 `387752090` bytes，SHA-256：

```text
d5fabd8ad3f2be5ecf3fcc52a18386d151732f7593a0daa2ca22181c0add5ce0
```

它包含 model、optimizer 和 scheduler，记录 Epoch 30 / iteration 191939。基于
1,445 targets × 19 conditions 的预览诊断位于 Git 忽略的
`output/EXP-20260809-009-cpm-head/diagnostic/E30_PREVIEW/full`。baseline BOP AR
`0.5994556` 与训练时 Epoch 30 的 `0.5994625` 一致。该 run 仅因 baseline
translation re-entry 最大误差 `0.000320 m` 高于旧 QC 容差 `0.00005 m` 被标记
QC FAILED；样本数量完整、没有意外非有限值且诊断未改变 checkpoint，因此可作
机制预览，但不能替代固定 Epoch 40 正式诊断。

关键 BOP AR：

| 条件 | BOP AR |
|---|---:|
| baseline | 0.59946 |
| GT-XYZ alpha 0.25 / 0.50 / 0.75 / 1.00 | 0.56205 / 0.46897 / 0.37097 / 0.30820 |
| XYZ permutation | 0.14534 |
| ROI permutation | 0.37114 |
| Region disruption / mean Region | 0.18668 / 0.18765 |
| coverage-only | 0.00000 |
| CXU-null | 0.15855 |

直接支持的事实是：当前 CPM 确实依赖 XYZ、ROI、object-space Region partition
和二阶 cross-covariance；coverage-only 不能维持 pose；更准确 XYZ 没有被稳定
转化为更好 pose，反而随 alpha 增大持续变差。它不直接证明学习率、moment
尺度、Region/XYZ 不一致或低阶矩表达不足中的哪一个是根因。

EXP009 使用 `8e-5`，而 CPM 头是新随机初始化的 822,281 参数模块；因此“学习率
过低导致优化不足”是合理但未证明的工程假设。EXP010 作为严格匹配控制，只把
学习率改为 `8e-4`，用于隔离这个解释。若 EXP010 仍不能改善固定 Epoch 40 的
性能和机制响应，应停止用学习率解释 EXP009，再检查 joint-moment 表达与
Region-conditioned aggregation 本身。

早期某次 Epoch 35 传输尚未完成时不能读取；后续完整文件为 `387752090` bytes，
SHA-256 `44129bc8ebd32bb99627fcd4170138a2afd3b32bdf17c39d0236724a81d4b196`，
并已用于预览诊断。它仍不替代固定 Epoch 40 正式结果。

## 结果与结论

`PENDING FORMAL EPOCH 40`。不得从 smoke/audit loss、LM-O 中间 checkpoint 或
当前 GPU 占用推导方法有效性。正式 Epoch 40 完成后，必须同时核验 BOP、
ADD(-S)、per-object 与预注册信息流诊断；CPM 失败时的结论边界仍仅限于“当前
Region-conditioned low-order joint moments 不足”，不能推出 2D–3D
correspondence 不重要。
