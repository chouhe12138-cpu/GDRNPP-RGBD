# EXP-20260809-009 — CPM-Head

状态：`FIXED EPOCH 40 COMPLETE — CPM_SCREEN_FAIL`

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

EXP009 formal 在同一冻结 release、镜像和容器中启动，正式 run 为：

```text
RUN-20260811-063626-formal-s42-a01
```

## 中断记录与固定 Epoch 40 最终证据

早期训练日志记录到 Epoch 38 / iteration `243123/255920`，恢复日志最终记录
`CUDA error: unspecified launch failure`。这些中断和恢复失败证据继续保留，不能
从历史 `console.log` 中删除。

2026-08-17 对外部 EXP009 目录重新核验后，确认同一 formal run 已形成固定
Epoch 40 checkpoint 和完整 BOP 评估。checkpoint 信息为：

```text
path:      E:\\6D姿态估计\\EXP-009\\model_epoch_040.pth
size:      387752090 bytes
sha256:    d447569bf7a1034bb57f38c90ef25bbaac8f1bb7ef3b9d74ef9db75eb32f040d
epoch:     40
iteration: 255919
```

本地 `torch.load` 读取成功，包含 optimizer、scheduler 和 384 个模型张量。
当前没有服务器端原 checkpoint SHA-256，因此只记录外部副本完整可读，不记录
两端哈希一致。

固定 Epoch 40 BOP 结果为：

| 指标 | 结果 |
|---|---:|
| BOP AR | 0.5983921569 |
| BOP AR ad | 0.1750865052 |
| BOP AR mspd | 0.8433217993 |
| BOP AR mssd | 0.5456055363 |
| BOP AR reS | 0.4412918108 |
| BOP AR teS | 0.6920415225 |
| BOP AR vsd | 0.4062491349 |

`scores_bop19_40epoch.json` SHA-256 为
`2e9c8fa5e13118451d9dd8cdfc68e5f7cad351aa5b82185eb92cf700cb938448`；
`40epoch.log` SHA-256 为
`6bd26586475138a92180ca9057f02c261fbcf643ecd9ca71a03f7815ec330e8f`。
该日志是 E40 BOP evaluator 日志且未记录错误。重新下载的完整 `console.log`
SHA-256 为
`258be3940b53012abb5099ee4582a75923df306e2bba994917d82502e26547e0`；它记录训练
到 Epoch 40 / iteration `255919/255920`（100%）、保存
`model_epoch_040.pth`，以及 `FINAL_EVAL_REUSED periodic_epoch=40`。早期中断
和 CUDA 错误仍保留在同一日志中，后续恢复完成证据也已补齐。

固定 E40 BOP AR 相对官方基线 `0.6904152249` 降低 `9.2023 pp`，相对 mandatory
B control `0.6919123414` 降低 `9.3520 pp`。

完整 console 同时记录固定 E40 ADD(-S)@0.1d `0.3806228374`。代码核对确认
`EVAL_SUMMARY` 读取 BOP ADD score 的 `recall` 字段，因此该值是 target-micro，
早期记录曾误标为 macro-object。逐物体结果为：obj 1 `0.182857`、5 `0.603015`、6
`0.274854`、8 `0.670000`、9 `0.150000`、10 `0.277778`、11 `0.521429`、
12 `0.335000`；其等权 macro-object 为 `0.3768665461`。相对冻结 official
baseline 非负物体为 `2/8`。该聚合语义修正不改变 BOP、ADD(-S) 和逐物体三项
gate 均失败的结论。

Epoch 35 BOP AR `0.5994232987`、ADD(-S) target-micro `0.3861591696` 继续作为
中间结果保留。

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

上述干预结果不能区分学习率、moment 数值尺度、Region/XYZ 关系或低阶矩表达
中的哪一个因素造成了当前响应；记录中不据此指定根因。

## 结果与结论

`CPM_SCREEN_FAIL`。当前 CPM 实现固定 Epoch 40 的三项预注册 gate 均未通过，
但该结论只约束当前 Region-conditioned low-order joint-moment 实现及其训练
协议；不能
推出 2D–3D correspondence 本身不重要，也不能仅凭现有结果指定学习率、moment
尺度或低阶矩表达能力中的哪一个是根因。

固定 E40 的 EXP011 机制诊断已完成，预注册 decision 为
`MISMATCH_IMPORTANT`：在 Pred Region 下，GT-XYZ 对 BOP/macro ADD 的 effect 为
`-0.28948/-0.27965`；在 GT Region 下减弱为 `-0.13811/-0.10250`，interaction
为 `+0.15136/+0.17715`，rescue ratio 为 `0.5229/0.6335`，8/8 objects 的 ADD
interaction 为正。该结果支持 XYZ–Region 不一致是既有 GT-XYZ oracle 恶化的
重要污染因素，但 GT Region 在 Pred XYZ 下本身降低绝对性能，且 hard GT Region
与 soft Pred Region 的熵差异仍是混杂。因此不能把它写成可直接部署的性能改进，
也不能声称它是 CPM 欠佳的唯一根因。完整证据见 EXP011 RECORD。
