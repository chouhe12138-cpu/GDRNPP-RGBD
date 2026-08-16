# GDRNPP 研究交接

最后更新：2026-08-16

本文件只提供新对话需要的当前摘要。稳定工作规则见根目录 `AGENTS.md`，当前
研究事实见 `research/STATUS_CN.md`，服务器动态事实见
`research/SERVER_RUNTIME_STATUS_CN.md`。不要用本摘要替代原始实验产物、run
manifest、实验 `RECORD.md` 或 stage 协议。

## 接管顺序

```bash
cd /home/wsluser/GDRNPP-RGBD
git status --short --branch
git rev-parse HEAD
```

然后依次阅读：

```text
AGENTS.md
README_CN.md
research/STATUS_CN.md
research/SERVER_RUNTIME_STATUS_CN.md
research/RESEARCH_PLAN.md
research/POSE_HEAD_DIAGNOSTIC_HANDOFF_CN.md
research/STAGE_03C_EXPERIMENT_OVERVIEW.md
```

只有追溯镜像构建、旧容器迁移或 C1 历史环境时，才读取
`research/SERVER_MIGRATION_HANDOFF.md`。

## 当前研究状态

```text
Stage 1:   COMPLETE — FAIL
Stage 2:   COMPLETE — PASS（XYZ GEOMETRY）
Stage 3A:  COMPLETE — CALIBRATION_MISMATCH
Stage 3B:  COMPLETE — PATCH_PNP_UNDERUTILIZATION
Stage 3C0: EXP005/B FORMAL COMPLETE
Stage 3C1: FORMAL COMPLETE — C1_SCREEN_FAIL
Stage 3C2: FORMAL COMPLETE — C2_SCREEN_FAIL
Stage 4:   EXP009/CPM FORMAL RUNNING
Stage 4C:  EXP010/CPM-LR CONTROL AUTHORIZED
```

已经由实验直接支持的主线事实是：预测 XYZ 是重要因果瓶颈；官方 direct
Patch-PnP 无法稳定把受控改善的 XYZ 转化为更好的 `R,t`；官方姿态头的信息流
诊断还显示 Region 输入占主导、XYZ 响应不稳定，ROI 2D 配对信息在现有压缩链中
快速衰减。对 shortcut 或具体内部机制的解释仍是机制假设，不是事实。

C1 和 C2 均已完成并未通过冻结筛选门槛。C2 固定 Epoch 40 BOP AR 为
`0.6930057670`，历史 ADD(-S) 证据缺失，不能用 BOP `ad` 替代。EXP005/B 是
mandatory matched-training-protocol 对照；EXP009/CPM 测试 Region-conditioned
低阶 2D–3D 联合矩是否足以改善 direct pose regression 的几何消费。

## 当前正式实验

两项正式实验均固定 seed `42`、40 epoch，并只以 Epoch 40 作正式比较：

| 实验 | 账户/物理 GPU | 容器 | 当前状态 |
|---|---|---|---|
| EXP005/B | lab0 / GPU 0 | `lab0_chx` | formal 完成；E40 BOP 0.691912、ADD(-S) 0.506574 |
| EXP009/CPM | lab1 / GPU 1 | `lab1_chx` | gate、smoke、audit PASS；formal 运行中 |

冻结 source release：

```text
652d7fd9d38f8ea5cea0c5a98cc9477b66623180
```

共用稳定 environment image：

```text
sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee
```

image build-source 为 `35313ae3d4139a559a97c01b2d3ee007dc16604c`；它与实验
source commit 分别记录，不要求相等。

有效受管运行：

```text
EXP005 smoke: RUN-20260811-061212-smoke-s42-a01
EXP005 audit: RUN-20260811-062719-audit-s42-a01
EXP009 smoke: RUN-20260811-061226-smoke-s42-a01
EXP009 audit: RUN-20260811-062736-audit-s42-a01
```

EXP005 formal run 已确认为 `RUN-20260811-063606-formal-s42-a01`；EXP009 最终
状态仍须从服务器只读指针和 run metadata 核验。

## 必须保留的失败证据

以下两个 `b39f680...` smoke 因 dataset cache 写入只读 source 失败，且旧入口
吞掉异常，最终表现为缺少 checkpoint：

```text
EXP005: RUN-20260811-052852-smoke-s42-a01
EXP009: RUN-20260811-052906-smoke-s42-a01
```

它们没有 checkpoint、指标或科学结论，只是无效基础设施运行，不能覆盖、删除
或解释为模型失败。`dcf6d57...` 的首次容器创建也因缺少只读 source 内的嵌套
`.cache` mountpoint 而在容器进程启动前失败，没有生成实验 run。最终修复位于
`652d7fd...`，无需重建 environment image。

## 训练结束后的下一步

1. 先只读核验两项 formal 的 `run_id`、source commit、image ID、resolved
   config、seed、run state、epoch 和 checkpoint 清单。
2. 等固定 Epoch 40 checkpoint 完成后，分别运行明确 checkpoint 的 eval；不从
   LM-O test 中间 checkpoint 选择最佳模型。
3. 保存 BOP AR、ADD(-S)@0.1d macro-object、per-object 和结构化指标；原始结果
   不覆盖，重新核验结果单独记录。
4. 对 EXP005、EXP009 运行同协议信息流诊断，再判断 CPM 的机制假设；最终精度
   不能代替机制验收。
5. 所有结论以原始产物与哈希、run manifest、标准化指标、实验 RECORD、stage
   协议的顺序解决冲突。

EXP010 已建立并获准作为 EXP009 的严格优化控制。其 CPM 结构、数据、
official 初始化、loss、warmup、seed 与 40-epoch 协议均继承 EXP009，只把 Ranger
学习率从 `8e-5` 改为 `8e-4`。它可在 lab0 与 lab1 的 EXP009 并行运行；最终
结论必须等待两者固定 Epoch 40。

正式训练期间不得 pull 或修改 release，不得重建/替换容器和镜像，不得覆盖
output，也不得停止其他用户进程。用户在服务器终端执行命令并回传输出；Agent
不主动 SSH。

## 本地验证基线

- 当前正式源码提交：`652d7fd9d38f8ea5cea0c5a98cc9477b66623180`。
- 本地研究测试基线：`147 passed`。
- 官方初始化权重 SHA-256：
  `bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c`。
- EXP009 参数量：`822,281`；本地完整工程链已通过，但不构成方法有效性结论。
