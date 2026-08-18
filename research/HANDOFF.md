# GDRNPP 研究交接

快照日期：2026-08-18

本文件保留 2026-08-18 时点的研究交接快照，用于追溯当时的上下文，不再承担
“当前状态”的唯一来源。后续实验推进后，不要求同步改写本文件中的每一项实验
编号、指标或服务器事实。

当前研究事实见 `research/STATUS_CN.md`，完整实验列表见
`research/EXPERIMENT_INDEX.md`，服务器动态事实见
`research/SERVER_RUNTIME_STATUS_CN.md`，稳定工作规则见根目录 `AGENTS.md`。
若本快照与上述入口或具体实验记录冲突，以原始产物及哈希、run manifest、
标准化指标、实验 `RECORD.md`、stage 协议和当前状态摘要的证据层级处理。

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
Stage 4:   EXP009/CPM FIXED EPOCH 40 COMPLETE — CPM_SCREEN_FAIL
Stage 4C:  EXP010/CPM-LR CONTROL AUTHORIZED
Stage 4D:  EXP011/CPM XYZ-REGION DIAGNOSTIC COMPLETE — MISMATCH_IMPORTANT
Stage 4E:  EXP012/HIERARCHICAL CORRESPONDENCE HEAD AUTHORIZED — LOCAL PREFLIGHT PASS
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
| EXP009/CPM | lab1 / GPU 1 | `lab1_chx` | E40完成；BOP 0.598392、ADD(-S) 0.380623；筛选失败 |

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

EXP005 formal run 已确认为 `RUN-20260811-063606-formal-s42-a01`；EXP009 formal
run 为 `RUN-20260811-063626-formal-s42-a01`，外部固定 E40 checkpoint 已读取并
完成 BOP 与 ADD(-S) 评估。服务器实时状态仍须重新只读核验。

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

## 固定 Epoch 40 后的下一步

1. 继续只读核验两项 formal 的 run state、checkpoint 清单和服务器原文件哈希；
   当前外部副本哈希不写成服务器两端一致。
2. 对 EXP005、EXP009 运行同协议信息流诊断，再判断 CPM 的机制假设；最终精度
   不能代替机制验收。
3. 完成 EXP010 固定 E40 后进行匹配学习率比较。
4. 所有结论以原始产物与哈希、run manifest、标准化指标、实验 RECORD、stage
   协议的顺序解决冲突。

EXP010 已建立并获准作为 EXP009 的严格优化控制。其 CPM 结构、数据、
official 初始化、loss、warmup、seed 与 40-epoch 协议均继承 EXP009，只把 Ranger
学习率从 `8e-5` 改为 `8e-4`。EXP009 固定 E40 已具备；最终匹配结论等待
EXP010 固定 E40 和同协议诊断。

EXP012 已注册为下一代 correspondence-preserving pose head。它保留逐像素
XYZ↔ROI2D，在全局压缩前进行局部与层级编码；Region 仅为零启动辅助残差，不
定义 pooling。当前仅有本地 CPU/CUDA preflight、单步 optimizer 和 strict checkpoint
roundtrip PASS，没有任何服务器 run 或性能结论。原始协议见 EXP012 RECORD。

正式训练期间不得 pull 或修改 release，不得重建/替换容器和镜像，不得覆盖
output，也不得停止其他用户进程。用户在服务器终端执行命令并回传输出；Agent
不主动 SSH。

## 本地验证基线

- 当前正式源码提交：`652d7fd9d38f8ea5cea0c5a98cc9477b66623180`。
- 本地研究测试基线：`147 passed`。
- 官方初始化权重 SHA-256：
  `bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c`。
- EXP009 参数量：`822,281`；本地完整工程链已通过，但不构成方法有效性结论。
