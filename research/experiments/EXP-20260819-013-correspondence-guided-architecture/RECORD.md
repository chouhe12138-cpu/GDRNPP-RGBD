# EXP-20260819-013 Correspondence-Guided 轻量 6D Pose 架构筛选

## 当前状态

- 状态：`PLANNED / ARCHITECTURE_SCREENING_OPEN / NO_FORMAL_40E`。
- EXP013 是完整架构阶段的起点，不是已经锁定实现的单一 pose-head 实验。
- 当前没有 EXP013 source commit、训练 checkpoint 或性能结果；在实现冻结前不得把候选结构写成实验结论。
- 40-epoch 全量 PBR 当前未授权。优先 small-scale screening，再决定是否进入正式长训练。

## 核心研究问题

在已知 dense 2D–3D correspondence 包含强姿态信息、但 learned pose head 未必能稳定利用的前提下，重新设计从视觉输入到 pose 的完整信息流：

```text
RGB -> lightweight visual features -> 2D-3D correspondence -> pose
```

目标不是继续修补 GDRNPP pose head，而是在 LM-O 上寻找可复现的 accuracy–efficiency trade-off，并为后续真实机器人验证保留部署可行性。

## 两条体系化候选

### A：显式几何 reference path

```text
Light Backbone
  -> correspondence + reliability/coverage
  -> weighted PnP
  -> R,t
```

A 的作用首先是几何基准：判断同一 correspondence frontend 在不经过 learned pose decoder 时能达到什么精度和稳定性。PnP/RANSAC/迭代求解是否适合最终部署不预设结论，必须同时测 GPU forward latency 与端到端 latency，包括可能的数据搬运和 solver 开销。

### B：correspondence-guided asymmetric R/t direct regression

```text
Light Backbone
  -> correspondence
  -> rotation-specific representation -> R
  -> translation-specific representation -> t
  -> projection / geometric consistency
```

这里的“解耦”不是只把最后两个 FC 拆开。Rotation 主要读取 normalized/canonicalized correspondence、物体内部和相对几何；Translation 主动保留 absolute 2D、ROI/bbox 尺度、相机内参 K、object extent 等与位置和深度直接相关的 metric cues。

R/t 在透视几何上并非真正独立，因此不禁止所有通信；若需要 cross-branch coupling，必须具有明确几何含义并单独验证。Geometric consistency 负责让 direct R,t 能解释预测 correspondence，但不能代替 GT XYZ / GT pose 监督，避免两端形成“彼此一致但都错误”的解。

## 设计原则

- correspondence 是核心信息载体，不在 pose 前过早压成低阶 moment 或不可解释的单一共享 latent。
- Backbone、Geometry/Correspondence representation、Region、Mask、Pose decoder 都重新按信息需求和计算成本审视；不因 GDRNPP 已存在而默认保留。
- Region 默认不进入主链；Visible/support 信号只保留 correspondence 构造实际需要的最小形式；Full/amodal Mask 不默认保留。
- reliability 与 coverage 只有在解决明确的 correspondence 质量/几何分布问题时才加入，不作为独立 trick 堆叠。
- 所有模块必须能回答“它解决 correspondence→pose 链上的哪个问题”。回答不清楚则不加入。

## 证据边界

**已被实验直接支持：** dense XYZ correspondence 含有强 pose 信息；CPM 当前 Region-conditioned low-order moment 路线失败；EXP012 E5/E10 较好、E15 严重后期退化且主要集中于 rotation；EXP012 强依赖 Region；Pred XYZ→GT XYZ 与 synced Region 均不能稳定 rescue learned pose；E10↔E15 checkpoint interpolation endpoints/QC PASS，并显示 BOP/AR_reS 连续恶化。Activation drift QC FAIL，只作线索。

**合理但尚未证实：** R/t 的学习表示需要非对称设计；R/t 共享后期 latent 可能带来优化干扰；geometric consistency 可能改善 direct decoder 对 correspondence 的真实利用；轻量 frontend 能在大幅降算力的同时保留足够 correspondence 质量。

**当前缺失的关键证据：** A/B 在同一 frontend 下的真实 accuracy–latency 对比；R/t factorization 是否改善后期 rotation stability；轻量 correspondence representation、reliability/coverage 的必要性和最小成本；Region/Full Mask 在新体系中是否确有增益。

## Screening 协议

第一阶段不直接跑 40 epoch。优先 E1/E3/E5 或等价小规模检查，并尽可能让 A/B 共用同一 correspondence frontend。必须同时记录 BOP AR、AR_reS、AR_teS、ADD(-S)、逐物体、参数量、FLOPs/MACs、peak VRAM、纯 GPU forward latency 和 end-to-end latency。

A/B 的诊断解释预先固定：A 好/B 差说明 learned mapping 仍是主要瓶颈；A/B 都差先修 frontend；A 差/B 好必须排查 direct branch 是否利用了合理额外线索还是 shortcut；A/B 都好才进入真实 accuracy–efficiency Pareto 选择。

## 下一步

先冻结 EXP013 第一版最小 shared frontend 与 A/B 接口，再写代码和 screening config。未经小规模结果支持，不扩展到 Transformer、复杂 Region、更多 auxiliary branches 或 40-epoch 全量 PBR。
