# Solver-in-the-loop 对应关系学习：文献与设计 Review

- 日期：2026-09-08。
- 性质：literature / design review。本页不产生新的实验指标，实验事实以各实验
  `RECORD.md` 为准；本页只整理文献证据、机制推理与口径建议，并按
  `AGENTS.md` 的证据分层标注 Observed / Derived / Interpretation / Decision。
- 触发：用户 2026-09-08 review 判定 EXP019 机制通过，并确定新的研究主线
  （可微 EPro-PnP 反传约束 Geometry/Correspondence Head）。决策记录见
  [DECISIONS](../DECISIONS.md)。

## 0. 摘要

1. **EXP019 机制结论成立**：α 从 0→1 时 EPro-PnP 的 ADD(-S)/BOP AR 从
   `0.5433/0.7009` 单调升到 `0.9924/0.9924`（Spearman≈1.0，相对 matched RANSAC
   的 recovery `0.992/0.967`），而同一份数据中 Patch-PnP 几乎不响应
   （`0.5066→0.4997` / `0.6904→0.6837`）。
2. **方法骨架已被 EPRO-GDR 占位**：GDRNPP + EPro-PnP 端到端训练、w²D 可靠性头、
   PBR-only 训练已在 EPRO-GDR（arXiv 2409.11819，AIxVR 2025）中实现，方法层面
   重合约 60–70%；但其科学问题与实验设计与本项目重合约 20–30%。
3. **EPRO-GDR 的空白即本项目的入口**：它没有消融、没有任何对应关系质量指标、
   T-LESS 崩塌（object 27 从 0.843 掉到 0.125）、不是 SOTA、未涉及 domain gap；
   且它自述预测点"disorganized"、模型"does not fully comprehend the entire shape
   of the object"。
4. **三条独立证据表明"只反传姿态损失"不足以让 head 学到几何**（见第 4 节）：
   EPro-PnP 自己的 Table 3、Linear-Covariance Loss 的"平均化"论证、EPRO-GDR 的
   定性观察。这支持"需要对应关系级约束"的方法学假设。
5. **跨域方向存在明确文献空白**：检索未找到任何"同一架构、只替换 correspondence
   head vs pose head、在同一跨域 benchmark 上对比"的受控实验（见第 6 节）。
6. **Gate 口径**：0.001 绝对容差在 1,445 targets 上约等于 1.44 个 target；本次
   六点偏差折算 2–6 个 target（0.14%–0.83% 相对），低于二项噪声（SE 约 1.8%–2.6%
   相对）。用户决定后续改用相对阈值（±3%–±5%），接近饱和的指标仍用绝对阈值。

## 1. 本仓库已有证据（Observed，摘录）

EXP004（`EXP-20260731-004`，官方 checkpoint、LM-O BOP19、GT bbox、无训练）：

| α | Patch ADD | Patch BOP | RANSAC ADD | RANSAC BOP |
|---:|---:|---:|---:|---:|
| 0.00 | 0.50242 | 0.69021 | 0.53841 | 0.69255 |
| 1.00 | 0.49550 | 0.68324 | 0.99377 | 0.99377 |

EXP019 full run（`RUN-20260907-182144-full-s20260730`，commit `69e0e8a`，1,445
targets，7,225 行，无训练；Patch/RANSAC/EPro 同 producer、同 `xyz_alpha`、同
fixed support）：

| α | Patch ADD | Patch BOP | RANSAC ADD | RANSAC BOP | EPro ADD | EPro BOP |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.5066 | 0.6904 | 0.5398 | 0.6909 | 0.5433 | 0.7009 |
| 0.25 | 0.5003 | 0.6899 | 0.6166 | 0.7176 | 0.6221 | 0.7298 |
| 0.50 | 0.4990 | 0.6897 | 0.7398 | 0.7831 | 0.7550 | 0.7899 |
| 0.75 | 0.4962 | 0.6898 | 0.8519 | 0.8539 | 0.8595 | 0.8647 |
| 1.00 | 0.4997 | 0.6837 | 0.9924 | 0.9924 | 0.9924 | 0.9924 |

原始 Gate A/B 通过；历史复现 gate 6 点中 5 点超出原 0.001 容差；原 evaluator
decision 为 `PROTOCOL_REPRODUCTION_FAILED_STOP`（见 EXP019 RECORD）。EPro-PnP 在
每个 α<1 上都不低于 matched RANSAC（如 α=0.5：ADD `0.7550` vs `0.7398`）。

## 2. EPRO-GDR 事实卡（Observed，多源核对）

- 论文：Pöllabauer, Li, Knauthe, Berkei, Kuijper, *End-to-End Probabilistic
  Geometry-Guided Regression for 6DoF Object Pose Estimation*，arXiv:2409.11819
  （2024-09-18，仅 v1），AIxVR 2025, pp. 35–43。
- 方法：保留 GDRNPP 的 YOLOX 检测器、ConvNeXt 骨干、visible/amodal mask 头、
  稠密 3D 坐标头（x³D）、surface region attention（SRA）头；把 Patch-PnP 替换为
  EPro-PnP（"replace GDRNPPs Patch-PnnP algorithm with EPro-PnnP"）；新增预测
  2D 权重 w²D 的头（CDPN 来源 + spatial Softmax + global scaling）；移除
  translation head。
- 训练：两阶段。阶段 1 用 Patch-PnP 训到收敛；阶段 2 换 EPro-PnP 继续训练，理由
  是 "EPRO-GDR benefits from good initialization"。阶段 2 损失：KL 权重 0.2、
  rotation angular loss 权重 1.0（GDRNPP 配置中该损失被设为 0）、SRA 权重从
  1.0 降到 0.005；400 步 warm-up；batch 48→72；Ranger，lr 8e-4，wd 0.01，
  40 epoch + early stopping。训练数据只用 BOP 的 BlenderProc PBR 合成图像。
- 结果（AR_BOP 均值）：

| 数据集 | GDRNPP | EPRO-GDR | SurfEmb | PFA |
|---|---:|---:|---:|---:|
| LM-O | 0.757 | 0.786 | 0.760 | 0.797 |
| YCB-V | 0.817 | 0.844 | 0.757 | 0.888 |
| ITODD | 0.356 | 0.412 | 0.538 | 0.469 |
| T-LESS | 0.856 | 0.765 | 0.828 | 0.779 |

- 空白：**无任何消融**（两阶段、KL 权重、w²D 头、phase-1 初始化、depth
  refinement 均无消融数字）；**无任何对应关系质量指标**（只有 BOP 指标与 ADD-S，
  对应关系只有一张定性图像网格 Table VI）；T-LESS objects 14/16/27/28/30 失败，
  object 27 从 0.843 掉到 0.125（单物体重训仅 0.615）；自述预测点 "disorganized"、
  "the model does not fully comprehend the entire shape of the object"；未讨论
  domain gap；未报告 runtime；Semantic Scholar 上仅 3 篇引用，无扩展或批评。

## 3. 与用户方案的重合度（Interpretation）

| 维度 | EPRO-GDR | 本项目方案 | 重合 |
|---|---|---|---|
| 前端 | GDRNPP（ConvNeXt + mask/XYZ/SRA 头） | 保留 GDRNPP 前端 | 高 |
| 后端 | EPro-PnP 替换 Patch-PnP | 同样 | 高 |
| 训练方式 | 端到端，KL 姿态分布损失反传进坐标头 | 姿态监督反传进 correspondence head | 高 |
| 可靠性 | 新增 w²D 头 | 计划包含 Reliability | 高 |
| 训练数据 | PBR-only → 真实测试 | LM-PBR → LM-O 真实 | 高 |
| 科学问题 | 能否输出概率分布/多假设、能否涨点 | 训练信号能否让 head 学到 solver-friendly 几何 | 低 |
| 消融/归因 | 无 | 受控干预、matched control | 低 |
| 对应关系质量 | 无指标 | 计划新增 | 低 |
| 跨域 | 未讨论 | 核心动机之一 | 低 |
| 失败模式 | T-LESS 定性 | 可系统化 | 中 |

方法骨架重合约 60–70%，科学问题重合约 20–30%。若论文卖点是"把 EPro-PnP 接进
GDRNPP 并涨点"，将被 EPRO-GDR 覆盖；若卖点是"训练信号如何塑造对应关系 + 跨域"，
EPRO-GDR 是最佳引用与起点。

## 4. 为什么"只反传姿态损失"不够（证据链）

三条独立证据指向同一结论：姿态级监督对对应关系是弱监督，需要对应关系级约束。

1. **EPro-PnP 自己的消融（Table 3，同一稠密网络）**：无坐标回归监督时，
   implicit differentiation（BPnP 式）**发散**，reprojection learning 仅
   `14.56` ADD(-S) 0.1d；加坐标回归后 implicit diff `53.33/88.74`、reprojection
   `60.65/92.04`、EPro-PnP Monte Carlo `63.80/92.66`（ADD(-S) 0.1d / 2°/2cm 口径）。
   官方 README 的 v2 改动也包含 "Use GaussianMixtureNLLLoss as auxiliary
   coordinate regression loss"、"Fix w2d scale handling (very important)"。
2. **Linear-Covariance Loss（ICCV 2023, arXiv:2303.11516）**：姿态级反传
   "conflicts with the averaging nature of the PnP problem"；"If more than 4
   correspondences are provided, the PnP solver performs a form of 'averaging'."，
   会产生 "gradients that degrade the accuracy of some of the 2D-3D
   correspondences"。原文明确："the LC loss cannot learn correspondences from
   scratch"，因此 "a surrogate loss function to supervise the correspondences
   remains necessary"。其消融（LM-O，GDR-Net 基线，ADD(-S) / 梯度正确率）：
   BPnP `64.1 / 53.9%`、EPro-PnP `64.5 / 59.3%`、LC `66.5 / 99.9%`。
3. **EPRO-GDR 的定性观察**：真的做了 solver-in-the-loop 训练，性能上涨，但预测点
   仍 "disorganized"、"does not fully comprehend the entire shape"。

机制解释（Interpretation）：姿态只有 6 自由度，对应关系有数千个点；很多组（甚至
退化的）对应关系都能让求解器给出正确姿态。仅用姿态损失训练，网络可能学到"能让
求解器凑出正确姿态的任意点集"，而不是物体真实几何。

## 5. 文献地图（证据强度：强 / 中 / 弱）

**可微 PnP 与 solver-in-the-loop**：

| 工作 | 会议/年份 | 关键事实 |
|---|---|---|
| BPnP | CVPR 2020 | 隐函数定理反传 PnP；初值敏感（LM 需初始化，常用 RANSAC）；只保证局部最优；正文未讨论非光滑/多解 |
| EPro-PnP | CVPR 2022 Oral / TPAMI 2024 | 概率化 PnP 层（SE(3) 位姿分布）+ KL 损失 + Monte Carlo/AMIS + Adaptive Huber + learned w²D；"correspondences X can be learned from scratch altogether"；6DoF 版基于 CDPN |
| EPRO-GDR | AIxVR 2025 | 见第 2 节 |
| Co-op | CVPR 2025 | 模板 semi-dense 对应（patch 分类 + offset 回归）+ 可微 PnP 精化；BOP 七个核心集 |
| PixLoc | CVPR 2021 | 可微 Levenberg-Marquardt + 学习阻尼 + 逐像素不确定性权重；初值差时落局部最优 |
| Blind PnP | ECCV 2020 Oral | Sinkhorn+RANSAC+PnP 作为声明层（DDN 框架），无需位姿先验 |
| Coupled Iterative Refinement / BD-PnP | CVPR 2022 | 双向深度增强可微 PnP，交替精化位姿与稠密对应 |

**对应关系级目标**：Linear-Covariance Loss（ICCV 2023，见第 4 节）、DCL-Net
（ECCV 2022，学习 correspondence confidence 并加权回归，指出 surrogate objective
"a step away from the true objectives"）、DGECN（可微 PnP/RANSAC + 图结构）、
SurfSurg6D（2026，几何嵌入一致性 + 硬负样本挖掘）。

**概率/歧义**：Corr2Distrib（RA-L 2025，从对应关系估计 6D 位姿分布）。

**证据强度**：强——BPnP 机制、EPro-PnP 概率层/KL/w²D 与 BPnP 对比、LC Loss 消融、
EPRO-GDR 事实；中——EPro-PnP-v2 具体新增项（论文只写 "new experiments"，清单来自
README）、EPRO-GDR 的增益归因；弱——"solver-aware training"作为术语（未找到以此
命名的论文）。

## 6. 跨域证据与边界（Observed + Interpretation）

**直接论证（最强）**：Perspective Flow Aggregation（ECCV 2022, arXiv:2203.09836）：
像素级对应关系监督迫使网络学低层特征、"thus generalize across domains"，而图像级
位姿监督学到的高层特征 "not to generalize well across domains"；纯合成训练在
Occluded-LINEMOD 上 ADD(-0.1d) `48.2`，超过多数用真实数据训练的基线（DeepIM 55.5、
GDR-Net 62.2 用了真实数据），加 20 张真实图后 `64.1`。

**机制论证**：Self6D++（TPAMI 2021, arXiv:2203.10339）——对应关系类方法
"robust to camera changes as they simply run PnnP using the new intrinsics"，而直接
回归 "always implicitly learn the camera intrinsics"。HomebrewedDB 上纯合成 `3.1`
→ 自监督 `84.4` → 真实标注上界 `93.8`；LINEMOD `77.4 → 88.5 → 91.0`。

**几何不变性工程验证**：Keypoint-Graph-Driven Learning（CVPR 2021）以 "the geometry
relations between keypoints are invariant across real/synthetic domains" 为前提，
LINEMOD 纯合成 `46.5% → 68.2%`。

**PBR → 真实的定量 gap（三源交叉验证）**：PBR-only 与 PBR+real 的差距从 2020 年
CosyPose `0.570 vs 0.637`（差 6.7 AR）缩到 2022 年 GDRNPP `82.7 vs 83.7`（差 1.0）
与 2023 年 GPose2023 `84.4 vs 85.6`（差 1.2）。因素消融（Tremblay, CVPRW 2018，
2D 检测任务）：固定光照 −6.1 AP、去随机纹理 −4.7 AP；相机差异是造成最严重跨域
失败的因素（Self6D++ HomebrewedDB）。

**反例与边界**：BOP 2020 冠军 CosyPose 是回归细化类（69.8 AR_C）；PViT-6D 声称
直接回归可竞争；ZebraPose 与 Pix2Pose 的 BOP 提交并非纯 PBR。**BOP 官方报告从未按
"correspondence vs regression" 分类统计**（其分类轴是 DNN vs PPF），因此跨域论证
应表述为"2022 年后排行榜顶部以对应关系/几何匹配类为主 + 多项机制性证据汇聚支持"，
并写明"尚无同架构受控 A/B 实验"这一空白。

**引用提醒**：Self6D++ 是 arXiv:2203.10339；iNeRF 是 IROS 2021；MegaPose 是
CoRL 2022；ZebraPose / Pix2Pose 训练数据非纯 PBR；纯 PBR 的 BOP 提交是
CosyPose / CDPNv2 / EPOS 的对应版本。

## 7. 设计启示（Interpretation）

把"能否让 head 学到 solver-friendly 对应关系"拆成可证伪子问题：

- **Q1（机制）**：solver-in-the-loop 训练是否改变对应关系的几何质量？用哪些指标
  能看出来？
- **Q2（归因）**：变化来自求解器在环，还是额外容量、损失重加权、两阶段初始化？
  EPRO-GDR 无消融，必须补。
- **Q3（跨域）**：训练信号是否改善 PBR→real / 跨数据集迁移？与直接回归监督相比
  迁移损失是否更小？
- **Q4（设计）**：Mask/Region/ROI2D/Reliability 中哪些能被求解器反馈有效约束？
  只有 XYZ 与权重直接进求解器；Mask 可定义 support、Region 可定义分组权重，否则
  与求解器没有通路。

建议的对照阶梯（每次只动一个结构因素）：GDRNPP/Patch-PnP 基线 → EPRO-GDR 式两阶段
（prior art 对照）→ 一阶段端到端 → 加入对应关系级约束/可靠性学习；消融 phase-1
初始化、KL 权重、w²D、GT 对应监督。

建议的评测：保留 BOP AR / ADD(-S) / AR_reS / AR_teS / 逐物体趋势；新增
solver-aware 对应关系指标（求解后内点率、空间一致性、w²D 校准、覆盖度）、
"几何利用斜率"（训练后重跑 α-sweep）、遮挡分层、跨数据集迁移。

工程前置项：EXP019 的 EPro-PnP 运行在独立 spawn 子进程（GDRNPP 与 EPro-PnP 都占用
顶层 `lib` 包名），诊断只做前向求解；端到端训练要求梯度同进程回流，需先解决包名
冲突并验证 KL 梯度能到达坐标头。

风险：对称物体多模态、近零深度/退化点集梯度、MC 采样 × 4096 点 × batch 48 的
训练成本、两阶段依赖（EPRO-GDR 的经验）、T-LESS 式崩塌。

## 8. Gate 口径建议（Derived + Decision）

0.001 绝对容差在 LM-O 1,445 targets 上约等于 **1.44 个 target** 翻转
（1/1445 = 0.000692）。EXP019 复现六点偏差折算：

| 对照点 | 偏差 | 折算 target | 相对偏差 |
|---|---:|---:|---:|
| Patch α=0 ADD | +0.00415 | +6.0 | +0.83% |
| Patch α=0 BOP | +0.00019 | +0.3 | +0.03% |
| RANSAC α=0 ADD | +0.00138 | +2.0 | +0.26% |
| RANSAC α=0 BOP | −0.00160 | −2.3 | −0.23% |
| RANSAC α=1 ADD | −0.00138 | −2.0 | −0.14% |
| RANSAC α=1 BOP | −0.00138 | −2.0 | −0.14% |

二项噪声标准误：p≈0.5 约 2.6% 相对，p≈0.68 约 1.8%，p≈0.99 约 0.26%。即
±3% 相对 ≈ 1.2–1.7 SE，±5% ≈ 2–2.8 SE。

**Decision（用户 2026-09-08）**：后续 gate 不再使用 0.001 这类绝对数值阈值，改为按
指标尺度设定的相对浮动（如 ±3%–±5%），重点判断机制、趋势与性能变化是否稳定成立。
建议分层：Integrity（绝对阈值，如 GT-XYZ < 0.5px、α=1 ≥ 0.95）、Mechanism
（Spearman ≥ 0.9、recovery ≥ 0.5、单调性、effect size）、Performance（相对阈值：
主指标 +3% 为有意义增益、非劣 −3%、逐物体一致性）、Reproduction（|Δ| ≤ 3% 相对
或配对 bootstrap 95% CI 包含 0）、Advisory。注意：相对阈值须写明口径（相对变化
而非百分点）；接近饱和的指标仍用绝对阈值；逐物体指标 n≈150–350 噪声约 4% 相对，
应使用配对检验；可考虑用同一 checkpoint 重复运行实测噪声底。

## 9. 来源

- EPRO-GDR：<https://arxiv.org/abs/2409.11819> ・ <https://ar5iv.labs.arxiv.org/html/2409.11819>
- EPro-PnP：<https://github.com/tjiiv-cprg/EPro-PnP> ・ <https://pubmed.ncbi.nlm.nih.gov/38227417/> ・ arXiv:2203.13254 / 2303.12787
- BPnP：<https://arxiv.org/abs/1909.06043>
- Linear-Covariance Loss：<https://arxiv.org/abs/2303.11516>
- DCL-Net：<https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2558_ECCV_2022_paper.php>
- Corr2Distrib：<https://arxiv.org/abs/2505.02501>
- Co-op：<https://arxiv.org/abs/2503.17731>
- Perspective Flow Aggregation：<https://ar5iv.labs.arxiv.org/html/2203.09836>
- Self6D++：<https://ar5iv.labs.arxiv.org/html/2203.10339>
- Keypoint-Graph-Driven Learning：<https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Keypoint-Graph-Driven_Learning_Framework_for_Object_Pose_Estimation_CVPR_2021_paper.pdf>
- BOP 2020/2022/2023/2024 报告：arXiv:2009.07378 / 2302.13075 / 2403.09799 / 2504.02812
- GDRN++：<https://arxiv.org/abs/2102.12145>

## 10. 证据边界

本页的文献结论区分三类：论文原文直引（Observed）、由原文数字直接计算（Derived）、
以及本项目语境下的推断（Interpretation）。EPRO-GDR 的增益归因、solver-in-the-loop
是否真正提升对应关系质量、以及跨域是否改善，均**没有**直接的一手受控实验证据，
属于本项目要回答的问题。本页不改变任何历史 gate、evaluator 输出或实验记录。
