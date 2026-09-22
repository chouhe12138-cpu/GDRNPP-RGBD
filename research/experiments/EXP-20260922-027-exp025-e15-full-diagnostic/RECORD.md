# EXP027 — EXP025 Full E15 checkpoint 主瓶颈诊断

## Identity / status

- experiment ID：`EXP-20260922-027-exp025-e15-full-diagnostic`
- run ID：`RUN-20260922-e15-full-s20260922-a03`；状态 `DIAGNOSTIC_COMPLETE`，不属于 EXP025 formal 评价点。
- 诊断源码 commit：`b7606405ae05cb8319ed3fd94651fe2c79837bc9`；正式模型源码 commit：
  `b9bddccef2f12b4365e0e1e2e68222d733de572b`。
- 配置：`configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_imagenet_full.py`。
- 权重：`E:\6D姿态估计\EXP025\Full\model_epoch_015.pth`，1,180,966,608 bytes，SHA256
  `7fbbfb8e168a9b8eec8cbb2ba621c07981a5c4c2c3e8045f78e6f4af2014b658`。
- 层级：`.local/dataset_cache/exp025/consistent_v3.npz`，SHA256
  `02ce090949bc40b2732417fec23984f3f748431098c5f67c853839d10ff1a373`。
- 外置结果：`E:\6D姿态估计\EXP025\EXP025_E15_FULL_checkpoint_diagnostic_RUN-20260922-a03\`。

## Question and protocol

问题：在 E15 Full checkpoint 上，pose 损失主要来自 backbone 未适配、T3 route、残差、mask，
还是 CAD cross-attention 弱用？诊断包是参考；现行代码、权重和已交付评价点为准。只改推理干预，
不改正式训练或模型权重。

本机 `pytorch22`、RTX 4060 Laptop GPU；沙箱内 GPU 不可见，经用户授权提权运行本地推理。
LM-O BOP test 的 GT box，seed `20260922`，各物体随机固定 16 个候选 ROI；从 128 个候选中，
6 个可见深度有效像素不足 16 而排除，最终 122 ROI / 92,197 个有效像素。ROI 实测深度按相机
内参与 GT pose 反投影为 GT CAD XYZ；路由/残差对称物体从 head 已登记的等价变换中选择与当前
预测锚点距离最小的分支（乐观上界）。GT XYZ upper bound 仍使用原始坐标。

PnP 使用现行 `GDRN_Evaluator.get_img_model_points_with_coords2d` 和 `misc.pnp_v2`：EPNP、
RANSAC reprojection 3 px / 100 iter、同一 ROI 固定 OpenCV seed。ADD(-S)0.1d 用各物体固定采样
最多 2,000 个模型点计算；失败解计失败，非有限或平移范数 >10 m 的不合理解计失败。
这是匹配样本的诊断成功率，非正式全量 BOP score。E15 Full 正式 ADD(-S)0.1d=0.480277，
本诊断正常分支=0.467213，量级一致。

## Observed — integrity

checkpoint 顶层键含 `model/optimizer/scheduler/gradscaler/epoch/iteration`；epoch=15、
iteration=95,969；611 个模型 tensor、98,352,068 参数，全部有限；所选 config 下严格完整
restore 成功。精确数据见 [checkpoint audit](evidence/checkpoint_audit.json)。

## Observed — matched oracle

122 ROI、同一 PnP 口径；不可用/不合理 PnP 解计作成功 0。完整紧凑结果见
[oracle summary](evidence/oracle_summary.json)。

| 干预 | ADD(-S)0.1d 成功 | 对 normal 差值 |
|---|---:|---:|
| normal：预测 XYZ + 预测 mask | 57/122 = 46.72% | — |
| anchor only：预测 T3 锚点、零残差 + 预测 mask | 42/122 = 34.43% | −12.30 pp |
| oracle residual on predicted route：预测锚点/radius，GT 精确残差投影到单位球 + 预测 mask | 105/122 = 86.07% | +39.34 pp |
| GT XYZ + 预测 mask | 117/122 = 95.90% | +49.18 pp |
| 预测 XYZ + GT mask | 65/122 = 53.28% | +6.56 pp |
| GT XYZ + GT mask | 118/122 = 96.72% | +50.00 pp |

同一 ROI 配对自助法（20,000 次、未按场景聚类）的差值 95% 描述区间：anchor only
`[-22.13,-2.46] pp`，pred-route oracle `[29.51,49.18] pp`，oracle→GT XYZ 的额外
`[4.10,15.57] pp`；GT mask `[-1.64,14.75] pp`。不是跨 seed、跨场景泛化置信区间。

有效像素的逐 ROI 均值：T1/T2/T3 top-1=80.59%/57.26%/28.49%，T3 top-5=65.79%，
预测球可达率=68.05%；锚点误差 19.66 mm，完整 XYZ 误差 18.53 mm，残差净收益
1.13 mm，65.87% 像素上残差减小距离。非对称 6 物体（92 ROI）normal/oracle/GT XYZ
成功分别为 40/80/89，说明主效应不依赖对称分支的乐观选择。

| object | valid ROI | normal | pred-route oracle | GT XYZ | route 可达率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 5 | 14 | 14 | 74.1% |
| 5 | 16 | 12 | 15 | 16 | 78.4% |
| 6 | 13 | 5 | 10 | 12 | 65.2% |
| 8 | 16 | 10 | 15 | 16 | 86.8% |
| 9 | 15 | 5 | 11 | 15 | 53.4% |
| 10 | 16 | 5 | 14 | 15 | 52.6% |
| 11 | 14 | 12 | 11 | 13 | 52.6% |
| 12 | 16 | 3 | 15 | 16 | 77.9% |

物体 11 的 pred-route oracle 比 normal 少 1 个成功：对称分支的“最接近锚点”不是训练
目标选择法，且 PnP 对不同可达像素的 RANSAC 敏感；该例外保留，不将 oracle 视为逐样本单调。

## Observed — representation and CAD intervention

与 oracle manifest 的每物体前 4 个匹配候选共 32 ROI，3 个低深度 ROI 排除，29 有效 ROI。
完整结果见 [interventions summary](evidence/interventions_summary.json)；8 幅代表图留在外置结果。

| 推理干预 | ADD(-S)0.1d | XYZ 平均误差 |
|---|---:|---:|
| Full E15 baseline | 13/29 | 27.23 mm |
| ImageNet backbone reset，固定 E15 head | 0/29 | 60.80 mm |
| official backbone transplant，固定 E15 head | 6/29 | 22.03 mm |
| bypass T1/T2/T3 CA | 14/29、13/29、12/29 | 27.43、27.39、27.36 mm |
| bypass all T1–T3 CA | 11/29 | 28.71 mm |
| T3 token 各通道独立重排，破坏 node 特征绑定 | 14/29 | 27.22 mm |

原包建议的“整体 T3 token 顺序 permutation”对无位置编码的 cross-attention 在数学上
严格置换不变；早期 a01 运行确实近乎 bitwise 不变，但不能据此判断模块未用。因此 canonical
改用各特征通道独立的 node 重排；仍未见显著受损，且 all-bypass 仅少 2/29 个成功。
这只支持本样本上 CAD CA 弱使用或存在冗余，不证明整个模块无效；按停止规则不继续画 attention heatmap。

Full 相对 ImageNet 权重 L2 漂移 / ImageNet L2=0.656；同样 29 ROI 的 stage-wise linear CKA
（Full vs ImageNet）为 0.566/0.786/0.0038/0.579，Full vs official 为
0.920/0.802/0.490/0.875。小样本 CKA 只作描述。ImageNet reset 0/29 是更直接的
inference-only 证据：不能称 E15 backbone 没有学到有效适配。official transplant 下降可能是
head/backbone co-adaptation mismatch，不代表官方 backbone 单独更差。

## Interpretation / decision

- **主瓶颈：预测路由条件下的残差/精细 XYZ 解码。** 实际残差平均仅改进 1.13 mm，但保持
  同一 T3 route、mask、PnP 后，oracle 残差额外带来 +39.34 pp；非对称物体同样大幅提升。
  实现中训练 target 用 GT T3 的 anchor/radius，而推理解码用预测 T3 argmax 的 anchor/radius；
  这是一个明确的 train–inference route mismatch，可能解释部分剩余空间，但本诊断未证明
  修改 target 就一定能达到 oracle 上界。
- **次要瓶颈：route/cell representability。** oracle 残差→GT XYZ 还有 +9.84 pp，
  物体 9/10/11 的可达率偏低；但 +39.34 pp 的残差上界更大，故不将 route 列为首改方向。
- mask 有 +6.56 pp 的观察增益，但配对区间跨零；不是主因。ImageNet reset 排除“Full
  backbone 基本没适配”；CA 消融不支持将 attention 可视化或单层权重分析列为优先；
  T4 传下来的旧 T3 radius 问题两臂共有，不能单独解释两臂 E15 差距。
- **唯一下一主修改方向：**在 EXP025 formal 完成后，设计 matched、预测路由对齐的 residual
  supervision/decoder 干预，保持 backbone、T3 hierarchy、mask、PnP 和训练预算不变；
  先以当前 oracle 矩阵为预注册机制判据，再看正式姿态指标。不在运行中的 formal release
  上改代码或选模。

E15 的 E10→E15 局部平台不能外推为长期平台：2026-09-22 新交付的 Full E25 BOP AR
0.693580、ADD(-S)0.1d 0.523183，高于其 E15 0.661326/0.480277。两臂比较仍须等
相同 epoch 的全部固定点，不以本诊断改变 EXP025 协议或最终结论。

## Boundaries / invalid attempts

- a01 oracle：未拒绝 PnP 极端数值解，均值失真；不进入结论。
- a02 oracle：未处理等价对称分支，物体 10/11 route 错误与可达率失真；不进入结论。
- a01 interventions：纯 token 顺序置换是 attention 不变换，不能作为因果消融；不进入结论。
- a02 interventions：空间图未屏蔽无效背景，原始聚合指标与 a03 相同，但图不使用。
- 当前只分析单个 E15 checkpoint、单 seed 的诊断 ROI；深度边缘、对称选择和采样近似会
  影响像素精度。未做训练干预、跨 seed、全量关键变体验证或 E40 归因。
