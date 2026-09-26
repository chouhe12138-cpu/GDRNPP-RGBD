# EXP026 adaptive E40 checkpoint 机制诊断

## Identity and protocol

- 母实验：`EXP-20260922-026-residual-aligned-sampling-ablation`；诊断 run：`RUN-20260925-e40-adaptive-s20260922-a01`。
- 正式模型源码：`8eebc5e50fec55b2dbe9bd655c0c1fffa8b7b4f5`；诊断工作树基线：`ba7a7c06692e92941fe8bfa5f26f0f7cf077db81`。使用当前 HEAD 的 EXP026 配置和 EXP025 诊断工具；配置、层级和模型 state_dict 合同由严格恢复审计覆盖。本次没有修改模型、正式配置、PnP 或 ADD 口径。
- 配置：`configs/gdrn/lmo_pbr/research/exp026_residual_aligned_sampling_ablation/train_adaptive_full.py`；`adaptive_l1_full`、`predicted_route`。
- 权重：`E:\6D姿态估计\EXP026\自适应\model_epoch_040.pth`，1,180,966,608 bytes，SHA256 `059fd40538751c527768cc333d7520587a2e6134f86f71f8ab2dd8052dec26c7`。epoch 40，iteration 255919；611 个 state tensor 全部有限，`strict=True` 恢复通过。层级 SHA256 `7ac75daa756ed7ae59463614737ccc46cd746173452cd1943a4a024d0817c631` 精确匹配。原始审计见 [checkpoint audit](evidence/checkpoint_audit.json)。
- EXP025 E15 原 `oracle/manifest.json` 与 deterministic 选样逐条相同：seed 20260922，每物体 16 候选，共 128 候选；6 个有效深度像素少于 16 被排除，122 valid ROI、92,197 valid pixels。EXP025/026 均用 LM-O BOP test GT-box、同一对称分支选择、同一 EPNP/RANSAC 3 px/100 iter 和每 ROI OpenCV seed、每物体固定最多 2,000 模型点 ADD(-S)@0.1d；无效或平移范数 >10 m 的 PnP 解计失败。这是匹配样本诊断，非正式全量评价。

## Observed：matched correspondence

逐 ROI 有效像素均值；误差和残差收益单位 mm。原始聚合和逐物体数据见 [oracle summary](evidence/oracle_summary.json) 与 [comparison](evidence/exp025_vs_exp026_comparison.json)。

| metric | EXP025 E15 | EXP026 E40 | delta |
|---|---:|---:|---:|
| T1 top1 | 80.59% | 82.28% | +1.69 pp |
| T2 top1 | 57.26% | 58.43% | +1.17 pp |
| T3 top1 | 28.49% | 29.45% | +0.96 pp |
| T3 top5 | 65.79% | 67.48% | +1.69 pp |
| GT T3 rank（逐 ROI median 的均值） | 9.82 | 10.00 | +0.18 |
| GT T3 probability | 0.2609 | 0.2830 | +0.0221 |
| predicted-cell representability | 68.05% | 52.75% | −15.29 pp |
| anchor XYZ error | 19.66 | 18.54 | −1.12 |
| full XYZ error | 18.53 | 16.90 | −1.63 |
| residual gain | 1.13 | 1.64 | +0.51 |
| residual-help fraction | 65.87% | 67.75% | +1.88 pp |

## Observed：pose oracle matrix

ADD(-S)@0.1d；分母均为 122 ROI，失败 PnP 记为 0。每 variant 的 rotation、translation、ADD(-S) 误差及 PnP 成功计数保留在 summary/comparison JSON。

| variant | EXP025 E15 | EXP026 E40 | delta | E40 PnP fail |
|---|---:|---:|---:|---:|
| normal | 57/122 = 46.72% | 65/122 = 53.28% | +6.56 pp | 1 |
| anchor_only | 42/122 = 34.43% | 50/122 = 40.98% | +6.56 pp | 0 |
| oracle_pred_route | 105/122 = 86.07% | 103/122 = 84.43% | −1.64 pp | 0 |
| gt_xyz_pred_mask | 117/122 = 95.90% | 119/122 = 97.54% | +1.64 pp | 1 |
| normal_gt_mask | 65/122 = 53.28% | 71/122 = 58.20% | +4.92 pp | 0 |
| gt_xyz_gt_mask | 118/122 = 96.72% | 118/122 = 96.72% | 0 pp | 0 |

## Derived：headroom

同一列内相减，单位 pp；详细机器可复算数值见 comparison JSON。

| difference | EXP025 E15 | EXP026 E40 |
|---|---:|---:|
| normal − anchor_only（真实残差 pose 收益） | 12.30 | 12.30 |
| oracle_pred_route − normal（残差 oracle 空间） | 39.34 | 31.15 |
| gt_xyz_pred_mask − oracle_pred_route（route/representation 空间） | 9.84 | 13.11 |
| normal_gt_mask − normal（mask 空间） | 6.56 | 4.92 |
| gt_xyz_gt_mask − normal（总体空间） | 50.00 | 43.44 |

## Per-object

成功数依次为 normal / predicted-route oracle / GT XYZ（均用预测 mask）。

| object | valid ROI | success | representable | anchor mm | full mm | residual gain mm |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 8 / 11 / 14 | 48.06% | 9.47 | 8.75 | 0.72 |
| 5 | 16 | 14 / 15 / 16 | 72.96% | 18.59 | 15.77 | 2.83 |
| 6 | 13 | 4 / 11 / 13 | 47.51% | 16.59 | 15.15 | 1.44 |
| 8 | 16 | 14 / 15 / 16 | 72.99% | 21.47 | 18.81 | 2.66 |
| 9 | 15 | 4 / 11 / 15 | 33.79% | 17.45 | 16.91 | 0.54 |
| 10 | 16 | 8 / 15 / 15 | 48.41% | 16.82 | 15.26 | 1.57 |
| 11 | 14 | 10 / 11 / 14 | 30.46% | 23.23 | 22.27 | 0.97 |
| 12 | 16 | 3 / 14 / 16 | 62.90% | 24.86 | 22.64 | 2.22 |

## Interpretation and decision

`predicted_route` 训练后真实残差 XYZ 收益从 1.13 增至 1.64 mm，normal 成功率提高 6.56 pp，oracle gap 缩小 8.20 pp，支持部分缓解旧 route mismatch；但单个 E15→E40、层级也不同，不能把变化单独归因于 target 修复。T3 top1 略升而 predicted-cell representability 降 15.29 pp，route/representation headroom 扩至 13.11 pp，**不支持可达率已改善**。残差 oracle 仍比 normal 高 31.15 pp，是当前最大的干预空间；fine XYZ / residual decoder 仍是主要瓶颈，特别是物体 6/9/10/12。GT mask 只增加 4.92 pp，GT XYZ + 预测 mask 达 97.54%，不支持 mask/PnP 是主要限制。

本诊断没有 LR 干预，不能证明学习率是原因，也不能单凭 E40 结果批准具体 LR 改动。下一轮可把 LR 作为预注册、matched 的候选干预，保留本 oracle 矩阵和正式固定评价点作为判据；EXP026 两臂当前 formal 协议不变。逐 ROI 原始数据保存在 `.local/exp026_e40_diagnostic/oracle/per_roi.json`，权重与完整数据不进入 Git。
