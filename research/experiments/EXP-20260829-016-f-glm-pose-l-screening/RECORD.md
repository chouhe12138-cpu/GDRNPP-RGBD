# EXP013F — GLM-Pose-L Screening

## 协议与运行

- 状态：`COMPLETE / SCREEN_FAIL_2_OF_4_GATES`
- M2 attention pooling + M3 ROI depth statistics；冻结 backbone/geometry
- 40 epoch；batch 48；Ranger lr `8e-4`；seed `42`；固定 E40 gate
- formal run：`RUN-20260829-103858-formal-s42-a01`；lab1
- source commit：`e924b96a5b05f61c70de1b56bf0f7e1a7f8a7bfb`
- 配置：`configs/gdrn/lmo_pbr/research/exp013/f_glm_pose_l/train.py`
- checkpoint：`model_epoch_040.pth`，epoch 40 / iteration 255919

v1 只在 offline dataset 增加 depth stats，而正式配置使用 online loader，导致
smoke `KeyError: roi_depth_stats`；同时统计作用在整图而非 ROI，batch shape 也不符。
v2 将统计统一到 online/offline 的 ROI crop，固定为 shape `[4]` 后通过本地门禁。
v1 没有正式指标，不进入科学结果。

2026-09-02 使用最终 lightweight framework 分别在 lab0、lab1 完成 smoke，训练
链路正常；该双机运行只验证 release/container/runtime 契约，不增加或改变 EXP013F
的科学结果。

## 正式轨迹

| Epoch | BOP AR | ADD(-S) target-micro | AR_reS | AR_teS |
|---:|---:|---:|---:|---:|
| 5 | 0.575645 | 0.329412 | 0.285582 | 0.704037 |
| 10 | 0.638674 | 0.452595 | 0.408766 | 0.765398 |
| 15 | 0.648992 | 0.458824 | 0.426990 | 0.771857 |
| 20 | 0.649107 | 0.458824 | 0.437601 | 0.765167 |
| 25 | 0.656858 | 0.467128 | 0.437601 | 0.788466 |
| 30 | 0.669123 | 0.496194 | 0.496886 | 0.781084 |
| 35 | 0.672293 | 0.501730 | 0.486044 | 0.784775 |
| 40 | **0.684129** | **0.504498** | **0.515802** | **0.799308** |

## E40 gate

| 指标 | 门槛 | E40 | 差值 | 判决 |
|---|---:|---:|---:|---|
| AR_reS | ≥ 0.4930 | 0.515802 | +0.022802 | PASS |
| BOP AR | ≥ 0.6838 | 0.684129 | +0.000329 | PASS |
| AR_teS | ≥ 0.8028 | 0.799308 | -0.003492 | FAIL |
| ADD target-micro | ≥ 0.5107 | 0.504498 | -0.006202 | FAIL |

四项必须全过，因此结论为 `SCREEN_FAIL`；两项差距都在 ±0.01 的单次评估噪声带
内，只能称为边缘失败。E40 macro-object ADD 为 `0.506465`；逐物体为：

| 物体 | ape | can | cat | driller | duck | eggbox | glue | holepuncher |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADD(-S) | 0.486 | 0.804 | 0.462 | 0.775 | 0.122 | 0.378 | 0.750 | 0.275 |

reS 为该家族第二高，仅次于 C；attention pooling 的 rotation 读出获得支持。
teS 高于 A、低于 B，M3 depth statistics 没有形成超越 B 的 translation 优势。
外置证据位于 `E:\6D姿态估计\EXP-013\实验F\`。
