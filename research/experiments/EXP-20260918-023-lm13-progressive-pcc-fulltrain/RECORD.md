# EXP023 — LM13 Progressive PCC 全 Backbone 训练

## Status

- lifecycle：`SUPERSEDED`
- execution：`FORMAL_NOT_STARTED`
- superseded by：EXP025 的 LM13 准备项
- final historical snapshot：`8c6ca86aa0668777548d4f30b8cc6f6ad9864067`
- server EGL、formal run、checkpoint 和精度指标：未生成。

## Research question

在标准 LINEMOD 13 类上，以 real LM + DeepIM renders 训练完整 ImageNet ConvNeXt backbone 与
EXP022 Progressive PCC。相对 LM-O 冻结臂，本实验同时改变数据集、hierarchy 和 backbone
策略，不能解释成单因素效应。

## Final intended protocol

| item | value |
|---|---|
| training data | `lm_13_train_online` + `lm_imgn_13_train_1k_per_obj_online` |
| test | `lm_13_test_online`，GT box |
| initialization | ImageNet ConvNeXt；340 tensors |
| optimizer | Ranger 1e-4；backbone LR×0.1；wd 0 |
| batch | physical 4，accumulation 6，effective 24 |
| schedule | 160 epoch；1000-step warmup；72% 后 cosine 到 0 |
| evaluation | 每 20 epoch，共 8 个固定点；best checkpoint disabled |
| renderer | train EGL；evaluation CPP |
| seed | 42 |

历史配置包括主实验、real-only 数据消融、PBR 协议和 BOP evaluator；它们均已退出 HEAD。
主实验的 `RESEARCH_PROTOCOL.FORMAL_READY` 从未打开。

## Dataset and interface facts

- `lm_imgn` 校验为 505,191 文件、6.1 GB、16 个对象目录、13 个 train lists；主实验使用
  13,000 renders。LM real train/test 分别 2,375/13,425 images。
- renderer pose 的 translation 单位为米，depth PNG 为毫米；无 mask 文件，mask 从 depth>0
  生成；`benchvise` 与 `benchviseblue` 指向同一序列。
- LM real 必须按 `image_set` 从 `lm/test` 取片，不能假设 `lm/train` 文件数相同。
- online data 检查中，相机内参与 bbox/mask 误差均为 0，bbox3d 与 models_info size 一致，
  在线渲染覆盖 source mask 约 0.954–0.989。
- DatasetContext 支持多个 train split，但额外 split 的对象顺序必须与第一项一致。

## Canonical local runs

| run | mode | status | scientific use |
|---|---|---|---|
| `RUN-20260918-lm13-gdrn-smoke-s42` | real entry, CPP, 1 epoch/4 iter | PASS, exit 0 | wiring only |
| `RUN-20260918-lm13-gdrn-pretraincheck-s42` | checkpoint/pretrain check | PASS | integrity only |
| final CUDA+CPP 20-step smoke | batch4 AMP | PASS | local engineering |
| CUDA+EGL smoke | local machine | BLOCKED | bindless texture unsupported |
| server EGL/formal | — | NOT RUN | no result |

真实入口 smoke 保存 `model_epoch_001.pth`（约 1.46 GB），checkpoint 记录 iteration3、optimizer
step4、base LRs `[1e-4,1e-5]`、GradScaler 65536 且无 skip；backbone 已不同于 ImageNet 源权重。
它只证明完整训练和 resume state 接线，不是收敛或精度结果。

## Engineering/integrity evidence

- CPU preflight 对正式、smoke 和 BOP-eval 配置均 PASS：13 objects、340 backbone tensors、
  `91,292,872` trainable parameters。
- CUDA+CPP real smoke 的 route/residual/mask 均 finite 并下降，无 AMP skip，峰值 allocated
  约 2.820 GB。
- test split 使用 online XYZ 后可在没有 `lm/test/xyz_crop` 的机器加载；旧预生成 XYZ split
  在相同条件下失败，作为对照。
- server preflight 曾实现对 LM real、lm_imgn、VOC、ImageNet weights、hierarchy 和 split count
  的 fail-closed 检查；launcher 资源 profile 和专用入口后来随历史执行面一起退出 HEAD。
- 本地 EGL 在进入模型计算前报 `Bindless Textures not supported`，所以没有伪造 EGL PASS；
  当时唯一 blocker 是服务器 EGL 验证。

服务器集成阶段还修过 first-create 目录职责、real-only 不要求 lm_imgn、未知 protocol name
fail-closed、hierarchy 在容器内生成等工程问题。这些变化不构成 Progressive PCC 机制证据，
也不代表服务器真实 mount/runtime gate 已运行。

## Evaluation boundary

正式评估原计划同时保留 legacy ADD(-S) 与 BOP AR，但本地缺少官方 Faster R-CNN
`bbox_faster_all.json`，只能使用 GT box，不能与 GDR-Net detector-box 论文数字直接比较。
LM13 也没有现成 matched reference model；历史 evaluator 只完成协议/manifest 接线。

## Decision

EXP023 完成本地数据、协议、训练入口和服务器资源门设计，但服务器 EGL 与 formal 没有运行，
因此没有精度或机制结论。随后统一 CAD attention 设计进入 EXP025；LM13 的后续准备改由
EXP025 独立配置承担，本实验关闭，不恢复旧 launcher/profile。

## Evidence / artifacts

- tracked evidence：无；事实由本 RECORD 与 Git snapshot 保存。
- 本地 smoke checkpoints、logs、dataset、hierarchy 和 manifest 位于 ignored `output/`、
  `.local/` 与外部 dataset 路径。

## Known gaps / boundaries

- 未运行真实服务器 bind mounts、container preflight、EGL smoke、formal 或八个评价点。
- 只有 GT-box 评价条件；detector-box 结果未生成。
- 没有 reference model 对应的正式 matched PnP 比较。
- T-LESS 数据和 variable-S symmetry 不在本实验范围。
