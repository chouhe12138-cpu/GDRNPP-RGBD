# EXP025 — 统一 T3 分类与层级 CAD Attention

## Status

- lifecycle：`ACTIVE`
- execution：`FORMAL_NOT_STARTED`
- last verified：`2026-09-20 / 0ec5f9ad72777d37f47b4489e252ffb09cb68413`
- decision：formal 前置条件已满足，两条 LM-O formal arm 已获启动资格。

## Research question

在相同统一 T3=512 层级 CAD attention head 下，比较“原 GDRNPP LM-O backbone 冻结”与
“ImageNet ConvNeXt 初始化并完整训练”两种组合策略的正式 LM-O 表现。初始化来源和训练范围
同时变化，因此结果不解释为单一冻结消融。

## Protocol

| item | value |
|---|---|
| experiment ID | `EXP-20260920-025-hierarchical-cad-attention` |
| dataset / bbox | LM-O PBR40 / GT box |
| seed | 42 |
| schedule | 40 epoch；E5/E10/E15/E20/E25/E30/E35/E40 固定评价 |
| batch | 真实 batch48；`IMS_PER_BATCH=REFERENCE_BS=48`；无 accumulation |
| optimizer | AdamW，lr 3e-4，wd .01，betas (.9,.999)，eps 1e-8 |
| LR schedule | 4% linear warmup；cosine 到初始 lr 的 .01 |
| AMP | enabled；initial scale 32768 |
| hierarchy | `consistent_v3.npz`，SHA256 `02ce090949bc40b2732417fec23984f3f748431098c5f67c853839d10ff1a373` |
| selection | 不按中间结果选模；best checkpoint disabled；不自动追加 seed |

正式配置：

| arm | machine | initialization | trainable / LR |
|---|---|---|---|
| `official_frozen` | lab0 / GPU 0 | 原 GDRNPP LM-O checkpoint | CAD head 3e-4；backbone frozen |
| `imagenet_full` | lab1 / GPU 1 | ImageNet ConvNeXt | CAD head 3e-4；backbone 3e-4 |

配置路径分别为
`configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_official_frozen.py` 和
`train_imagenet_full.py`，共享 `common.py`。

## Final model contract

- 四级 Image-SA 在 8/16/32/64 分辨率执行，前三层 attention 结果写回并进入下一 transition；
  最终 64×64 tokens 与 T0/T1/T2/T3 CAD tokens 依次 cross-attention。
- 只输出 T3=512 logits；T2/T1 是 FP32 log-probability marginal。T1/T2/T3 NLL 相加。
- GT 使用 nested parent-local 最近邻；同一 SE(3) 对称分支同时用于 route 与 residual。
- Residual V2 读取 detached predicted soft-T3 CAD context，最后一层零初始化；不使用 GT route。
- mask 为 BCE；推理以 T3 argmax 的 anchor/radius 加 bounded residual 解码 XYZ。
- `BACKBONE_INIT` 只负责初始化；`MODEL.WEIGHTS` 只接受完整 GDRN_CAD checkpoint，eval
  缺少完整组件时 fail-closed。

## Canonical runs

| run / evidence | source | mode | status | scientific validity | note |
|---|---|---|---|---|---|
| lab0 gate 65536 a01 | `e3e0e65` | EGL batch48, 8-step gate | FAIL step 1 | negative control | `mask_predictor.weight` 非有限；loss、logits 和其他梯度有限 |
| `RUN-20260920-111840-gate-s42-a01` | `e3e0e65` | lab0 EGL batch48 @32768 | PASS | integrity | 0 skipped；90/90 Image-SA 更新；checkpoint roundtrip |
| lab1 gate 65536 a01 | `e3e0e65` | EGL batch48, 8-step gate | FAIL step 1 | negative control | 当时 backbone lr 为 3e-5；同一 Mask 权重非有限 |
| lab1 gate 32768 a02 | `e3e0e65` | EGL batch48 gate | PASS | superseded integrity | backbone lr 3e-5，后由最终配置重跑取代 |
| lab1 gate 32768 a03 | `b9bddcc` | final-config EGL batch48 gate | PASS | canonical integrity | backbone/head 均 3e-4；0 skipped；90/90 更新；roundtrip |
| formal arms | not generated | 40-epoch formal | NOT STARTED | — | 尚无 formal checkpoint 或指标 |

服务器报告中的 `device=cuda:0` 是容器内可见设备编号；物理映射仍为 lab0→GPU0、lab1→GPU1。

## Formal results

Not generated。不得用 smoke、fixed-batch、profile 或 gate loss 替代正式精度指标。

| Epoch | BOP AR | ADD(-S) | AR_reS | AR_teS |
|---:|---:|---:|---:|---:|
| E5–E40 | 未生成 | 未生成 | 未生成 | 未生成 |

## Key engineering and integrity evidence

1. 当前结构的 CPU preflight、本地 CUDA/CPP smoke、四级 Image-SA 梯度/更新覆盖和 hierarchy
   identity 检查通过；服务器 gate 再次记录了 hierarchy SHA 和全部 stage 更新。
2. LightningLite optimizer wrapper 的 resume 边界已修复：底层 optimizer state/param groups
   与 scheduler LR 恢复到实际执行 step 的对象；回归测试覆盖连续训练与 resume。
3. GradScaler 跳过 optimizer update 时 scheduler 不推进；`SOLVER.AMP.INIT_SCALE` 已接入
   生产 precision plugin，并随 checkpoint 恢复。
4. 服务器 batch48 gate 在 65536 的失败集中于 Mask 头缩放后的 fp16 梯度，前向量和 loss
   有限；32768 是两臂共同采用并实际通过的初始 scale。
5. lab1 原 backbone lr 3e-5 来自 LM13 协议继承，未有 LM-O formal 依据；官方 LM-O 配方对
   ImageNet ConvNeXt backbone 不降倍率，因此 formal 前将 multiplier 从 .1 修订为 1.0。
6. lab1 在该 LR 修订后重新执行 32768 gate，optimizer groups 明确记录 head/backbone 均为
   3e-4，未出现 skipped step。

## Derived / interpretation

- lab0 65536 报告的其他模块最大未缩放梯度为 0.482，lab1 旧 LR gate 为 0.499；两次失败
  都只标记同一 Mask 权重。结合有限前向量与 32768 PASS，这支持“loss-scale fp16 边界”解释，
  不支持“结构或 loss 已发散”。
- lab0 32768 的中位 step 时间为 334.5 ms，峰值 allocated/reserved 13.72/14.26 GB；lab1
  最终配置为 471.4 ms 和 22.60/23.26 GB。计时来自固定 8-step gate，不含重复加载/渲染，
  只作资源 advisory。
- 8-step loss 下降只说明工程可训练性，不外推为 formal 精度或收敛结论。
- lab1 最终 LR 配置没有重新测试 65536；其 canonical 结论仅为 32768 PASS。旧 LR 的 65536
  FAIL 作为 scale 选择过程的 negative control 保留，不冒充最终配置结果。

## Decision

- `FORMAL_READY=True`；两臂共同使用 `SOLVER.AMP.INIT_SCALE=32768`。
- 从同一 clean release 启动 lab0 `official_frozen` 与 lab1 `imagenet_full` formal。
- 按全部固定评价点记录结果，不根据中间点改变协议、选模或追加 seed。
- formal 期间不 pull、不改 release、不替换镜像。
- LM13 仅为下一阶段准备项；当前保持 server-disabled，LM-O 完成前不启动。

## Evidence map

| claim | canonical evidence |
|---|---|
| lab0 65536 Mask overflow | [official_frozen a01](evidence/exp025_server_gate_official_frozen_a01.json) |
| lab0 32768 PASS | [official_frozen a02](evidence/exp025_server_gate_official_frozen_a02.json) |
| lab0 run/source/image provenance | [official_frozen a02 metadata](evidence/exp025_server_gate_official_frozen_a02_metadata.json) |
| lab1 旧 LR 65536 Mask overflow | [imagenet_full a01](evidence/exp025_server_gate_imagenet_full_a01.json) |
| lab1 最终 LR 32768 PASS | [imagenet_full a03](evidence/exp025_server_gate_imagenet_full_a03.json) |
| 当前结构、配置与 contract | 当前 source tree、正式 configs 与 `research/exp025/tests` |

被后续配置取代的 a02/recheck、重构前 fixed-batch、工具调试和重复 smoke JSON 已从当前 HEAD
移除；其历史事实由本 RECORD 摘要，原文件可从 Git history 恢复。checkpoint、完整日志、
固定 batch 和 tensor dump 继续外置。

## Known gaps / boundaries

- 未运行：两臂 formal、E5–E40 正式评估、跨 seed 复验。
- 未运行：重构后结构的 route/residual fixed-batch 机制复验；重构前数值不能迁移。
- 未运行：lab1 最终 3e-4 配置的 65536 gate；不据旧 LR negative control声称已测试。
- 不能推出：单个 gate 的短程 loss、资源或梯度覆盖不代表正式姿态精度。
- 外置原始产物：服务器 run 目录、本地 `output/diagnostics`、checkpoint 和完整 console log。
