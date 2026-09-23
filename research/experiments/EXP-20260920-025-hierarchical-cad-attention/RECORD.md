# EXP025 — 统一 T3 分类与层级 CAD Attention

## Status

- lifecycle：`ACTIVE`
- execution：`FORMAL_COMPLETE`（两臂 E5–E40 八个固定评价点全部交付）
- last verified：`2026-09-23 / formal source b9bddccef2f12b4365e0e1e2e68222d733de572b`
- decision：formal 前置条件已满足，两臂 formal 已于 2026-09-20 启动；全部固定点已于 2026-09-23 核对交付，最终结论待用户判定。

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

两臂 formal 的 source commit 均为 `b9bddccef2f12b4365e0e1e2e68222d733de572b`；镜像
`gdrnpp-research:torch220-cu121-sm89-e3e0e651eac4`（image_id
`sha256:73ced4b86b18e83b702f20b6192d3fc1274e043db8d445d0437edd290937441d`，build revision
`e3e0e651eac4690fc6d2a4b8ef68f37e437ffb23`）。run：lab0 `official_frozen` =
`RUN-20260920-120844-formal-s42-a01`；lab1 `imagenet_full` =
`RUN-20260920-120843-formal-s42-a01`。

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
| `RUN-20260920-120844-formal-s42-a01` | `b9bddcc` | lab0 official_frozen 40-epoch formal | COMPLETE（E5–E40 全部固定点已评估） | canonical formal | console 显示完成 E40 训练与 `FINAL_EVAL_REUSED periodic_epoch=40` |
| `RUN-20260920-120843-formal-s42-a01` | `b9bddcc` | lab1 imagenet_full 40-epoch formal | COMPLETE（E5–E40 全部固定点已评估） | canonical formal | console 显示完成 E40 训练与 `FINAL_EVAL_REUSED periodic_epoch=40` |

服务器报告中的 `device=cuda:0` 是容器内可见设备编号；物理映射仍为 lab0→GPU0、lab1→GPU1。

## Formal results

两臂八个固定评价点（E5/E10/E15/E20/E25/E30/E35/E40）全部交付，共 16 个点。
不得用 smoke、fixed-batch、profile 或 gate loss 替代正式精度指标。

| Epoch | arm | BOP AR | ADD(-S)0.1d | AR_reS | AR_teS |
|---:|---|---:|---:|---:|---:|
| 5 | official_frozen | 0.668524 | 0.482353 | 0.506805 | 0.769089 |
| 5 | imagenet_full | 0.625211 | 0.440830 | 0.438062 | 0.733103 |
| 10 | official_frozen | 0.685128 | 0.546713 | 0.551557 | 0.794925 |
| 10 | imagenet_full | 0.660824 | 0.498270 | 0.485813 | 0.773010 |
| 15 | official_frozen | 0.687057 | 0.513495 | 0.543483 | 0.795386 |
| 15 | imagenet_full | 0.661326 | 0.480277 | 0.504037 | 0.766321 |
| 20 | official_frozen | 0.679033 | 0.492734 | 0.548789 | 0.787082 |
| 20 | imagenet_full | 0.667956 | 0.460208 | 0.526644 | 0.770934 |
| 25 | official_frozen | 0.690955 | 0.526644 | 0.556171 | 0.798847 |
| 25 | imagenet_full | 0.693580 | 0.523183 | 0.539100 | 0.806920 |
| 30 | official_frozen | 0.692291 | 0.534948 | 0.560784 | 0.798616 |
| 30 | imagenet_full | 0.691954 | 0.532180 | 0.556863 | 0.791234 |
| 35 | official_frozen | 0.689696 | 0.519031 | 0.551557 | 0.798385 |
| 35 | imagenet_full | 0.700330 | 0.545329 | 0.563552 | 0.804152 |
| 40 | official_frozen | 0.690254 | 0.520415 | 0.557093 | 0.800461 |
| 40 | imagenet_full | 0.703682 | 0.557093 | 0.562168 | 0.814994 |

BOP AR、AR_reS、AR_teS 来自各 epoch 的 `scores_bop19_*epoch.json`；ADD(-S)0.1d 来自
同一运行日志的 `EVAL_SUMMARY.add_s_0.1d`，不以 BOP toolkit 的 AD 口径替代。两臂共 16 个
BOP AR 点与同 epoch score JSON 一致，且与 `eval_summary.jsonl` 的 `bop_ar` 逐点相等。

### 共同 E15 与两臂最终 E40 的逐物体 ADD(-S)0.1d

| object | Frozen E15 | Full E15 | Frozen E40 | Full E40 |
|---|---:|---:|---:|---:|
| ape | 0.411429 | 0.245714 | 0.434286 | 0.508571 |
| can | 0.768844 | 0.698492 | 0.778894 | 0.839196 |
| cat | 0.532164 | 0.444444 | 0.491228 | 0.508772 |
| driller | 0.790000 | 0.740000 | 0.810000 | 0.810000 |
| duck | 0.255556 | 0.333333 | 0.272222 | 0.316667 |
| eggbox | 0.372222 | 0.311111 | 0.400000 | 0.472222 |
| glue | 0.714286 | 0.728571 | 0.678571 | 0.764286 |
| holepuncher | 0.275000 | 0.350000 | 0.295000 | 0.255000 |

全部 16 个点的逐物体 recall 见两臂 `eval_summary.jsonl`。

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

### 两臂固定点差（imagenet_full − official_frozen，E5–E40 全部八个点）

单位为百分点；只作描述，固定点是预定评价，不构成按中间结果选模。

| Epoch | Δ BOP AR | Δ ADD(-S)0.1d | Δ AR_reS | Δ AR_teS |
|---:|---:|---:|---:|---:|
| 5 | −4.331 | −4.152 | −6.874 | −3.599 |
| 10 | −2.430 | −4.844 | −6.574 | −2.191 |
| 15 | −2.573 | −3.322 | −3.945 | −2.907 |
| 20 | −1.108 | −3.253 | −2.215 | −1.615 |
| 25 | +0.263 | −0.346 | −1.707 | +0.807 |
| 30 | −0.034 | −0.277 | −0.392 | −0.738 |
| 35 | +1.063 | +2.630 | +1.200 | +0.567 |
| 40 | +1.343 | +3.668 | +0.507 | +1.453 |

各指标并非单调。Frozen 的已交付最佳点：BOP/ADD/reS 在 E30（0.692291 / 0.534948 /
0.560784），teS 在 E40（0.800461）；Full 的已交付最佳点：BOP/ADD/teS 在 E40
（0.703682 / 0.557093 / 0.814994），reS 在 E35（0.563552）。这些最佳点是事后描述，
不改变固定点评价协议。两臂同时改变初始化来源与可训练范围，上述差值是组合策略差，
不是单一冻结消融效应；共同 E5–E20 为 Frozen 领先，E25 起 BOP AR 差距收敛并在
E35/E40 转为 Full 领先，此为对固定点序列的描述，不构成选模或显著性结论。

## Decision

- `FORMAL_READY=True`；两臂共同使用 `SOLVER.AMP.INIT_SCALE=32768`。
- 两臂 formal 已于 2026-09-20 从同一 release（`b9bddcc`）启动；八个固定评价点
  E5–E40 已于 2026-09-23 全部按预定协议记录（紧凑副本来自用户保留的
  `E:\6D姿态估计\EXP025\Frozen` 与 `...\Full`）。
- console 证据显示两臂训练自然到达 E40 并 `FINAL_EVAL_REUSED periodic_epoch=40`；
  未见 error/NaN/traceback。最终组合策略结论待用户判定，不根据固定点结果自动选模、
  改协议或追加 seed。
- formal 期间不 pull、不改 release、不替换镜像；该约束随两臂训练结束而完成。
- LM13 仅为下一阶段准备项；当前保持 server-disabled，EXP025 收官判定前不启动。

## Evidence map

| claim | canonical evidence |
|---|---|
| lab0 65536 Mask overflow | [official_frozen a01](evidence/exp025_server_gate_official_frozen_a01.json) |
| lab0 32768 PASS | [official_frozen a02](evidence/exp025_server_gate_official_frozen_a02.json) |
| lab0 run/source/image provenance | [official_frozen a02 metadata](evidence/exp025_server_gate_official_frozen_a02_metadata.json) |
| lab1 旧 LR 65536 Mask overflow | [imagenet_full a01](evidence/exp025_server_gate_imagenet_full_a01.json) |
| lab1 最终 LR 32768 PASS | [imagenet_full a03](evidence/exp025_server_gate_imagenet_full_a03.json) |
| lab0 formal E5–E40 指标 | [official_frozen run](evidence/RUN-20260920-120844-formal-s42-a01/) |
| lab1 formal E5–E40 指标 | [imagenet_full run](evidence/RUN-20260920-120843-formal-s42-a01/) |
| 当前结构、配置与 contract | 当前 source tree、正式 configs 与 `research/exp025/tests` |

被后续配置取代的 a02/recheck、重构前 fixed-batch、工具调试和重复 smoke JSON 已从当前 HEAD
移除；其历史事实由本 RECORD 摘要，原文件可从 Git history 恢复。checkpoint、完整日志、
固定 batch 和 tensor dump 继续外置。

## Known gaps / boundaries

- E15 Full 的本地 checkpoint 机制诊断见[内部诊断记录](diagnostics/e15_full_checkpoint/RECORD.md)；
  其 122 ROI 的 oracle/ablation 不代替 EXP025 正式全量评分或最终组合策略比较。
- 未交付：无——两臂全部 16 个固定评价点已交付；跨 seed 复验未开始。
- 未确认：两个 run 的容器 exit code 无独立记录（console 以
  `FINAL_EVAL_REUSED periodic_epoch=40` 收尾，训练进度 100%）。
- 未运行：重构后结构的 route/residual fixed-batch 机制复验；重构前数值不能迁移。
- 未运行：lab1 最终 3e-4 配置的 65536 gate；不据旧 LR negative control声称已测试。
- 不能推出：单个 gate 的短程 loss、资源或梯度覆盖不代表正式姿态精度；固定点序列
  的事后最佳点不构成显著性或选模结论。
- 外置原始产物：服务器 run 目录、本地 `output/diagnostics`、checkpoint 和完整 console log。
  本记录的紧凑副本来自用户保留目录 `E:\6D姿态估计\EXP025\Frozen` 与 `...\Full`（两臂
  `scores_bop19_*epoch.json` 原样复制、`EVAL_SUMMARY` 行提取为 `eval_summary.jsonl`，
  2026-09-23 核对时两目录均含 E5–E40 全部分数与完整 console）。
