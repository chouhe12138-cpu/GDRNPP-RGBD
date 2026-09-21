# 当前研究状态

最后核对：2026-09-21
verified commit：`0ec5f9ad72777d37f47b4489e252ffb09cb68413`

## Active experiment

`EXP-20260920-025-hierarchical-cad-attention` — 在统一 T3=512 CAD attention 结构上比较
两条 LM-O 主干策略。

状态：`ACTIVE / FORMAL_NOT_STARTED`（formal 前置条件已满足）。

正式比较包含两个组合策略，不能解释为单一冻结消融：

| arm | server | initialization | trainable parameters |
|---|---|---|---|
| `official_frozen` | lab0 / GPU 0 | 原 GDRNPP LM-O checkpoint | CAD head；backbone frozen |
| `imagenet_full` | lab1 / GPU 1 | ConvNeXt ImageNet checkpoint | CAD head + backbone，均为 3e-4 |

两臂共享 LM-O PBR40、GT-box、seed42、真实 batch48、AdamW、40 epoch、E5–E40 固定评价点、
AMP 初始 scale 32768 和相同 EXP025 head。lab1 的最终 `BACKBONE_LR_MULT=1.0`。

已完成前置：

- lab0 在 EGL 真实 batch48、scale 32768 上 PASS；0 skipped step、四级 Image-SA 90/90
  参数更新、checkpoint 往返一致。
- lab1 已在最终 backbone/head 均为 3e-4 的配置上重跑同一 gate 并 PASS，证据来自
  `exp025_server_gate_imagenet_full_a03.json`。
- 65536 的历史 gate 在 Mask 头出现缩放后 fp16 非有限；共同正式初始 scale 固定为 32768。
- `SOLVER.AMP.INIT_SCALE=32768` 与 `RESEARCH_PROTOCOL.FORMAL_READY=True` 已写入正式配置。

当前唯一下一步：

- 从同一 clean source release 启动 lab0 `official_frozen` 与 lab1 `imagenet_full` 的 40 epoch
  formal；按 E5/E10/E15/E20/E25/E30/E35/E40 记录全部预定指标，不按中间结果选模。

## Next stage

EXP026 已完成独立三层 GA-HFPS 本地离线比较；λ1 可作为 EXP025 formal 完成后
`official_frozen` matched 训练的候选，但尚无训练或姿态精度证据，详见其 RECORD。

LM13 已完成 ImageNet ConvNeXt 全量训练配置、hierarchy 和本地 CPU/CUDA smoke；服务器
profile 仍禁用，LM-O 完成前不启动。

## Historical experiments

EXP000–024 的事实、结论与证据缺口只以 [实验索引](EXPERIMENT_INDEX.md) 和各实验
`RECORD.md` 为准。
