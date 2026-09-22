# 当前研究状态

最后核对：2026-09-22
verified commit：`b9bddccef2f12b4365e0e1e2e68222d733de572b`

## Active experiment

`EXP-20260920-025-hierarchical-cad-attention` — 在统一 T3=512 CAD attention 结构上比较
两条 LM-O 主干策略。

状态：`ACTIVE / FORMAL_IN_PROGRESS`（Frozen 已评估到 E20；Full 已评估到 E25）。

正式比较包含两个组合策略，不能解释为单一冻结消融：

| arm | server | initialization | trainable parameters |
|---|---|---|---|
| `official_frozen` | lab0 / GPU 0 | 原 GDRNPP LM-O checkpoint | CAD head；backbone frozen |
| `imagenet_full` | lab1 / GPU 1 | ConvNeXt ImageNet checkpoint | CAD head + backbone，均为 3e-4 |

两臂共享 LM-O PBR40、GT-box、seed42、真实 batch48、AdamW、40 epoch、E5–E40 固定评价点、
AMP 初始 scale 32768 和相同 EXP025 head。lab1 的最终 `BACKBONE_LR_MULT=1.0`。

已完成：

- 前置 gate：lab0 与 lab1 分别在 EGL 真实 batch48、scale 32768 上 PASS（0 skipped step、
  四级 Image-SA 90/90 参数更新、checkpoint 往返一致）；65536 的历史 gate 在 Mask 头
  缩放后 fp16 非有限，两臂共同正式初始 scale 固定为 32768。
- 两臂 formal 已于 2026-09-20 从同一 release（`b9bddcc`）启动，run 分别为 lab0
  `RUN-20260920-120844-formal-s42-a01`、lab1 `RUN-20260920-120843-formal-s42-a01`。
- 九个预定评价点（Frozen E5/E10/E15/E20；Full E5/E10/E15/E20/E25）的指标已记录，
  见 EXP025 RECORD；其余固定点未交付。

当前唯一下一步：

- 等两臂剩余 E25/E30/E35/E40 完成并按预定评价点逐点记录；不按中间结果选模、不改协议、
  不追加 seed。

## Next stage

EXP025 内部已完成 Full E15 本地 checkpoint 诊断：匹配预测 T3 route 的 oracle residual 显示
明显 headroom，ImageNet backbone reset 降低姿态表现；它是单 checkpoint 的机制证据，
不改变 EXP025 formal 评价与选模协议，详见 EXP025 内部诊断 RECORD。

EXP026 已保留三层 GA-HFPS 离线比较证据；共同修复 residual 的 ImageNet Full
`uniform_512` vs `adaptive_512_l1` 两臂 matched 配置和本地 unit/CPU/CUDA/fixed-batch
gate 已通过，状态 `LOCAL_FORMAL_READY / SERVER_BLOCKED`。本地 object-11 固定批次只证明
可学习性，尚无正式训练或姿态精度证据；需等待 EXP025 完成并由用户另行放行，详见其 RECORD。

LM13 已完成 ImageNet ConvNeXt 全量训练配置、hierarchy 和本地 CPU/CUDA smoke；服务器
profile 仍禁用，LM-O 完成前不启动。

## Historical experiments

EXP000–024 的事实、结论与证据缺口只以 [实验索引](EXPERIMENT_INDEX.md) 和各实验
`RECORD.md` 为准。
