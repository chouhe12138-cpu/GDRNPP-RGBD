# 当前研究状态

最后核对：2026-09-25（依据用户提供的 EXP026 训练快照、EXP027 lab2 gate 报告与启动报错）
EXP025 formal source commit：`b9bddccef2f12b4365e0e1e2e68222d733de572b`

## Active experiment

`EXP-20260922-026-residual-aligned-sampling-ablation` — 在 matched ImageNet Full 与
predicted-route residual 条件下比较三层普通采样和几何自适应采样。

状态：`FORMAL_RUNNING / PARTIAL_EVAL_AVAILABLE`。两臂已从同一 source commit 启动正式训练：
lab0 `uniform_full`、lab1 `adaptive_l1_full`。截至收到的日志快照，均匀臂完成 E5/E10
评价，日志到 E11；自适应臂完成 E5/E10/E15/E20 评价，日志到 E21。服务器 batch48/EGL
gate 的原始输出未包含在本次材料中，不能据此独立核验 gate。

当前唯一下一步：交付 EXP027-A launcher 修复 bundle，lab2 切换到新 release 并重跑
同 commit 的真实 batch48/EGL gate，PASS 后启动 formal；EXP026 两臂继续
原 formal，不更换 lab0/lab1 checkout、容器或镜像。EXP026 完整固定点评价、逐物体与
correspondence 指标及服务器 gate 原始证据仍待收集，阶段性结果不用于最终结论。

## Prior experiment

`EXP-20260920-025-hierarchical-cad-attention` — 在统一 T3=512 CAD attention 结构上比较
两条 LM-O 主干策略。

状态：`ACTIVE / FORMAL_COMPLETE`（两臂 E5–E40 八个固定评价点全部交付，2026-09-23 核对）。

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
- 全部 16 个预定评价点（两臂各 E5/E10/E15/E20/E25/E30/E35/E40）已记录，见 EXP025
  RECORD；console 显示两臂自然完成 E40 并 `FINAL_EVAL_REUSED`，无 error/NaN。

EXP025 的最终组合策略结论仍待用户判定；本次仅授权推进独立的 EXP026，不选 EXP025 模型、
不改其协议，也不追加 seed。

## Next stage

EXP027 多尺度 CAD Interaction A/B 已完成本地实现、parent-query gate 收口及 batch4 工程
复核；B 固定批次后期 route 回升得到缓解。用户决定先在 lab2/GPU 2 独立运行 A；
lab2 已建立首版 release/容器，补齐遗漏的 VOC 列表后 batch48/EGL gate 报告 PASS；
formal 启动因 launcher 未输出整数 AMP scale 而失败，训练尚未启动。本地正在交付
launcher 修复 bundle，需更新 lab2 release/容器并重跑同 commit gate。
B 仍阻断，lab0/lab1 的 EXP026 formal 不动。详见
[EXP027 RECORD](experiments/EXP-20260924-027-multiscale-cad-interaction/RECORD.md) 与
[lab2 A 步骤](exp027/LAB2_A_CN.md)。

EXP025 formal 已完成全部固定点交付；最终组合策略结论只由用户基于 RECORD 判定。

EXP025 内部已完成 Full E15 本地 checkpoint 诊断：匹配预测 T3 route 的 oracle residual 显示
明显 headroom，ImageNet backbone reset 降低姿态表现；它是单 checkpoint 的机制证据，
不改变 EXP025 formal 评价与选模协议，详见 EXP025 内部诊断 RECORD。

EXP026 已保留三层 GA-HFPS 离线比较证据；共同修复 residual 的 ImageNet Full
`uniform_512` vs `adaptive_512_l1` 两臂 matched 配置和本地 unit/CPU/CUDA/fixed-batch
gate 已通过。本地 object-11 固定批次只证明可学习性，本身不构成正式姿态精度证据；
服务器真实 batch48 gate 的原始证据待补存；formal 已启动，现有阶段性指标见其 RECORD。

LM13 已整理为独立 CAD candidate 配置入口，与 LM-O 仅靠配置切换；真实数据、CPU 模型、
本地 CUDA/CPP batch4 和 legacy GT-oracle evaluator 短测通过。本机 EGL 因 bindless textures
不支持而阻塞，服务器 profile 仍禁用；没有 formal 或模型精度结论，详见
[候选记录](lm_candidate/README.md)。第二轮结构收口已将 LM/LM-O 都切到中性 CAD
config/helper 公共层，六个入口的完整 effective config 与 `d0bd434` 基线一致；旧 EXP025
import 保留兼容，科学协议与安全 gate 未变。EXP025 收官判定前不启动。

## Historical experiments

EXP000–024 的事实、结论与证据缺口只以 [实验索引](EXPERIMENT_INDEX.md) 和各实验
`RECORD.md` 为准。
