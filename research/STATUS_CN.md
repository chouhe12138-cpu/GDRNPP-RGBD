# 当前研究状态

最后核对：2026-09-20。

## EXP025

状态：`STRUCTURE_REFACTORED / LOCAL_SMOKE_PASS / SERVER_BATCH48_GATE_PENDING /
FORMAL_NOT_READY`。

当前模型使用四级 Image-SA 写回、统一 T3=512 分类、T2/T1 marginal NLL、预测 soft-T3
条件化的 Residual V2、可见 mask BCE 和固定 `consistent_v3` hierarchy。optimizer resume、
GradScaler 跳步时 scheduler gate 与可配置 `SOLVER.AMP.INIT_SCALE` 已接入生产训练路径。

正式比较包含两个明确 arm：

| arm | server | initialization | backbone |
|---|---|---|---|
| `official_frozen` | lab0 / GPU 0 | 原 GDRNPP LM-O checkpoint | frozen |
| `imagenet_full` | lab1 / GPU 1 | ConvNeXt ImageNet checkpoint | trainable，LR×0.1 |

两臂共享 LM-O PBR40、GT-box、seed42、真实 batch48、AdamW、40 epoch、E5–E40 固定评价点
和相同 EXP025 head。初始化和训练范围同时变化，不能作单因素冻结消融。

当前两个正式配置均为 `FORMAL_READY=False`，且不设置 AMP 初始 scale。必须先在各自服务器
运行 EGL 真实 batch48 gate，按 `65536 → 32768 → 16384` 测量；选择两臂共同通过的最高
scale。任一臂 batch48 OOM 或 16384 仍失败时停止，不自动改梯度累积，也不启动任一 formal。

历史 EXP000–024 已退出 HEAD 的执行面。所有原始状态、结论和证据缺口继续由
[实验索引](EXPERIMENT_INDEX.md) 链接的 RECORD 保存。
