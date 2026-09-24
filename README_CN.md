# GDRNPP-RGBD 中文入口

当前主线是 EXP026：在 LM-O 上比较普通与几何自适应三层 CAD 采样；设计和证据见
[EXP026 RECORD](research/experiments/EXP-20260922-026-residual-aligned-sampling-ablation/RECORD.md)。
下一阶段 EXP027 多尺度 CAD Interaction A/B 已完成本地框架和验证，见
[EXP027 RECORD](research/experiments/EXP-20260924-027-multiscale-cad-interaction/RECORD.md)。
EXP025 已交付两臂全部固定评价点，最终策略结论待用户判定。当前进展和唯一下一步只维护在
[当前状态](research/STATUS_CN.md)。

阅读顺序：

1. [当前状态](research/STATUS_CN.md)
2. [EXP026 协议与证据](research/experiments/EXP-20260922-026-residual-aligned-sampling-ablation/RECORD.md)
3. [服务器运行手册](research/RUNBOOK_CN.md)
4. [实验索引](research/EXPERIMENT_INDEX.md)

历史实验只保留 `research/experiments/*/RECORD.md` 与紧凑证据；精确复现按 RECORD 中的
source commit 建独立 worktree，不把旧配置或 runner 放回当前 HEAD。

LM13 的非正式 CAD 候选配置与本地检查见 [LM candidate](research/lm_candidate/README.md)。

数据、权重、checkpoint、完整日志、缓存和 secrets 不进入 Git；`output/` 与 `.local/`
只属于本机。
