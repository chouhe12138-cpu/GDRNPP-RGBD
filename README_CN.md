# GDRNPP-RGBD 中文入口

当前主线是 [EXP025](research/exp025/README.md)：在 LM-O 上比较统一 T3=512 CAD attention
的两条主干策略。当前进展和唯一下一步只维护在 [当前状态](research/STATUS_CN.md)。

阅读顺序：

1. [当前状态](research/STATUS_CN.md)
2. [EXP025 设计与入口](research/exp025/README.md)
3. [服务器运行手册](research/RUNBOOK_CN.md)
4. [实验索引](research/EXPERIMENT_INDEX.md)

历史实验只保留 `research/experiments/*/RECORD.md` 与紧凑证据；精确复现按 RECORD 中的
source commit 建独立 worktree，不把旧配置或 runner 放回当前 HEAD。

LM13 的非正式 CAD 候选配置与本地检查见 [LM candidate](research/lm_candidate/README.md)。

数据、权重、checkpoint、完整日志、缓存和 secrets 不进入 Git；`output/` 与 `.local/`
只属于本机。
