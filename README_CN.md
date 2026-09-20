# GDRNPP-RGBD 中文入口

当前唯一活动实验是 [EXP025](research/exp025/README.md)：在统一 T3=512 CAD attention
结构上比较两条正式训练策略。lab0 使用原 GDRNPP LM-O 主干并冻结；lab1 使用 ImageNet
ConvNeXt 初始化并训练完整主干。两臂同时改变初始化来源和主干训练范围，结果只解释为组合
策略差异。

LM-O 仍是当前优先正式实验。LM13 已按历史 GDR-Net 协议准备好 ImageNet ConvNeXt 全量
训练配置、13 类 hierarchy 和本地 CUDA smoke，但尚未加入服务器 launcher，也不会先于
LM-O 启动。

阅读顺序：

1. [当前状态](research/STATUS_CN.md)
2. [EXP025 设计与入口](research/exp025/README.md)
3. [服务器运行手册](research/RUNBOOK_CN.md)
4. [实验索引](research/EXPERIMENT_INDEX.md)

历史实验只保留 `research/experiments/*/RECORD.md` 与紧凑证据；精确复现按 RECORD 中的
source commit 建独立 worktree，不把旧配置或 runner 放回当前 HEAD。`core/` 中保留的历史
实现是稳定代码，但 EXP025 的构建、数据上下文和服务器 profile 不依赖旧实验命名空间。

数据、权重、checkpoint、完整日志、缓存和 secrets 不进入 Git；`output/` 与 `.local/`
只属于本机。
