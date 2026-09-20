# 长期研究决策

- 当前唯一活动实验为 EXP025；EXP000–024 的事实与缺口以各自 RECORD 为准。
- lab0 运行 `official_frozen`，lab1 运行 `imagenet_full`；两臂差异不能解释为单一冻结效应。
- smoke、oracle 和 fixed-batch 诊断只证明工程链路或机制响应，不替代正式指标。
- 历史实验执行入口不常驻 HEAD；精确复现使用 RECORD 的 source commit 和独立 worktree。
- Git 只保存代码、有效配置、RECORD 和紧凑证据；数据、权重、完整日志和缓存外置。
- 服务器只运行本地提交生成的只读 release；正式训练期间不修改 checkout 或镜像。
- 正式模型只按预定评价点判断，不事后改变 gate，不自动多 seed 或按中间结果选模。
