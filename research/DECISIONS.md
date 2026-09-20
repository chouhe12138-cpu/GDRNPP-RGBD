# 长期研究决策

- 当前进展和唯一下一步只维护在 `STATUS_CN.md`；实验事实与缺口只维护在各自 RECORD。
- smoke、oracle 和 fixed-batch 诊断只证明工程链路或机制响应，不替代正式指标。
- 历史实验执行入口不常驻 HEAD；精确复现使用 RECORD 的 source commit 和独立 worktree。
- Git 只保存代码、有效配置、RECORD 和紧凑证据；数据、权重、完整日志和缓存外置。
- 服务器只运行本地提交生成的只读 release；正式训练期间不修改 checkout 或镜像。
- 正式模型只按预定评价点判断，不事后改变 gate，不自动多 seed 或按中间结果选模。
