# C1 历史实验追加复验（2026-08-08）

状态：`PASS_WITH_NOTE`。

本次只读检查了 `E:\6D姿态估计\26-08-02`，没有覆盖原始记录、移动历史
文件或重新训练。

- Epoch 40 checkpoint 的 SHA-256 与 `RECORD.md` 完全一致。
- checkpoint 可按正式 C1 config 严格加载，无 missing/unexpected key。
- 392 个官方模型张量逐位未变；新增 9 个张量全部属于
  `quality_coverage_net`，符合 C1 隔离定义。
- 最终日志直接给出 BOP AR `0.6897416378316032`、ADD(-S)@0.1d 对象宏平均
  `0.5057`，与原记录一致，`C1_SCREEN_FAIL` 结论不变。
- 七个阶段日志的哈希均与 `RECORD.md` 一致。

保留说明：外部目录没有复制完整的 C1 raw evaluator 目录。日志末尾的
summarizer traceback 是历史冒号/等号目录匹配问题；同一日志在报错前已经完成
ADD(-S) 计算并打印 `GT=1517, targets=1445, TP=728, macro=0.5057`，不能把该
traceback 解释为指标缺失。

机器可读详情见 `REVALIDATION_20260808.json`。原 `ACCEPTANCE.json` 是首次验收
快照，继续只读保留。
