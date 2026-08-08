# EXP-20260731-006-quality-coverage 历史验收

- 状态：`PENDING_EXTERNAL`
- 验收 commit：`4edab641cbe0aa43e9220d92d4f785a2b920cb31`
- 范围：本地 C1 pilot 与 formal 配置；Epoch 40 checkpoint、评估产物等待服务器只读核验
- 检查数：9（PASS=7，PENDING_EXTERNAL=2）

原始 `RECORD.md` 和历史 `output/` 未被覆盖。详细机器可读结果见 `ACCEPTANCE.json`。

## 备注

- formal Epoch 40 指标来自历史服务器记录，等待用户执行只读证据脚本。
- 历史 evaluator 曾混用 error:ad_ntop=* 与 error=ad_ntop=* 查找规则；原结果不覆盖。
