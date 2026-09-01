# EXP006 — Quality/Coverage Residual Attention

- 状态：`COMPLETE / C1_SCREEN_FAIL`
- 协议：冻结官方网络，只训练 identity-initialized quality/coverage 模块
- 固定 E40：BOP AR `0.689742`；ADD(-S) macro-object 约 `0.5057`

40 epoch 曲线没有形成稳定、达到门槛的提升，正式结论为
`C1_SCREEN_FAIL`。历史逐 epoch 数值和执行实现可由 Git 历史恢复；当前树只
保留结论，避免继续维护已关闭方案。
