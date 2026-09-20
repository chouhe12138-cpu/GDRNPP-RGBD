# 当前研究计划

EXP025 检验统一层级 CAD attention 在两种主干策略下的正式 LM-O 表现：lab0 为原
GDRNPP LM-O 主干冻结臂，lab1 为 ImageNet ConvNeXt 全量训练臂。两者是组合策略比较。

正式训练前的 gate 已执行完：两条配置的 CPU preflight；服务器 EGL 真实 batch48、8 步
AMP；有限 loss/gradient；optimizer update；冻结/更新范围；四级 Image-SA 更新；checkpoint
和 scaler 往返；峰值显存。65536 在两臂同一步溢出、32768 两臂均通过，共同 AMP 初始 scale
固定为 32768 并已开放 formal。lab1 的主干 lr 在 formal 前由 3e-5 改为 3e-4（与头同值，
依据见 EXP025 RECORD 的 Decision），需在最终配置下重跑一次 gate。

正式运行固定 seed42、40 epoch、E5/E10/E15/E20/E25/E30/E35/E40 评价，不按中间结果选模，
不自动改 batch、追加 seed 或恢复旧实验。结果至少记录 BOP AR、ADD(-S)、reS、teS、逐物体
趋势、run ID、source commit、checkpoint 和退出状态。

LM13 已作为下一阶段配置准备，但执行顺序固定在 LM-O 之后。其协议沿用历史 GDR-Net 的
real+DeepIM、160 epoch flat-and-anneal 配置；当前只保留本地 preflight/smoke 证据，不开放
服务器 profile 或 formal。
