# 当前研究计划

EXP025 检验统一层级 CAD attention 在两种主干策略下的正式 LM-O 表现：lab0 为原
GDRNPP LM-O 主干冻结臂，lab1 为 ImageNet ConvNeXt 全量训练臂。两者是组合策略比较。

正式训练前只完成以下 gate：两条配置的 CPU preflight；服务器 EGL 真实 batch48、8 步
AMP；有限 loss/gradient；optimizer update；冻结/更新范围；四级 Image-SA 更新；checkpoint
和 scaler 往返；峰值显存。两臂共同通过后固定共同 AMP 初始 scale并同时开放 formal。

正式运行固定 seed42、40 epoch、E5/E10/E15/E20/E25/E30/E35/E40 评价，不按中间结果选模，
不自动改 batch、追加 seed 或恢复旧实验。结果至少记录 BOP AR、ADD(-S)、reS、teS、逐物体
趋势、run ID、source commit、checkpoint 和退出状态。
