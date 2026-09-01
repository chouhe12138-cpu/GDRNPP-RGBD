# 当前研究计划

## 目标

围绕 LM-O 遮挡场景，研究 GDRNPP dense XYZ、Region、ROI2D 和深度统计如何被
直接姿态头有效利用。当前论文链保持 RGB 主干与 geometry head 的可比性，不把
早期 oracle、PBR calibration 或 smoke 指标当作正式性能结果。

## 已建立的证据

- Oracle/RANSAC 表明预测 correspondence 中存在可用于姿态的信息。
- 官方 Patch-PnP 对受控 XYZ 改善利用不足，简单低阶 moment、质量覆盖模块和
  联合适应均未形成稳定正式增益。
- EXP012 保留局部 correspondence 后在 E40 达到稳定平台。
- EXP013A 证明独立 XYZ–ROI2D 残差路径有价值；C 说明 R/t 专用表示有 rotation
  收益但不足以保证整体提高。
- E 表明官方头即使随机初始化也能恢复较强 rotation；F 的 attention pooling 与
  深度统计只获得局部改善。

## 后续决策顺序

1. 先从 E/F 与 EXP012/013A/C 的差异中提出一个单变量、可证伪的新假设。
2. 在本地只做配置加载、模型接口和小规模真实数据 smoke。
3. 用户确认后才在分配的 L40/GPU 上正式训练；固定比较点，不按 LM-O 中间结果
   选择模型。
4. D 保持暂停，除非用户明确恢复并重新定义其显存与 renderer 方案。

## 结果口径

正式比较至少报告 BOP AR、ADD(-S) 聚合口径、AR_reS、AR_teS 和逐物体趋势。
一次边缘结果不自动触发多 seed；需要重复时必须说明它解决的具体不确定性。
