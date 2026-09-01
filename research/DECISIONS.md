# 长期研究决策

- 使用 GDRNPP/ConvNeXt-Base，LM-O GT-box 作为当前机制研究的主比较协议；检测器
  误差和跨数据集验证单独报告。
- Camera-XYZ RGB-D 融合候选暂缓。当前主线研究 correspondence 到 pose head 的
  信息利用，不自动恢复早期方案。
- 正式模型只由固定评估点判断；smoke、oracle 和结构诊断只解释机制或工程可行性。
- 每次只改变清晰的结构因素，保留 seed 42；不做无目标超参数扫描或自动多 seed。
- Git 只保存代码、当前有效配置和紧凑 RECORD；数据、权重、完整日志和缓存外置。
- 已结束且不再作为直接开发入口的实验配置不要求常驻 HEAD。精确历史复现以
  `RECORD.md` 中记录的 source commit 为准，恢复当时代码与配置，而不是让旧
  config 在当前 core 上运行。
- EXP005 PnP-only 保留一个按当前配置体系重建的长期 matched control；它用于未来
  方法统一比较，不替代 EXP005 历史 source commit 的精确复现。
- 服务器只是运行端，本地工作区是唯一代码修改来源；服务器隔离规则优先于运行
  便利性。
- 2026-09-01 起采用轻量实验工作流：一份 RECORD、一个中央索引、一个安全启动器，
  不再维护环境身份、哈希证据链、manifest/state/index 多层框架。
- “紧凑 RECORD”不等于只保留状态：正式曲线、指标口径、预注册门槛、决定性
  逐物体/条件结果和失败边界属于科学证据，必须随 RECORD 保留；只删除重复状态、
  文件哈希和基础设施审计噪声。
- 不恢复 `experiment_system`、`managed_runtime`、旧 server scripts 或旧
  EXPERIMENT/ACCEPTANCE JSON；当前 E/F 代码直接保留，不从历史分支重新合并。
