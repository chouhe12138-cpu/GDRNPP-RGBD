# EXP013E 结构与证据边界

E 在冻结特征协议下从**随机初始化**训练官方 `ConvPnPNet` 头,与 EXP013A/B/C 唯一的
差异是头架构(官方 vs 新头家族);与 EXP005 唯一的差异是初始化(随机 vs 官方预训练
继续)。两个对照共同回答:官方头的 reS 0.5444 有多少来自预训练继承、多少来自结构。

- **随机初始化保护机制(v2)**:头经 `OfficialConvPnPNetRandomInit` wrapper 把官方
  模块嵌套在 `head.*` 子模块下,状态字典键为 `pnp_net.head.*`,与官方 ckpt 的
  `pnp_net.*` 永不重名;官方 warm start 时已知官方 pnp 键被显式丢弃
  (与 EXP012/A/B/C 的 legacy 过滤同机制)。`MODEL.WEIGHTS` 保持原始官方 ckpt,
  无任何派生文件。(v1 的 pnp 剥离派生权重方案因服务器 gate→smoke 之间派生文件
  缺失而失败,已废弃;失败 smoke 记录 RUN-20260829-063652-smoke-s42-a01。)
- 头标志完全复刻官方(GN+gelu、flatten fc1、region attention、无 mask 乘性门控);
  不含 EXP013 家族的任何头设置。
- 训练渲染器为**关闭**状态(冻结几何监督关闭,引擎不构造 CPP/EGL 渲染器);
  这不是科学变量。
- E 只能回答"读出结构 vs 预训练继承",不能回答全量训练制度问题(属 EXP013D),
  也不能单独证明 GLM-Pose M2 有效(需与 M3 联合筛选,即 EXP013F)。
