# EXP013F 结构与证据边界

F 在冻结特征协议下训练 **GLM-Pose-L** 头,与 EXP013A 唯一的差异是头部晚期结构
(M2)与深度统计输入(M3);协议常数(40ep / batch 48 / Ranger 8e-4 / seed 42 /
GT 框 / 固定 epoch_040)与 A/B/C 完全一致,单变量=头。

## M1:输入管线逐字继承 A

`GLMPoseLNet` 继承 `XYZResidualBypassPnPNet`:XYZ extent 反归一化、绝对 ROI2D、
Region 零起步残差、可见掩码乘性门控、`_ConvNormAct(5,…)` 主编码器全部原样保留;
删除的只有 A 的几何残差支路与共享晚期解码器(`geometry_*`、`pose_fc1/fc2`)。

## M2:注意力池化替代 flatten/FC

16×16 token 网格 → 线性投影到 256 维 + 固定 2D 位置编码 → 1 层
`nn.TransformerEncoderLayer`(embed 256,8 头,FFN 512,dropout 0,gelu)→
tanh 打分 softmax 注意力池化 → 共享 fc 256→256 → 晚期解耦输出
`pose_rotation 256→6`(allo rot6d)与 `pose_translation 260→3`(centroid_z)。

## M3:深度统计注入平移支路

`INPUT.HEAD_DEPTH=True`(独立开关,默认关闭,对所有现有配置零行为变化):
- 统计:`core/gdrn_modeling/datasets/roi_depth_stats.py`,ROI 裁剪深度上做
  mask-free anchor-band:中心窗中位深度为锚,带 |d−center|≤extent_z 取
  [median/ext_z, center/ext_z, variance/median², band_fraction] 4 维;
  训练(在线类)与测试(离线类)同法同参。
- 注入:`GDRN_double_mask` 仅当 `PNP_NET.INIT_CFG.use_depth_stats=True` 且
  batch 携带 `roi_depth_stats` 时,以 `depth_stats` kwarg 传给头;A/B/C/E 头
  签名不变、永远收不到该参数。`depth_stats=None` 时头内零填充
  (诊断/smoke 兼容,preflight 断言零填充与显式零向量严格相等、旋转对深度
  不敏感、平移对深度敏感)。
- 深度只做统计,**不**拼接进 backbone 输入(官方 3 通道主干不动);
  `HEAD_DEPTH` 与 `BP_DEPTH` 互斥(初始化即拒绝)。

## 保证与边界

- **渲染器为关闭状态**(冻结几何监督关闭,引擎不构造 CPP/EGL 渲染器),
  不是科学变量;EGL 仅属 EXP013D。
- 参数预算:头 929,175(80 万–110 万 preflight 硬断言);损失零改动
  (ROT allo_rot6d / TRANS centroid_z,XYZ L1 家族)。
- F 回答 M2+M3 联合是否有效,不单独拆分 M2/M3 归因;不能回答全量训练制度
  问题(属 EXP013D)。
- 上次(F v1)失败根因:深度统计只加进了离线数据类 `GDRN_DatasetFromList`,
  而 lmo_pbr 配置 `XYZ_ONLINE=True` 时训练走 `GDRN_Online_DatasetFromList`,
  该类没有 HEAD_DEPTH 支持 → `KeyError: roi_depth_stats`。v2 修复:两个数据
  类(训练在线 + 测试离线)都实现 HEAD_DEPTH,统计一律对 ROI 裁剪深度计算
  (v1 误用整图深度),并修掉 cv2 单通道维度丢失导致的索引错误。
