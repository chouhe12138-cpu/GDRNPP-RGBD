# EXP013C 结构与证据边界

C 在正式运行前于 2026-08-26 明确修订为继承已通过 gate 的 A，不再包含 B 的局部几何注意力。该修订发生在 B 严格 BOP gate 失败之后，必须作为 protocol revision 记录，不能声称它仍是原先预注册的 B-based C。

rotation 使用主分支空间网格与几何 `8×8` 描述；translation 使用主分支全局统计、几何均值/方差、ROI2D 支持中心与尺度以及 object extent。两个分支各有独立两层 MLP和独立可学习 `geometry_scale_r/t`，不存在共享后期 pose FC，也不增加跨分支通信。

训练工程上，Geometry Head 冻结时显式设置 `TRAIN_SUPERVISION=False`：预测 Geometry Head forward 继续执行，online mapper 和预测 XYZ/Region/Mask 仍保留，但不初始化 renderer、不生成不会参与 loss 的 GT XYZ/Region。该优化不是科学变量；未来解冻 Geometry Head 前必须恢复监督。

C 只能回答后期 R/t 专用表示是否比 A 更合适，不能归因局部 attention，也不能仅凭机制诊断断言共享 latent 是 A 的唯一瓶颈。
