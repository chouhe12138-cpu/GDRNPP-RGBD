# EXP013C 结构与证据边界

C 固定继承 B 的 correspondence encoder 和局部几何注意力，不根据 A/B 数值临时改结构。rotation 使用主分支空间网格与几何 `8×8` 描述；translation 使用主分支全局统计、几何均值/方差、ROI2D 支持中心与尺度以及 object extent。两个分支各有独立两层 MLP，不存在共享后期 pose FC，也不增加跨分支通信。

C 只能回答后期 R/t 专用表示是否比 B 更合适，不能把 A 或 B 未解决的问题归因于共享 latent。
