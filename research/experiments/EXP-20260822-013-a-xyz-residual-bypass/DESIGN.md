# EXP013A 结构与证据边界

## 唯一结构变化

EXP012 的主分支与 Region 输入保持不变。visible mask 先作用于 metric XYZ 和 absolute ROI2D，随后一条不读取 Region 的 `5→32→48→32` 支路保留到 `8×8` 的空间描述，经线性层形成 256 维几何特征，以初值 `0.1` 的可学习标量加入共享 pose latent。

## 可回答的问题

若 A 通过正式门槛，可支持“独立 XYZ 到姿态输出的路径改善了当前结构中的几何信息利用”。它不能单独证明 Region 有害，也不能证明局部 attention 或 R/t 解耦有效。
