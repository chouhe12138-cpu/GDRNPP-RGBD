# EXP013A — XYZ–ROI2D Residual Bypass

## 协议与运行

- 状态：`COMPLETE / PASS_E40_FORMAL_GATE`
- 相对 EXP012 只增加 Region-free XYZ–ROI2D 独立残差路径
- formal run：`RUN-20260822-112714-formal-s42-a01`；lab0；seed 42；40 epoch
- source commit：`e316b9f33abf51cc8d80550e9dfd52ebb0420088`
- 配置：`configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual/train.py`
- checkpoint：`model_epoch_040.pth`，epoch 40 / iteration 255919

## E40 正式 gate

| 指标 | EXP012 | A E40 | 增量 | 门槛 | 判决 |
|---|---:|---:|---:|---:|---|
| BOP AR | 0.678800 | 0.683956 | +0.005156 | ≥ +0.005 | PASS |
| ADD(-S) | 0.494118 | 0.510727 | +0.016609 | ≥ +0.010 | PASS |
| 逐物体非负 | — | 5/8 | — | ≥ 5/8 | PASS |

AR_reS `0.498039`，AR_teS `0.797693`。逐物体非负的是 ape、can、cat、duck、
eggbox。三项全部通过。

## E40 结构诊断

4 batch × 2 个真实 LM-PBR 样本，只作机制解释：normal 为 `4.33°/1.35 cm`；关闭
geometry residual 后为 `46.98°/15.29 cm`；Region 清零后为
`18.01°/13.22 cm`。打乱 geometry 8×8 网格使 rotation 平均变化 `47.30°`。
共享参数 R/t 梯度余弦均值 `-0.049`，4 batch 中 50% 为负。

结果支持独立几何残差路径并实际使用空间布局，不支持删除 Region，也不能仅凭
小样本梯度诊断断言 R/t 解耦一定提高正式精度。
