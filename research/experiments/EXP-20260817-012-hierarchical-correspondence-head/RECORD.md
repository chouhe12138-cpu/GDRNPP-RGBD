# EXP012 — Hierarchical Correspondence Pose Head

## 协议

- 状态：`COMPLETE / EPOCH_040_STABLE_PLATEAU`
- 问题：在全局压缩前保留局部 XYZ–ROI2D 配对，能否改善直接姿态读出？
- LM-PBR train、LM-O BOP19 GT-box；seed `42`；40 epoch；每 5 epoch 评估
- 官方 checkpoint 初始化；冻结 backbone/geometry head，只训练 pose head
- source commit：`2ca752b3f091292172044209f7c8651280d377bd`
- 配置：`configs/gdrn/lmo_pbr/research/exp012_hierarchical_corr_head/train.py`
- checkpoint：`model_epoch_040.pth`；固定 E40，不按中间 LM-O 结果选模

## 正式轨迹

| Epoch | BOP AR | ADD(-S)@0.1d | AR_reS | AR_teS |
|---:|---:|---:|---:|---:|
| 5 | 0.642973 | 0.489273 | 0.345790 | 0.783391 |
| 10 | 0.645972 | 0.473356 | 0.428835 | 0.773472 |
| 15 | 0.544083 | 0.377163 | 0.124798 | 0.798847 |
| 20 | 0.644362 | 0.428374 | — | — |
| 25 | 0.671084 | 0.474048 | — | — |
| 30 | 0.678768 | 0.493426 | — | — |
| 35 | 0.679061 | 0.492042 | — | — |
| 40 | **0.678800** | **0.494118** | **0.491349** | **0.791926** |

E15 是 rotation-specific 临时退化；E20 后恢复，E30–E40 稳定在 BOP 约 `0.679`。
现存归档没有可靠的 E40 逐物体表，因此不补造该项。

## 结构诊断与边界

- Region×0 的 BOP：E5 `0.353449`、E10 `0.316653`、E15 `0.251550`；对应 normal
  为 `0.642784/0.646028/0.544002`，说明输出强依赖 Region。
- E10→E15 checkpoint interpolation 的 BOP 与 AR_reS 单调下降，线性拟合
  `R²=0.978998/0.988699`；AR_teS 的 `R²=0.359859`，退化主要集中于 rotation。
- 后续 E20–E40 的恢复证明 interpolation 只描述该参数连线，不能预测训练终态。
- 这些结果没有证明 Region 依赖有害，也没有证明共享 R/t latent 是唯一原因。

E40 的四项值是 EXP013 的冻结比较基准。
