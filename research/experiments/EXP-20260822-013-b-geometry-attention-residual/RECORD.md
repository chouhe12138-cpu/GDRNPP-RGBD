# EXP013B 运行记录

- 最终状态：`COMPLETE / FAIL_STRICT_E40_BOP_GATE / ATTENTION_EFFECTIVE_VS_A`
- formal run：`RUN-20260822-112724-formal-s42-a01`，lab1，seed 42，40/40 epoch 完成。
- source commit：`e316b9f33abf51cc8d80550e9dfd52ebb0420088`。
- environment image：`sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee`。
- E40 checkpoint：epoch 40 / iteration 255919 / 470 model tensors，SHA-256 `7f4e80c9494b6faf16d1729ccf5aa0e939040d9ad0d901dd3f1ea79bca341d04`。

## E40 正式结果与 gate

| 指标 | EXP012 E40 | EXP013B E40 | 增量 | 门槛 | 结论 |
|---|---:|---:|---:|---:|---|
| BOP AR | 0.678800 | 0.683691 | +0.004891 | ≥ +0.005 | FAIL（差 0.000109） |
| ADD(-S) | 0.494118 | 0.514187 | +0.020069 | ≥ +0.010 | PASS |
| 逐物体非负 | — | 5/8 | — | ≥ 5/8 | PASS |

BOP 增量 `+0.004891` 小于冻结门槛，因此 B 没有同时通过相对 EXP012 的全部 gate，不能用四舍五入事后放宽。

B 与 A 的 BOP 差为 `-0.000265`，在预注册的 `±0.001` 区间内，且 ADD(-S) 比 A 高 `+0.003460`；所以单看 B−A 规则，attention 判为有效。但 C 还要求 B 先通过全部 EXP012 gate，该条件未满足。

AR_reS 为 `0.498039`，AR_teS 为 `0.801153`。相对 EXP012 的逐物体非负项为 ape、driller、duck、eggbox、holepuncher。

## E40 结构诊断

同样使用 seed 42、4 batch × 2 个真实样本；诊断前后权重不变。

- normal：rotation `4.62°`，translation `1.52 cm`。
- 关闭 attention 后为 `8.10° / 2.08 cm`，attention 确实被使用。
- attention scale 从 `0.1` 学到约 `0.177`；geometry scale 约 `0.099`。
- Region 清零后为 `11.52° / 12.56 cm`；主路径和几何路径仍强耦合。
- 共享参数 R/t 梯度余弦均值 `+0.143`，负值 batch 为 `25%`；但平移梯度范数约为旋转的 `39` 倍。
- 打乱 geometry 网格使 rotation 输出平均变化 `59.70°`，空间顺序十分重要。

完整本地结果位于 Git 忽略目录 `output/diagnostics/exp013b_e40/`。
