# EXP013A 运行记录

- 最终状态：`COMPLETE / PASS_E40_FORMAL_GATE`
- formal run：`RUN-20260822-112714-formal-s42-a01`，lab0，seed 42，40/40 epoch 完成。
- source commit：`e316b9f33abf51cc8d80550e9dfd52ebb0420088`。
- environment image：`sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee`。
- E40 checkpoint：epoch 40 / iteration 255919 / 463 model tensors，SHA-256 `00466999836f2406321b14e325bc84dbae6c58bd6db40372aaad357784118faa`。

## E40 正式结果与 gate

| 指标 | EXP012 E40 | EXP013A E40 | 增量 | 门槛 | 结论 |
|---|---:|---:|---:|---:|---|
| BOP AR | 0.678800 | 0.683956 | +0.005156 | ≥ +0.005 | PASS |
| ADD(-S) | 0.494118 | 0.510727 | +0.016609 | ≥ +0.010 | PASS |
| 逐物体非负 | — | 5/8 | — | ≥ 5/8 | PASS |

AR_reS 为 `0.498039`，AR_teS 为 `0.797693`。相对 EXP012 的逐物体非负项为 ape、can、cat、duck、eggbox；三项 gate 全部通过。结果支持“Region-free XYZ–ROI2D 独立残差路径改善了当前 pose head 的几何信息利用”，不支持删除 Region。

## E40 结构诊断

固定 seed 42、4 batch × 2 个真实 LM-O PBR 样本。诊断不是正式 BOP 评估，只用于解释机制；诊断前后 pose-head state SHA-256 一致。

- normal：rotation `4.33°`，translation `1.35 cm`。
- 关闭几何残差后为 `46.98° / 15.29 cm`；几何支路单独工作时 translation `75.18 cm`。主路径和残差路径形成互补，并非两条独立完整 pose 路径。
- Region 清零后为 `18.01° / 13.22 cm`，Region 仍是有效且重要的输入。
- Pred XYZ 逐步替换为 GT XYZ 没有单调改善，correspondence-to-pose 接口仍存在分布适配问题。
- 共享参数 R/t 梯度余弦均值 `-0.049`，4个 batch 中 `50%` 为负；支持继续研究 R/t 解耦，但不证明 C 会提高正式精度。
- 打乱 geometry 8×8 网格使 rotation 输出平均变化 `47.30°`，空间顺序被强烈使用。

完整本地结果位于 Git 忽略目录 `output/diagnostics/exp013a_e40/`。
