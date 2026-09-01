# EXP011 — CPM XYZ–Region Consistency Diagnostic

## 协议

- 状态：`COMPLETE / MISMATCH_IMPORTANT`
- run：`RUN-20260817-023144-full-s20260817-a02`
- checkpoint：EXP009 E40；LM-O BOP19 1,445 targets；GT bbox；seed `20260817`
- 固定权重、FP32、无训练；只替换 XYZ/Region，其他输入与 support 不变
- 2×2 endpoints：Pred/GT XYZ × Pred/GT Region；full 扩展为两种 Region 下
  α=`0/.25/.50/.75/1`

GT Region 是由 metric GT XYZ 生成的 hard one-hot，而 Pred Region 是 soft
posterior，因此 Region 主效应混合了标签正确性与 entropy/sharpness；科学判断
优先使用 XYZ×Region interaction。

早期 smoke/audit 因过严 re-entry 容差失败，首次 full 又因误把 micro baseline
当 macro calibration 而标记 QC failure。它们都完成了预期计算，但不进入最终
判决；a02 修正口径后另建目录，1,445×10 conditions 完整且 baseline 复现通过。

## 最终结果

- baseline BOP AR `0.598404`，与 EXP009 E40 `0.598392` 一致。
- baseline ADD(-S) target-micro `0.380623`，macro-object `0.376751`。
- Pred/GT Region argmax agreement `0.572080`；Pred posterior 对 GT label 的平均
  概率 `0.564918`。

| 指标 | GT-XYZ effect / Pred Region | GT-XYZ effect / GT Region | interaction | rescue ratio |
|---|---:|---:|---:|---:|
| BOP AR | -0.289476 | -0.138113 | +0.151363 | 0.522887 |
| ADD(-S) macro-object | -0.279648 | -0.102498 | +0.177150 | 0.633474 |

8/8 物体的 ADD interaction 为正，超过预注册 ≥5/8 gate；两项主指标 interaction
均为正且 rescue ratio ≥0.5，因此 decision 为 `MISMATCH_IMPORTANT`。

完整 alpha sweep BOP AR：

| Region | α=0 | α=.25 | α=.50 | α=.75 | α=1 |
|---|---:|---:|---:|---:|---:|
| Pred | 0.598404 | 0.559163 | 0.466157 | 0.371013 | 0.308927 |
| GT | 0.492494 | 0.493430 | 0.454685 | 0.398907 | 0.354381 |

GT Region 在 Pred XYZ 下会把 BOP 从 `0.598404` 降至 `0.492494`，macro ADD 从
`0.376751` 降至 `0.258834`。因此结论是 mismatch 会污染 GT-XYZ intervention，
不是“GT Region 本身提升绝对性能”，也不证明 mismatch 是 CPM 欠佳的唯一根因。
