# EXP-20260817-012 层级密集 Correspondence Pose Head

## 当前状态

- 状态：`COMPLETE / CLOSED_AFTER_E15_DIAGNOSTICS`。
- 正式实现 source commit：`2ca752b3f091292172044209f7c8651280d377bd`。
- 原计划 40 epochs；实际已有 E5/E10/E15 正式评估。E15 出现严重后期泛化退化后，转入无重训练诊断并关闭该结构路线，不把它记为“完成 E40”。
- EXP012 不再默认继续训练、恢复或追加 head rescue；后续进入 EXP013 完整架构阶段。

## 原始研究问题

EXP012 检验：在任何全局压缩前保留逐像素 metric XYZ↔absolute ROI2D 配对，先学习局部关系、再做层级空间聚合，能否比 CPM 更好地消费冻结 geometry 输出中的 pose 信息。

结构保持 backbone/geometry head 冻结，只训练 868,746 参数的 `HierarchicalCorrespondencePnPNet`。Region 仅作为零启动辅助残差，不参与 grouping/pooling；后期仍使用共享 2208→256→256 pose representation 输出 allo rot6d 与 centroid-z translation。

## 正式评估事实

训练损失从 E5→E10→E15 继续下降，但 LM-O pose 泛化没有同步改善：

| Epoch | BOP AR | ADD(-S)@0.1d | AR_reS | AR_teS |
|---:|---:|---:|---:|---:|
| 5 | 0.642973 | 0.489273 | 0.345790 | 0.783391 |
| 10 | 0.645972 | 0.473356 | 0.428835 | 0.773472 |
| 15 | 0.544083 | 0.377163 | 0.124798 | 0.798847 |

E10→E15 的主要退化集中在 rotation：AR_reS 从 `0.428835` 降至 `0.124798`，而 AR_teS 从 `0.773472` 变为 `0.798847`。因此 E15 不是 translation 同步崩塌。

## 无重训练诊断事实

Region×0 显示 EXP012 对 Region 输入依赖很强：BOP AR 在 E5 `0.642784→0.353449`、E10 `0.646028→0.316653`、E15 `0.544002→0.251550`。但依赖强度没有在 E15 突然出现新的数量级变化，因此不能把“Region 依赖突然增强”写成 E15 崩塌原因。

Pred XYZ→GT XYZ 的 alpha sweep 没有带来稳定 pose 改善；在 synced Region 路径下也没有 endpoint rescue。E5/E10 Three-Path 最终 summary 为 QC PASS，E15 Three-Path 为 QC PASS。这个事实说明 learned EXP012 decoder 的输出并不随 point-wise XYZ 改善而稳定改善，但不证明 correspondence 本身无效。

E10↔E15 checkpoint interpolation 只插值 pose-head 浮点参数，冻结参数逐值不变，endpoints QC PASS：

| alpha | BOP AR | AR_reS | AR_teS |
|---:|---:|---:|---:|
| 0.00 | 0.646009 | 0.429066 | 0.773933 |
| 0.25 | 0.636141 | 0.374394 | 0.769781 |
| 0.50 | 0.600161 | 0.265283 | 0.762399 |
| 0.75 | 0.574572 | 0.183622 | 0.772318 |
| 1.00 | 0.544127 | 0.124798 | 0.798847 |

BOP 与 AR_reS 沿参数连线单调下降，线性拟合 R² 分别为 `0.978998` 与 `0.988699`；AR_teS 不呈相同行为。该结果支持“E10→E15 参数连线上存在连续、rotation-specific 的泛化恶化方向”，但不能等同于真实训练轨迹。

Activation drift 诊断 QC=`FAIL`，只能保留为线索，不作为上述机制判断的主要证据。

## 结论边界

**已被实验直接支持：** EXP012 早期 E5/E10 有竞争力但 E15 严重退化；退化主要集中于 rotation；Region 是强依赖输入；改善 XYZ 或同步 Region 没有稳定 rescue learned pose；checkpoint interpolation 的强 QC 证据表明后期退化不是孤立坏 checkpoint。

**合理但尚未证实：** R/t 共享后期 latent 可能造成不同优化需求之间的干扰；learned pose decoder 可能形成了对 Region 或训练分布的 shortcut。这些解释都不能写成 EXP012 已证明结论。

**当前缺失证据：** 在相同轻量 correspondence frontend 下，显式几何求解与 R/t 非对称 direct regression 的精度、稳定性和真实 latency 对比；以及 R/t 表示解耦是否真正改善后期 rotation stability。

## 决策

EXP012 结构路线关闭，进入 `EXP-20260819-013-correspondence-guided-architecture`。下一阶段不默认继承 GDRNPP 的 Backbone、Region、双 Mask 或 learned pose head，也不默认 PnP 必须作为最终部署结构。所有新模块必须服务于统一的 `correspondence → pose` 信息流，并先通过小规模 screening；未通过前不启动 40-epoch 全量 PBR。
