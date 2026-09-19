# 当前研究计划

## 目标

2026-09-19 最新安排：先完成 CAD hierarchy 公共加载、几何和诊断的轻量整理，
保持旧模型与训练协议。consistent_v3 的 surface oracle 支持将 T3+residual 作为
下一阶段候选；尚不证明网络路由与残差可学习性。本轮不预留 EXP025、不实现新模型。
下述既有实验安排保留为背景，后续训练等待用户安排。

围绕 LM-O 遮挡场景，研究 GDRNPP dense XYZ、Region、ROI2D 和深度统计如何被
直接姿态头有效利用。当前论文链保持 RGB 主干与 geometry head 的可比性，不把
早期 oracle、PBR calibration 或 smoke 指标当作正式性能结果。

2026-09-14 起主线进入 EXP021：用固定 CAD `64×64` 层级、条件子区域路由和受限
残差约束 dense correspondence 的表面身份，再检验 8×8 全局图像—CAD 交互能否
改善共同姿态模式与 matched classical PnP/RANSAC。V1 只训练新增 CAD head，冻结
官方 backbone/decoder/mask/Patch-PnP。EXP020 已于 2026-09-16 按用户决定结束后续
补证；正式 matched PnP 未运行，作为前序证据缺口保留，不由 EXP021 替代。

2026-09-16 起的 EXP022 第一阶段以相同官方冻结 RGB backbone 和固定 8⁴ CAD hierarchy
研究四级条件对应与多尺度 PCC；2026-09-17 完成 image self-attention / 局部 Q/K/V
结构重构和本地工程对比。服务器 EGL、正式训练和 matched PnP 尚未运行，不能据本地
smoke 判断精度或机制。geometry-adaptive partition 与 fragment adjacency 延后。

2026-09-19 用户另指定 lab1 开展 EXP024：保持 EXP022 的 LM-O/PBR40 与 PCC 方法，
改用 ImageNet ConvNeXt-Base 初始化并训练完整 backbone。EXP022 冻结臂在 lab0 继续；
EXP023 LM13 留待 EXP022 完成后进行。EXP024 与 EXP022 同时改变初始化和冻结状态，
结果比较只能描述组合策略的差异，不作单因素解冻归因。

## 已建立的证据

- Oracle/RANSAC 表明预测 correspondence 中存在可用于姿态的信息。
- 官方 Patch-PnP 对受控 XYZ 改善利用不足，简单低阶 moment、质量覆盖模块和
  联合适应均未形成稳定正式增益。
- EXP012 保留局部 correspondence 后在 E40 达到稳定平台。
- EXP013A 证明独立 XYZ–ROI2D 残差路径有价值；C 说明 R/t 专用表示有 rotation
  收益但不足以保证整体提高。
- E 表明官方头即使随机初始化也能恢复较强 rotation；F 的 attention pooling 与
  深度统计只获得局部改善。
- EXP017 E40 未通过 rotation/BOP 门槛；EXP017-B 的 detach 消融仅有小幅
  rotation/BOP 提升，ADD 下降，未形成整体优势。最终结果见各自 RECORD。
- EXP018 E40 相对 EXP013A 四项聚合指标均略升，但 BOP `+0.002346` 未达到设计阶段
  建议的 `+0.003`；单 seed、小 effect size 不足以支持扩展，已收口。
- EXP019 证明 matched RANSAC/EPro-PnP 能稳定消费逐步改善的 XYZ（α 0→1 单调
  改善），而官方 Patch-PnP 对同样改善不响应；用户 2026-09-08 判定该机制通过，
  并把训练信号（而非求解器本身）确定为新主线要解决的问题。

## 后续决策顺序

1. EXP017/EXP017-B/EXP018 已收口，不安排追加训练或诊断。
2. EXP019 已完成 engineering preflight、32-target smoke 与 1,445-target full，
   未训练模型。原始 Gate A/B 通过，历史复现六点中五点越界；原 evaluator decision
   为 `PROTOCOL_REPRODUCTION_FAILED_STOP`（保留为历史输出）。2026-09-08 用户
   review 判定机制通过。完整事实见
   [EXP019 RECORD](experiments/EXP-20260907-019-epro-geometry-utilization/RECORD.md)，
   分析与口径见 [review](notes/20260908-solver-in-the-loop-review.md)。
3. EXP020 已完成实现、review-fix、36 项实验测试、EGL server smoke、fixed-support
   evaluator smoke 与梯度尺度标定；A/B 日志已到达 E40，全部 E5–E40 direct-pose
   telemetry 已同步并保存紧凑原始证据。run exit code 与 matched PnP/RANSAC 正式
   评价未生成；B E15/E20/E25 score 与日志存在 epoch 冲突，reS/teS 未核对。
   用户于 2026-09-16 决定结束后续补证，保留上述缺口；详见 RECORD。不以
   direct-pose 趋势代替主结论。
4. EXP021 V1 的 B/C formal 训练已结束（source `effc99b`，两臂均到
   `iter 255919/255920`），E5–E40 八个固定点的常规 direct-pose 结果与逐物体
   ADD(-S)0.1d 已全部记录；仍缺 run exit code 与 固定 support 的 K=1/2/4/8
   matched evaluator。2026-09-19 用户判定当前代码与网络结构设计需要修正，因此
   不在现有设计上直接继续，matched PnP/K sweep 是否补做由用户决定。
   不启动原设计中的 backbone 联合微调。
   EXP022 第一阶段在本机重构后仍须先过服务器 EGL 真 batch 资源检查；与 EXP021
   的正式 matched comparator 待其证据补齐后确定。
5. 需要训练的新实验仍由用户确认后才在分配的 L40/GPU 上启动；固定比较点，不按 LM-O 中间结果
   选择模型。
6. D 保持暂停，除非用户明确恢复并重新定义其显存与 renderer 方案。

## 结果口径

正式比较至少报告 BOP AR、ADD(-S) 聚合口径、AR_reS、AR_teS 和逐物体趋势。
一次边缘结果不自动触发多 seed；需要重复时必须说明它解决的具体不确定性。
