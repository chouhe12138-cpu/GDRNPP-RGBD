# EXP021 全局引导的层级 CAD 对应预测

- `experiment_id`: `EXP-20260914-021-global-guided-hierarchical-cad-correspondence`
- 状态：`FORMAL_BC_E40_COMPLETE / EXIT_CODE_UNCONFIRMED / MATCHED_PNP_PENDING`
- 日期：2026-09-14
- seed：42（训练）；20260914（CAD 表面采样）；20260730+目标序号（RANSAC）
- 实现开始时的父 commit：`c2b7c2f`；正式 run 记录实际 release commit
- 用户 2026-09-19 确认正式训练已经结束，本实验的记录与结果按原样提交。

## 研究问题与固定范围

在 RGB、官方 backbone/decoder/mask/Patch-PnP 全冻结的条件下，用固定 CAD
`64×64` 层级、GT-parent 子路由和球形受限残差替换独立 Region+XYZ 输出，是否能
降低表面切向偏移；在此基础上加入 8×8 全局图像—CAD 交互，是否能进一步改善
matched RANSAC-PnP。V1 只做冻结阶段，不执行原方案中的 backbone 联合微调。

对称物体使用 instance-consistent single branch：每个实例在完整 BOP SE(3) 等价
变换中选择加权三项监督总损失最小的一条，coarse/fine/XYZ 共用该分支。

## 对照、训练和评价协议

- A：官方 GDRNPP checkpoint，只用于 GT-box matched evaluator 与 profile。
- B：层级路由 + 受限残差。
- C：B + 两层全局 Transformer、全局粗区域偏置、零初始化 256→1024 残差注入。
- B/C：PBR40、batch 48、16 DataLoader workers、40 epoch、Ranger `lr=8e-4`、
  `wd=0.01`、warmup 200、seed 42；正式训练前用一个真实 batch 校准三项梯度尺度。
- 并行执行固定为 B→lab0/物理 GPU 0、C→lab1/物理 GPU 1；两台 profile 使用相同
  source、镜像、配置、seed、数据和权重，机器分配不随结果调整。
- 主评价：官方 A 固定 support；K=1/2/4/8；matched RANSAC-PnP；完整 LM-O 上报告
  correspondence/normal/tangent/reprojection、路由诊断和 BOP AR/ADD(-S)/reS/teS。
- gate 与命令见 [EXP021 README](../../exp021/README.md)。

## Observed（本地工程证据）

- hierarchy `hierarchy_v1.npz` 已确定性生成到 ignored dataset cache：coarse
  `[8,64,3]`、fine `[8,64,64,3]`；叶子半径范围
  `0.0005100–0.0058643 m`；八对象对称分支数 `[1,1,1,1,1,2,2,1]`。
- EXP021 单元/契约测试：12 passed。覆盖 artifact 形状/生成契约、残差半径、K beam、C
  零初始化、前景监督与反传、完整 SE(3) 变换、B/C 配置和共享初始化、gate。
- 完整 `pytest -q research`：180 passed。初次运行发现 bundle 测试仍要求 `main`，
  而脚本从 `698a8fe` 起已允许任意已附着研究分支；2026-09-14 按用户确认将测试
  同步为 attached-branch + clean-tree + unique-output + bundle-verify 契约，并增加
  临时 `exp-test` 分支实际创建 bundle 的集成覆盖。
- CPU preflight B PASS：233,347 个 trainable 参数；官方 checkpoint 仅缺 14 个
  `cad_head.*` key；只有 CAD head 进入 optimizer；优化步后冻结张量不变。
- CPU preflight C PASS：2,923,587 个 trainable 参数；官方 checkpoint 仅缺 60 个
  `cad_head.*` key；只有 CAD head 进入 optimizer；优化步后冻结张量不变。
- 真实 LM-O 单目标 evaluator 接线 smoke COMPLETE：A 与随机初始化 B 的 K=1/2/4/8
  均完成 fixed-support RANSAC，并生成几何、路由和计时字段。该 smoke 只证明接线，
  随机 B 数值不进入科学结论。
- 本机提升权限后，`pytorch22` 的 PyTorch 2.2.0 / CUDA 12.1 能识别 RTX 4060 Laptop
  8 GB。EGL context 在进入模型计算前因 `Bindless Textures not supported` 失败。
- 显式改用诊断级 CPP renderer 后，真实 batch-1 梯度标定 PASS：原始 CAD-head
  gradient norm coarse/fine/XYZ 为 `7.8406/1.2715/0.06849`；建议权重
  `0.125/1/16`，加权后相对中位数 `0.8943/1.1603/1.0000`。该权重已固定到 B/C
  共享配置，仍须在服务器 EGL batch 上复核。
- 本机 CUDA+CPP batch-1 one-step smoke B/C 均 PASS。B：14 个活跃梯度张量、峰值
  allocated `978,876,416` bytes；C：60 个活跃梯度张量、峰值
  `1,034,792,960` bytes；两臂冻结参数均逐张量不变。报告明确标记
  `formal_renderer_match=false`，不作为正式 EGL smoke。
- 按标定建议更新共享权重后，原始梯度复测完全复现上述数值；B/C smoke 再次 PASS。
  加权 loss 分别为 B `0.5452/4.1316/0.1060`、C
  `0.5181/4.1125/0.05608`。ignored JSON 位于当前 experiment output 下的
  `local-calibration-cpp-b1-weighted-config.json` 与
  `local-smoke-cpp-b1-weighted-config.json`。
- source commit `9399608` 的服务器 EGL 标定、B/C one-step smoke 和 batch-48 audit
  均 PASS，标定仍建议 `0.125/1/16`；batch-48 峰值 allocated 为 B
  `3,056,498,176`、C `4,875,776,512` bytes。用户报告 managed smoke 已完成；但 B
  在 iter 499 的累计耗时为 `0.9508 s/iter`，明显慢于历史 batch-4 smoke，故未启动
  formal，并转入训练性能修复。
- 修复前本机 RTX 4060 + CPP 分段结果：B batch 4 不同真实 batch 总计约
  `0.353 s/iter`；B batch 48 前向+反向约 `12.0 s`，其中 backward 约 `9.6 s`，
  峰值 allocated `3,014 MiB`。逐实例、逐对称分支、逐 coarse-parent 构图导致
  batch 扩展时 backward 超线性退化；PnP 未进入训练路径，B 也未执行 Transformer。
- loss 执行重写后，reference oracle 覆盖重复类别、双对称分支和空前景，三项 loss、
  分支选择、输入及全部 CAD-head 参数梯度均在浮点容差内一致。13 项 EXP021 测试通过。
  本机 B batch 4 连续真实 batch 总计约 `0.292 s/iter`。正式诊断入口经 5 步 warmup
  和 20 个不同真实 batch 测得：B batch 48 平均 forward+backward `1.933 s`、CPP
  全流程 `2.556 s`、`18.78 samples/s`、峰值 `1,618 MiB`；C 分别为 `2.229 s`、
  `2.966 s`、`16.18 samples/s`、`3,380 MiB`。B/C 达到预设模型训练步门槛
  `2.0/2.5 s`，batch-48 吞吐高于 batch 4；本地 CPP 全流程不作为服务器 EGL 门槛。
- source `f29f9a0` 的 lab1/L40/EGL batch-48 profile 使用 2 workers 时，B/C 总耗时
  均值为 `5.487/5.065 s`，其中 DataLoader 均值为 `3.588/2.777 s`；EGL geometry
  仅为 `0.262/0.263 s`，不是主瓶颈。改为 8 workers 后，总耗时均值降至
  `2.214/3.002 s`、中位数 `1.619/2.635 s`，DataLoader 均值降至
  `0.621/0.507 s`，但仍出现 `3.590/4.511 s` 的偶发等待峰值。
- 用户报告随后启动的 FP32 formal 正在运行：截图中 lab0/B 在 iter 999 的全局平均
  `1.8918 s/iter`、lab1/C 在 iter 499 为 `2.9619 s/iter`，峰值 allocated 约
  `5286/5461 MiB`；run ID 与精确 source 尚未提供。用户决定待 AMP release 通过
  服务器 gate 后终止两臂，并从官方 checkpoint 重新开始，不续训 FP32 checkpoint。
- AMP + feature-only 本地实现跳过训练时零权重的 legacy output convolutions，B/C
  均只运行一次 `geo_head.forward_features()`；C 先做全局增强，梯度仍经冻结 decoder
  回到 global-guidance。推理路径未变。EXP021 16 项测试覆盖一次调用、无 legacy
  outputs、C 全局梯度和未来配置 AMP 默认。
- 本机 RTX 4060 + CPP batch-1 AMP smoke B/C PASS：GradScaler `65536→65536`，
  optimizer 确实更新 CAD 参数，冻结参数不变；loss 与 FP32 结果接近。AMP 标定原始
  gradient norm 为 `7.8417/1.2717/0.06850`，仍建议 `0.125/1/16`。
- 同代码、seed、batch 48、16 workers、5 warmup + 20 measured 的本机 CPP matched
  profile：B FP32→AMP 的 forward+backward 为 `2.230→1.851 s`（`-17.0%`），端到端
  median `2.709→2.344 s`（`-13.5%`），peak allocated `1.699→1.717 GB`；C 为
  `2.215→2.029 s`（`-8.4%`）、median `2.647→2.568 s`（`-3.0%`）、peak
  `3.548→3.241 GB`（`-8.6%`）。DataLoader 各有一次 2.9–5.9 秒长尾，本机总耗时
  只用于工程比较，不能替代服务器 EGL gate。
- AMP + feature-only 修改后的完整 `pytest -q research`：184 passed；B/C CPU
  preflight 均 PASS，trainable 参数数与官方 checkpoint 缺失 key 契约未变。
- source `8a736c8`、lab0/L40、真实 batch-2 EGL AMP 标定 PASS。原始 gradient norm
  coarse/fine/XYZ 为 `4.10657/0.90651/0.032990`，服务器建议权重为
  `0.25/1/16`，对应相对中位数 `1.1325/1.0000/0.5823`。该建议与本机 CPP 的
  `0.125/1/16` 不同，按预定规则以正式 renderer 结果更新 B/C 共享配置并重做 release。
- 同 release 的 B/C batch-2 EGL AMP one-step smoke 均 PASS：GradScaler 均为
  `65536→65536`，optimizer step 已执行，冻结参数逐张量不变；活跃梯度张量分别为
  `14/60`，峰值 allocated 为 `992,137,728/1,088,520,192` bytes。
- lab0/L40/EGL、batch 48、16 workers、5 warmup + 20 measured profile 全部 PASS，
  FP32/AMP 均无非有限梯度或跳步。B FP32→AMP：端到端 median
  `1.4572→1.4199 s`（`-2.6%`），mean `1.7832→1.8984 s`（`+6.5%`），峰值
  `1.701→1.717 GB`（`+1.0%`）。C：median `1.6498→1.3350 s`（`-19.1%`），mean
  `1.9897→1.7137 s`（`-13.9%`），峰值 `3.551→3.241 GB`（`-8.7%`）。四组
  DataLoader 最大等待为 `4.78–6.93 s`；B 的 mean 和 forward+backward 未显示稳定
  AMP 加速，因此只把这些结果作为工程资源证据，不把单次长尾差异解释为模型收益。
- 上述六份紧凑原始 JSON 保存在本 RECORD 同目录的 `evidence/8a736c8-lab0-egl/`。
  lab0 已切换到新 release 才能产生这些结果，因此旧 FP32 B 已被替换；其最终 iteration
  尚未随 JSON 提供。lab1 的旧 FP32 C 在本轮 gate 期间继续运行，待更新权重复核通过
  后再精确终止。
- 2026-09-15 本地工程优化将训练 loss 的 `(descriptor,parent)` 路由改为 GPU 排序后
  的 256 点 padded blocks；每个 block 只保留一份 64-way fine anchor/token bank，
  coarse/fine label 使用 FP32 平方距离矩阵，删除训练路径的 `cpu().tolist()`、
  data-dependent Python group loop 和 `torch.cdist`。C 的全局交互与 loss 同时复用一次
  CAD descriptor 编码。hierarchy、三项 loss、对称分支选择和 checkpoint key 未改。
- 同次优化增加 CAD-only online-geometry batch：pose/class/crop K 从原 CPU batch 直接
  提供给现有逐 ROI renderer，并批量送往 GPU backprojection；不修改 EGL/native API，
  不再生成未使用的旧 Region `cdist`、ROI2D、pose points 等 GPU targets。DataLoader
  仍为每步预取，正式 16-worker 策略未改。
- 优化后 EXP021 20 项测试通过，包括旧实现 oracle 与新实现的 label、loss、输入及
  参数梯度等价、跨 256 点 block、重复类别、对称分支、空前景和轻量 online batch；
  完整 `pytest -q research` 为 188 passed。B/C CPU preflight PASS，trainable 参数仍为
  233,347 / 2,923,587，冻结参数不变。
- 本机 RTX 4060 + CPP、batch 2、AMP、固定真实 batch 20-step smoke B/C PASS；GradScaler
  均保持 `65536`，无非有限梯度或跳步。B 总 loss 前/后 5 步均值
  `5.21869→5.11502`，C 为 `5.29877→5.07820`；三项 loss 全程 finite，CAD 参数更新，
  冻结参数不变。峰值 allocated 为 B/C `1.103/1.170 GB`。CPP smoke 只验证本地工程与
  可优化性，不替代服务器 EGL gate。
- 优化后本机 CPP、batch-1、AMP 标定 raw gradient norm 为
  `7.841665/1.271669/0.0684941`，复现优化前 `7.8417/1.2717/0.06850`；本机仍建议
  `0.125/1/16`。这支持真实 batch 上梯度语义未改变，但正式配置继续采用已决定的
  EGL 建议 `0.25/1/16`，等待服务器复核。
- 本机 CPP、batch 48、16 workers、5 warmup + 20 measured 的 AMP profile PASS。
  B forward/backward 均值为 `327.38/99.00 ms`、端到端中位数 `0.8821 s`、峰值
  `2.659 GB`；C 为 `352.09/170.62 ms`、`1.0084 s`、`4.186 GB`。相对上次同机 AMP
  记录，模型前反向由 B/C `1.851/2.029 s` 降至 `0.426/0.523 s`，但峰值显存提高；
  DataLoader 仍出现 `8.95/6.87 s` 最大等待。不同 sampled batch、coarse weight 版本
  与系统长尾限制严格 matched 解释，本次不设置时间 gate，只将方向与资源量记录为
  工程证据。
- 由上述同机记录计算，B/C 模型前反向分别加速 `4.34×/3.88×`，即耗时下降
  `76.96%/74.24%`；端到端中位数分别加速 `2.66×/2.55×`，即耗时下降
  `62.37%/60.73%`。公式为 `speedup = old/new`、`reduction = 1-new/old`；这些派生量
  沿用上一条的可比性限制，不作为 formal 科学 gate。
- 2026-09-15 用户确认远程 formal 已在训练；本次加速属于同一 EXP021 formal 的工程
  实现更新，不新建 experiment_id 或 formal 实验，训练完成后直接在加速版本进入下一
  阶段。当前尚未提供唯一 run_id、source commit、进度与退出状态，完成后再同步记录；
  在此之前不干扰服务器训练。

## Derived / Interpretation / Decision

- Derived：C 比 B 新增约 2.69M trainable 参数；真实延迟和显存必须以 CUDA profile
  为准，不能从参数量推出资源 gate。
- Interpretation：本地证据支持实现契约与梯度隔离，尚不支持任何机制效果结论。
- Interpretation：服务器 w2/w8 对比表明总耗时首先受 DataLoader 并发影响；近期
  batch-48 formal/audit 均使用 16 workers，EXP021 先前继承公共基线的 8 属配置遗漏。
- Interpretation：AMP 主要缩短 tensor-core 友好的前向；CAD 标签、排序、索引和
  FP32 `cdist` 限制 B/C 的端到端收益，C 的本机 DataLoader 长尾进一步稀释总收益。
- Interpretation：lab0 EGL profile 中 AMP 对 C 的端到端中位数和显存有明确工程收益；
  B 的 20-step 结果受 DataLoader 长尾影响，且模型 forward+backward 没有稳定缩短，
  不据此承诺 B formal 的显著加速。AMP 的数值完整性与冻结隔离门禁已经通过。
- Interpretation：本地向量化结果支持此前训练热路径包含 CPU 调度与小 kernel 问题；
  同时 online renderer 和 DataLoader 长尾仍独立存在，不能把端到端波动只归因于 AMP
  或 CAD loss。padded blocks 以额外临时显存换取矩阵化吞吐，当前本机 B/C 峰值仍在
  8 GB 内，但正式资源事实须由新 release 的 L40/EGL profile 复核。
- Decision：本次向量化和 AMP 路径优化只改变工程实现与资源表现，不改变 EXP021 的
  hierarchy、loss、科学变量或结果语义，因此并入现有 formal，不产生新的 formal
  实验。等待当前远程训练完成后，直接进入固定 checkpoint 的下一阶段评价。
- Decision（2026-09-14，后被当前执行状态取代）：原空闲 GPU `≤1.2/1.5 s` 启动 gate
  及实际偏离继续作为 Observed 保留；当时计划以同代码 FP32→AMP 相对提速和数值完整性
  做工程 review，并在新 release 的 EGL 标定、B/C smoke 和 matched profile 后重启。
  该历史决策不再表示 formal 尚未启动。
- Decision：正式 EGL 单 batch 标定优先于本机 CPP 标定；coarse/fine/XYZ 已从
  `0.125/1/16` 更新为 `0.25/1/16`。2026-09-15 用户确认远程 formal 正在训练，当前
  不干扰运行；具体 source 与 gate 证据待训练完成后随 run metadata 一并核对。

## 待生成的正式证据

- 两臂的 run exit code（所获 `console.log` 只到 100% 迭代、E40 checkpoint 保存日志与
  `FINAL_EVAL_REUSED periodic_epoch=40`，没有退出码行，因此不据此断言进程退出状态）。
- 含 GPU route blocks 与 CAD-only online batch 的 L40/EGL profile（工程补充证据，
  不构成新 formal）。
- B/C E5–E40 checkpoint 文件名/epoch 的**服务器文件存在性与内容**核验；所获日志已含
  全部八个保存记录，但不等于对服务器文件的核验。
- E40 A/B/C matched K sweep、完整 fixed-support matched PnP、gate report、
  batch-1 profile。当前只完成了常规 direct-pose 评估，未执行 EXP021 专用
  fixed-support matched RANSAC-PnP。
- 失败 run 保留原因与有效证据边界。

## 2026-09-16 同步：B/C formal E5/E10/E15（Observed）

本节只记录所获训练日志和常规 direct-pose 评估，不执行 EXP021 专用 fixed-support
matched RANSAC-PnP、K sweep、机制 gate 或 batch-1 资源 gate。来源为
`E:\6D姿态估计\EXP021\B` 和 `E:\6D姿态估计\EXP021\C`；两臂各有一份
`console.log` 及 E5/E10/E15 的 `scores_bop19_<epoch>epoch.json`。从日志逐字提取
`EVAL_SUMMARY` JSON，并将六份 score JSON 按字段和值保存为紧凑副本：

- B：[`RUN-20260914-125103-formal-s42-a01/eval_summary.jsonl`](evidence/RUN-20260914-125103-formal-s42-a01/eval_summary.jsonl)；同目录含三份 score JSON。
- C：[`RUN-20260914-125349-formal-s42-a01/eval_summary.jsonl`](evidence/RUN-20260914-125349-formal-s42-a01/eval_summary.jsonl)；同目录含三份 score JSON。

两份 `RUN_INFO` 均记录 seed `42`、source
`effc99b93eb70a9197805a427ce0156365a2bbae`，B 使用 `b_hierarchical.py`，C
使用 `c_global.py`，镜像标签同为
`gdrnpp-research:torch220-cu121-sm89-c0be1ade7ea9`。本次 formal 的 source 是
`effc99b`，不是后续本地 GPU 加速提交 `f01dff5`。日志显示 native FP16 AMP、
从官方 checkpoint 加载并自 iteration 0 开始。E5/E10/E15 的
`model_epoch_005.pth`、`model_epoch_010.pth`、`model_epoch_015.pth` 均有保存日志，
但未核验服务器上权重文件的存在或内容；用户提供的 C 目录另有
`model_epoch_010.pth`，本次未读取权重，也未将其纳入 Git。

六份 score JSON 的 `bop19_average_recall` 与对应 epoch 日志的
`EVAL_SUMMARY.bop_ar` 全部逐点相等。表中 ADD(-S)0.1d 来自
`EVAL_SUMMARY.add_s_0.1d`，不是 score JSON 的 `bop19_average_recall_ad`；
BOP、reS、teS 来自各自的 score JSON。以下三点均为预定中间评估点，并非
E40 最终评价：

| Epoch | 臂 | BOP AR | ADD(-S)0.1d | AR_reS | AR_teS |
|---:|---|---:|---:|---:|---:|
| 5 | B | 0.683502 | 0.525952 | 0.521799 | 0.799539 |
| 5 | C | 0.685772 | 0.498270 | 0.543483 | 0.790542 |
| 10 | B | 0.685010 | 0.526644 | 0.521107 | 0.793080 |
| 10 | C | 0.693179 | 0.560554 | 0.538639 | 0.803460 |
| 15 | B | 0.676270 | 0.489965 | 0.525952 | 0.788697 |
| 15 | C | 0.683116 | 0.491349 | 0.531488 | 0.787313 |

### 相同 epoch 的 C−B 差值（Derived）

由上方原始 JSON 中未舍入的对应数值按 `C − B` 计算，表中仅作显示舍入：

| Epoch | Δ BOP AR | Δ ADD(-S)0.1d | Δ AR_reS | Δ AR_teS |
|---:|---:|---:|---:|---:|
| 5 | +0.002270 | -0.027682 | +0.021684 | -0.008997 |
| 10 | +0.008168 | +0.033910 | +0.017532 | +0.010381 |
| 15 | +0.006847 | +0.001384 | +0.005536 | -0.001384 |

### 已获三个评估点的逐物体 ADD(-S)0.1d（Observed）

以下数值对应 `EVAL_SUMMARY.add_s_obj_recalls`，完整精度见随附 JSONL；
三点全部列出，不把当前 E15 当作完整 formal 的最终点。

| 物体 | B E5 | C E5 | B E10 | C E10 | B E15 | C E15 |
|---|---:|---:|---:|---:|---:|---:|
| ape | 0.440000 | 0.497143 | 0.474286 | 0.468571 | 0.314286 | 0.468571 |
| can | 0.698492 | 0.673367 | 0.703518 | 0.809045 | 0.698492 | 0.743719 |
| cat | 0.450292 | 0.345029 | 0.403509 | 0.385965 | 0.391813 | 0.403509 |
| driller | 0.805000 | 0.815000 | 0.760000 | 0.855000 | 0.785000 | 0.800000 |
| duck | 0.416667 | 0.316667 | 0.450000 | 0.516667 | 0.255556 | 0.233333 |
| eggbox | 0.500000 | 0.305556 | 0.483333 | 0.405556 | 0.450000 | 0.233333 |
| glue | 0.671429 | 0.735714 | 0.721429 | 0.764286 | 0.600000 | 0.728571 |
| holepuncher | 0.235000 | 0.310000 | 0.240000 | 0.285000 | 0.395000 | 0.325000 |

日志副本的最后一条 `my_writer`：B 为 `20260916_011855`、epoch 16、
iter `100499/255920`（39.3%）、`max_mem=1738M`；C 为 `20260916_045330`、
epoch 16、iter `97499/255920`（38.1%）、`max_mem=3176M`。两份所获日志中均未
检出 `Traceback`、`CUDA out of memory`、`OutOfMemoryError` 或 `ERROR`；
日志副本没有 run exit code，不能据最后 iteration 推断训练已结束或此后的进度。
正式 matched PnP/K sweep 和 E40 结果在所获文件中均未生成。

## 2026-09-19 同步：B/C formal E5–E40 全部预定评估点（Observed）

来源同上一节：`E:\6D姿态估计\EXP021\B` 与 `E:\6D姿态估计\EXP021\C`。本次两臂各获得
一份更新的 `console.log`、E5–E40 全部八份 `scores_bop19_<epoch>epoch.json`。从两份
日志逐字提取全部 `EVAL_SUMMARY`（每臂 8 条），并按 epoch 顺序保存为紧凑副本；score
JSON 按原字段和值复制。仍**只记录常规 direct-pose 评估**，不执行 EXP021 专用
fixed-support matched RANSAC-PnP、K sweep、机制 gate 或 batch-1 资源 gate。

- B：[`RUN-20260914-125103-formal-s42-a01/`](evidence/RUN-20260914-125103-formal-s42-a01/)
  （`eval_summary.jsonl` 8 条 + `scores_bop19_{5,10,15,20,25,30,35,40}epoch.json` 8 份）。
- C：[`RUN-20260914-125349-formal-s42-a01/`](evidence/RUN-20260914-125349-formal-s42-a01/)
  （同上）。

### 运行状态（Observed）

- 两份 `RUN_INFO` 与 2026-09-16 所获一致：B `RUN-20260914-125103-formal-s42-a01`、
  C `RUN-20260914-125349-formal-s42-a01`，seed `42`，source
  `effc99b93eb70a9197805a427ce0156365a2bbae`，镜像
  `gdrnpp-research:torch220-cu121-sm89-c0be1ade7ea9`；日志第 27 行为
  `Using 16bit native Automatic Mixed Precision (AMP)`。
- 两臂日志最后一条 `my_writer` 均为 `epoch: 40 iter: 255919/255920[100.0%]`；
  其后各有 `FINAL_EVAL_REUSED periodic_epoch=40`。
- 两臂各有八条 `Saving checkpoint to .../checkpoints/model_epoch_0{05,10,15,20,25,30,35,40}.pth`
  记录；本次未读取服务器上的权重文件。
- 两份日志全文均未检出 `Traceback`、`CUDA out of memory`、`OutOfMemoryError`、
  `RuntimeError`；两份日志都**没有退出码行**，因此进程退出状态仍未确认。
- E40 最后一条 `my_writer`：B `time: 1.3521`、`max_mem: 1738M`、`total_loss: 2.364 (2.569)`；
  C `time: 1.3924`、`max_mem: 3180M`、`total_loss: 1.775 (2.118)`。括号内为自训练开始的
  累计平均，不与窗口中位数逐点比较。
- B 目录另有 `scores_bop19.json`，与 `scores_bop19_20epoch.json` 逐字节相同
  （`md5 001165bac95ed4fdb25ef9276bf4da5a`）。它只是 E20 点的未命名副本，**不构成
  独立评估点**，本次不另行复制或计入统计。

### 一致性核对（Observed）

八份 score JSON 的 `bop19_average_recall` 与同 epoch 日志 `EVAL_SUMMARY.bop_ar` 在
全部 16 个点（2 臂 × 8 epoch）上逐点相等。`ADD(-S)0.1d` 取日志
`EVAL_SUMMARY.add_s_0.1d`，**不是** score JSON 的 `bop19_average_recall_ad`
（后者是 BOP toolkit 的 AD 口径，两者定义与阈值不同，不可混用比较）。

### 固定评估点聚合指标（Observed）

| Epoch | 臂 | BOP AR | ADD(-S)0.1d | AR_reS | AR_teS |
|---:|---|---:|---:|---:|---:|
| 5 | B | 0.683502 | 0.525952 | 0.521799 | 0.799539 |
| 5 | C | 0.685772 | 0.498270 | 0.543483 | 0.790542 |
| 10 | B | 0.685010 | 0.526644 | 0.521107 | 0.793080 |
| 10 | C | 0.693179 | 0.560554 | 0.538639 | 0.803460 |
| 15 | B | 0.676270 | 0.489965 | 0.525952 | 0.788697 |
| 15 | C | 0.683116 | 0.491349 | 0.531488 | 0.787313 |
| 20 | B | 0.676593 | 0.511419 | 0.514879 | 0.786621 |
| 20 | C | 0.693368 | 0.547405 | 0.538639 | 0.800692 |
| 25 | B | 0.683903 | 0.506574 | 0.526413 | 0.792849 |
| 25 | C | 0.687476 | 0.532180 | 0.547866 | 0.793310 |
| 30 | B | 0.680676 | 0.500346 | 0.526182 | 0.785698 |
| 30 | C | 0.689677 | 0.523183 | 0.542099 | 0.800231 |
| 35 | B | 0.674044 | 0.476817 | 0.526413 | 0.777624 |
| 35 | C | 0.680687 | 0.485121 | 0.551326 | 0.788466 |
| 40 | B | 0.688999 | 0.525952 | 0.537024 | 0.795386 |
| 40 | C | 0.693675 | 0.534256 | 0.546021 | 0.806690 |

### 相同 epoch 的 C−B 差值（Derived）

由上表未舍入的原始值按 `C − B` 计算，单位 pt（= 相对百分点，不是相对变化率）：

| Epoch | Δ BOP AR | Δ ADD(-S)0.1d | Δ AR_reS | Δ AR_teS |
|---:|---:|---:|---:|---:|
| 5 | +0.227 | -2.768 | +2.168 | -0.900 |
| 10 | +0.817 | +3.391 | +1.753 | +1.038 |
| 15 | +0.685 | +0.138 | +0.554 | -0.138 |
| 20 | +1.678 | +3.599 | +2.376 | +1.407 |
| 25 | +0.357 | +2.561 | +2.145 | +0.046 |
| 30 | +0.900 | +2.284 | +1.592 | +1.453 |
| 35 | +0.664 | +0.830 | +2.491 | +1.084 |
| 40 | +0.468 | +0.830 | +0.900 | +1.130 |

方向统计（仅描述，不用于重选模型）：C 高于 B 的评估点为 BOP AR `8/8`、
AR_reS `8/8`、AR_teS `7/8`（E5 为负）、ADD(-S)0.1d `7/8`（E5 为负）。

### 各臂最佳点（Observed / Derived）

- B：BOP AR 与 AR_teS 的最佳点均为 E40（`0.688999` / `0.795386`）；ADD(-S)0.1d 最佳
  E10（`0.526644`）；AR_reS 最佳 E40（`0.537024`）。
- C：BOP AR、ADD(-S)0.1d、AR_reS、AR_teS 的最佳点在 E5–E40 内分别为
  E40（`0.693675`）、E10（`0.560554`）、E35（`0.551326`）、E40（`0.806690`）。
- 这些只是事后描述，**不用于重选模型**；正式比较仍按预注册的固定点进行。

### 逐物体 ADD(-S)0.1d（Observed）

对应 `EVAL_SUMMARY.add_s_obj_recalls`，键为 BOP 物体 ID。完整精度见随附 JSONL。

| Epoch | 臂 | ape(1) | can(5) | cat(6) | driller(8) | duck(9) | eggbox(10) | glue(11) | holepuncher(12) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | B | 0.440000 | 0.698492 | 0.450292 | 0.805000 | 0.416667 | 0.500000 | 0.671429 | 0.235000 |
| 5 | C | 0.497143 | 0.673367 | 0.345029 | 0.815000 | 0.316667 | 0.305556 | 0.735714 | 0.310000 |
| 10 | B | 0.474286 | 0.703518 | 0.403509 | 0.760000 | 0.450000 | 0.483333 | 0.721429 | 0.240000 |
| 10 | C | 0.468571 | 0.809045 | 0.385965 | 0.855000 | 0.516667 | 0.405556 | 0.764286 | 0.285000 |
| 15 | B | 0.314286 | 0.698492 | 0.391813 | 0.785000 | 0.255556 | 0.450000 | 0.600000 | 0.395000 |
| 15 | C | 0.468571 | 0.743719 | 0.403509 | 0.800000 | 0.233333 | 0.233333 | 0.728571 | 0.325000 |
| 20 | B | 0.457143 | 0.693467 | 0.461988 | 0.700000 | 0.355556 | 0.422222 | 0.728571 | 0.300000 |
| 20 | C | 0.451429 | 0.809045 | 0.502924 | 0.815000 | 0.488889 | 0.327778 | 0.800000 | 0.215000 |
| 25 | B | 0.468571 | 0.688442 | 0.298246 | 0.730000 | 0.305556 | 0.500000 | 0.664286 | 0.390000 |
| 25 | C | 0.514286 | 0.798995 | 0.461988 | 0.815000 | 0.416667 | 0.277778 | 0.778571 | 0.220000 |
| 30 | B | 0.405714 | 0.668342 | 0.409357 | 0.765000 | 0.250000 | 0.444444 | 0.678571 | 0.380000 |
| 30 | C | 0.440000 | 0.673367 | 0.415205 | 0.820000 | 0.183333 | 0.477778 | 0.692857 | 0.470000 |
| 35 | B | 0.331429 | 0.698492 | 0.397661 | 0.705000 | 0.172222 | 0.422222 | 0.685714 | 0.400000 |
| 35 | C | 0.377143 | 0.773869 | 0.473684 | 0.795000 | 0.150000 | 0.327778 | 0.685714 | 0.295000 |
| 40 | B | 0.485714 | 0.733668 | 0.415205 | 0.795000 | 0.333333 | 0.427778 | 0.707143 | 0.315000 |
| 40 | C | 0.491429 | 0.804020 | 0.461988 | 0.835000 | 0.250000 | 0.377778 | 0.750000 | 0.310000 |

### 未采用的外部分析（Observed）

`E:\6D姿态估计\EXP021\compare\` 另有一份 2026-09-16 生成、只覆盖 E5/E10/E15 的
B/C 对比报告与 CSV/图。它由用户侧脚本产生，**不是本仓库的正式证据**，本次不引用其
结论文字。其数值与本次提取不矛盾：同一批数据，该报告用百分数并保留两位小数
（E15 B ape 记 `31.43`），本次表用 0–1 分数记 `0.314286`。差别只在覆盖范围与
舍入方式，本记录一律以 `console.log` 直读的 `EVAL_SUMMARY` 与 score JSON 为准。

## Decision（2026-09-19）

- 用户确认 EXP021 的训练与全部 E5–E40 固定评估点已经完成，本实验的记录和结果可以
  提交本地 Git 并推送到 GitHub。
- 用户同时指出当前代码与网络结构设计存在问题，后续不能在现有设计上直接继续；
  EXP022 及之后的实验暂不修改，等待用户安排。
- 原 `FORMAL_BC_E15_AVAILABLE` 状态由本轮 E5–E40 证据取代；matched PnP/K sweep
  与资源 gate 仍未生成，作为**未生成的证据缺口**保留，不因实验收口而补做或改写。
- 本节只新增 Observed 与 Derived 事实层；不对 B/C 的机制差异作裁决，不据 8 个
  direct-pose 点重选模型，也不把常规 direct-pose 评估当作 matched correspondence
  结论。
