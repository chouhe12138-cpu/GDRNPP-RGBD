# EXP021 全局引导的层级 CAD 对应预测

- `experiment_id`: `EXP-20260914-021-global-guided-hierarchical-cad-correspondence`
- 状态：`LOCAL_GPU_VECTORIZATION_PASS / FORMAL_TRAINING_ACTIVE_USER_REPORTED / RUN_METADATA_PENDING`
- 日期：2026-09-14
- seed：42（训练）；20260914（CAD 表面采样）；20260730+目标序号（RANSAC）
- 实现开始时的父 commit：`c2b7c2f`；正式 run 记录实际 release commit

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

- 当前远程 formal 的唯一 run_id、source commit、进度与退出状态（用户报告训练中，
  待完成后同步）。
- 含 GPU route blocks 与 CAD-only online batch 的 L40/EGL profile（工程补充证据，
  不构成新 formal，也不阻塞训练后的下一阶段）。
- B/C checkpoint 文件名/epoch、全部预定正式评估点。
- E40 A/B/C matched K sweep、完整 BOP evaluator 输出、gate report、batch-1 profile。
- 最终与最佳点的聚合和逐物体结果；失败 run 保留原因与有效证据边界。
