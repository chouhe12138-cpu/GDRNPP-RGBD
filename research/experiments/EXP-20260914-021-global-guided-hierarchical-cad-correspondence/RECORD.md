# EXP021 全局引导的层级 CAD 对应预测

- `experiment_id`: `EXP-20260914-021-global-guided-hierarchical-cad-correspondence`
- 状态：`AMP_FEATURE_ONLY_LOCAL_PASS / SERVER_EGL_REVALIDATION_PENDING / FP32_FORMAL_ACTIVE_PENDING_RESTART`
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

## Derived / Interpretation / Decision

- Derived：C 比 B 新增约 2.69M trainable 参数；真实延迟和显存必须以 CUDA profile
  为准，不能从参数量推出资源 gate。
- Interpretation：本地证据支持实现契约与梯度隔离，尚不支持任何机制效果结论。
- Interpretation：服务器 w2/w8 对比表明总耗时首先受 DataLoader 并发影响；近期
  batch-48 formal/audit 均使用 16 workers，EXP021 先前继承公共基线的 8 属配置遗漏。
- Interpretation：AMP 主要缩短 tensor-core 友好的前向；CAD 标签、排序、索引和
  FP32 `cdist` 限制 B/C 的端到端收益，C 的本机 DataLoader 长尾进一步稀释总收益。
- Decision：原空闲 GPU `≤1.2/1.5 s` 启动 gate 及当前实际偏离继续作为 Observed
  保留。2026-09-14 用户接受训练耗时，并决定以同代码 FP32→AMP 相对提速和数值完整性
  做本轮工程 review；新 release 必须先在服务器完成 EGL 标定、B/C smoke 和 matched
  profile。通过后终止待替换 FP32 runs，正式配置固定 AMP + 16 workers，按 B→lab0、
  C→lab1 从官方 checkpoint 重启；不用 smoke 选择 checkpoint，不因 direct-pose
  telemetry 改写 matched-PnP 主 gate。

## 待生成的正式证据

- loss 梯度标定 JSON、B/C CUDA/EGL smoke 输出与峰值显存。
- B/C 唯一 run_id、源码 commit、checkpoint 文件名/epoch、全部预定正式评估点。
- E40 A/B/C matched K sweep、完整 BOP evaluator 输出、gate report、batch-1 profile。
- 最终与最佳点的聚合和逐物体结果；失败 run 保留原因与有效证据边界。
