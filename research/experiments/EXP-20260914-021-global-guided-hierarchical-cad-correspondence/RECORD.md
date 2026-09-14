# EXP021 全局引导的层级 CAD 对应预测

- `experiment_id`: `EXP-20260914-021-global-guided-hierarchical-cad-correspondence`
- 状态：`IMPLEMENTED / LOCAL_CUDA_CPP_SMOKE_PASS / EGL_SMOKE_PENDING / FORMAL_NOT_STARTED`
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
- B/C：PBR40、batch 48、40 epoch、Ranger `lr=8e-4`、`wd=0.01`、warmup 200、
  seed 42；正式训练前用一个真实 batch 校准三项梯度尺度。
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

## Derived / Interpretation / Decision

- Derived：C 比 B 新增约 2.69M trainable 参数；真实延迟和显存必须以 CUDA profile
  为准，不能从参数量推出资源 gate。
- Interpretation：本地证据支持实现契约与梯度隔离，尚不支持任何机制效果结论。
- Decision：待服务器真实 CUDA/EGL loss 标定确认当前权重且 B/C one-step smoke
  通过后，才开始 formal；
  不用 smoke 选择 checkpoint，不因 direct-pose telemetry 改写 matched-PnP 主 gate。

## 待生成的正式证据

- loss 梯度标定 JSON、B/C CUDA/EGL smoke 输出与峰值显存。
- B/C 唯一 run_id、源码 commit、checkpoint 文件名/epoch、全部预定正式评估点。
- E40 A/B/C matched K sweep、完整 BOP evaluator 输出、gate report、batch-1 profile。
- 最终与最佳点的聚合和逐物体结果；失败 run 保留原因与有效证据边界。
