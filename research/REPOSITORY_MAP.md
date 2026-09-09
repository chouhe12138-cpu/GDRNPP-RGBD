# 仓库地图

本页说明代码归属、入口及适用边界，不维护实验结果。当前进展见
[STATUS](STATUS_CN.md)，实验事实见 [索引所链接的 RECORD](EXPERIMENT_INDEX.md)，
操作流程见 [RUNBOOK](RUNBOOK_CN.md)，长期规则见 [AGENTS](../AGENTS.md)。

## 代码与配置归属

| 类别 | 位置 | 职责与边界 |
|---|---|---|
| 上游与稳定代码 | `core/`、`lib/`、`det/` | 模型、数据、训练/评估、几何、renderer/native 和检测器；core 并非纯上游快照 |
| 已集成研究扩展 | `core/gdrn_modeling/models/heads/` | hierarchical、EXP013、GLM、EXP017/B、GCR、official random 等 head；保留原 registry 和 checkpoint 路径 |
| EXP020 几何重投影监督 | `core/gdrn_modeling/losses/correspondence_reprojection_loss.py` | EXP020 纯 Tensor loss；经 `GDRN_double_mask.gdrn_loss`/`engine` 透传在线 `roi_zoom_K`；不新增模型参数。stats 区分 loss 值与真实 `mean_reproj_px`；`valid_ratio` 以 GT foreground 为分母；`REPROJ_LW>0 && USE_MTL=True` fail-fast |
| 研究集成点 | `core/gdrn_modeling/models/`、`datasets/`、`engine/` | 模型工厂和 GDRN_double_mask 接线；F 深度统计、GCR image_hw/pose/loss 传递；早期 CPM/quality 接线仍存在 |
| 实验专用验证 | `research/next_pose_head/`、`exp013/`、`exp014/`、`exp017/`、`exp017b/`、`exp018/`、`exp019/`、`exp020/` | preflight、测试和说明；next_pose_head 对应 EXP012；真实 smoke 入口在 EXP017/018；EXP020 含 preflight、real_smoke、`matched_pnp_eval.py`（主下游，跨 checkpoint fixed support）与 `calibrate_reproj_weight.py`（REPROJ_LW 梯度尺度标定） |
| 公共研究工程检查 | `research/run_contract.py`、`research/tests/` | 当前 smoke/formal/eval 配置约束、launcher/cache 等回归；训练协议约束不适用于所有诊断 |
| 可复用结构诊断 | `research/diagnostics/pose_structure/` | D1–D6、指标、模型访问和报告；支持范围受 head/decode 契约限制 |
| 专用诊断 | `research/diagnostics/exp019_epro/`、`pose_structure/f_glm_diagnostic.py` | EXP019 固定历史协议；F 专用干预。EXP019 使用说明在 `research/exp019/README.md` |
| 通用工具 | `tools/` 及 core/lib/det 内部 tools/scripts | checkpoint 转换、BOP 合并/时间处理、上游数据准备和可视化；不是统一测试套件 |
| 环境与运行脚本 | `scripts/`、`docker/l40/`、core/det 的 shell 入口 | 环境构建、诊断、受管服务器运行、bundle，以及上游 train/test；不同契约不混用 |
| shared/base 配置 | `configs/gdrn/lmo_pbr/research/_base_/`、`controls/`、`templates/` | 公共评估/训练协议、长期 control、新实验配置模板 |
| 实验配置 | 同配置根目录下 EXP012、EXP013、EXP017、EXP018 | train/eval/smoke，以及部分实验的 audit48；并非每个实验都具备四种文件。EXP020 在 `exp020_geometry_aware_corr/`，A/B 用 `control.py`/`reproj.py`（唯一差异 `REPROJ_LW`） |
| 诊断配置 | `configs/gdrn/lmo_pbr/research/exp019/epro_diagnostic.py`、`research/diagnostics/exp019_epro/config.py` | 前者是声明文件；实际 runner 使用后者的 ExperimentConfig，模型配置另由 CLI 传入 |

研究验证和诊断调用 core，core 使用 lib 的几何、renderer 与 evaluator。det 也使用
core/lib 公共工具，core 的 demo 依赖 det；GT-box 研究未经过检测器不表示 det 可删除。
多个 preflight 复用 EXP013 helper，EXP017B 测试还导入 EXP017 测试 helper；这些是
现有依赖，并未在本次导航整理中抽取。相似的 XYZ、support、PnP 或 decode helper
可能使用不同 dtype、采样与历史协议，不能仅按名称合并。

## 测试与工程验证

测试保留在原实验或模块旁；以下是用途分类，不要求搬成四套目录。

| 类型 | 现有位置或入口 | 能证明什么 |
|---|---|---|
| Unit | 各实验 `tests/`、`pose_structure/tests/`、`exp019_epro/tests/` | 张量、坐标、support、metric、gate 等局部行为 |
| Integration | EXP013 integration、EXP018 模型/mapper/mock-loader 测试、公共配置与 launcher 测试 | 模块接线与契约；同一测试文件可混合 unit 和 integration |
| Regression | checkpoint round-trip、zero-init 等价、detach、历史 Rodrigues、renderer/cache/launcher 测试 | 已知行为和修复边界；不是额外一套实验 |
| CPU preflight | 各实验 `preflight.py` | 模型构建、冻结/加载/梯度等工程检查；按入口可能需要本地权重，不是性能结果 |
| 真实数据 smoke | EXP017/018 `real_smoke.py`、各实验 `smoke.py`、EXP019 32-target 模式 | 受限数据下执行链；本地数步 smoke 与服务器一 epoch smoke 不等价 |
| Profiling / audit | EXP013 `profile_head()`（D/017 复用）、部分配置 `audit48.py` | head-only FLOPs/latency 与 batch48 工程检查；不替代正式精度或完整训练资源测量 |

本地先激活 Conda `pytorch22`。研究测试可从仓库根目录通过
`PYTHONPATH="$PWD" pytest -q research` 覆盖现有测试目录；仅 `--collect-only` 成功不表示测试通过。
裸跑仓库根目录 pytest 还可能收集上游/第三方测试，不等于上述研究测试范围。
当前 `research/tests/test_active_configs.py` 的目录列表未包含 EXP018/019，它们另有
局部测试；不能将该列表视为全部现有配置。

## 主要入口与已知边界

- [EXP018 说明](exp018/README.md)：GCR 实现契约及手动 smoke；当前状态回 RECORD。
- [EXP019 说明](exp019/README.md)：对应 `scripts/run_exp019_epro.sh` 和
  `research.diagnostics.exp019_epro.{preflight,runner,evaluate}`。脚本含本机路径及
  full 模式的研究分支检查；运行 metadata 中的实际配置不能被当前 CLI 默认值替代。
- `tools/run_pose_structure_diagnostics.py`：D1–D6。旧路径捕获 pnp_net 后重新 decode，
  不能直接代表 EXP018 的 post-correction final pose；不应未经适配套用于该模型。
- `tools/run_exp014_f_diagnostics.py`：名称含 EXP014，实际是 EXP013F 专用诊断，
  不是暂停的 EXP014-D。其历史报告面向 EXP017 设计，不代表当前任务安排。
- `pose_structure/reporting.py` 当前允许已有输出目录并写固定文件名；运行前应遵守
  唯一目录规则。本次未修复代码。EXP019 evaluator 则要求新建 bop_results 目录，
  不能假设已有部分评估产物时直接重试必定成功，也不应为重试删除原始证据。
- `docker/l40/experiment.sh` 是当前服务器 check/create/run/eval/status/logs 入口；
  操作前遵守 [服务器安全规则](SERVER_SAFETY_CN.md)。`create_bundle.sh` 当前硬编码
  main，与当前研究分支发布需求不一致；这是一项未修复限制，不能描述为已支持。
- `scripts/install_deps.sh`、`compile_all.sh` 是上游环境安装/编译脚本，包含包修改和
  构建清理；当前 Docker 流程不能按宿主机安装说明直接照搬。
- `tools/process_bop_results_time.py` 会备份后原地改写 CSV；其他转换工具也会写出
  文件。工具名称不意味着只读，不应用它们改写已留存的原始科学证据。

`output/` 与 `.local/` 是 ignored 本机内容，不保证随 checkout 可用。EXP019 的四份
小型原始 JSON 随其 RECORD 保存，完整 poses、BOP 明细和日志仍外置。历史退出入口
按 RECORD 的 source commit 在 Git 历史中查找，不为复用旧命令恢复旧执行框架。
