# GDRNPP-RGBD 中文项目入口

本仓库以 GDRNPP/ConvNeXt-Base 为基线，研究遮挡场景下的 6D 位姿估计与
Patch-PnP 信息利用。当前主线是 RGB-only 的几何、聚合和姿态头机制研究；
早期 Camera-XYZ RGB-D 融合方案处于暂缓状态，不是当前正式方法。

## 从这里开始

1. 查看 [当前研究状态](research/STATUS_CN.md)，了解已提交并有证据支持的阶段结论。
2. 查看 [实验索引](research/EXPERIMENT_INDEX.md)，再进入对应实验的 `RECORD.md` 和 `EXPERIMENT.json`。
3. 需要运行、同步或恢复实验时查看 [实验与服务器运行手册](research/RUNBOOK_CN.md)。
4. 需要服务器动态事实时查看
   [服务器运行状态](research/SERVER_RUNTIME_STATUS_CN.md)，并重新执行其中的只读检查。
5. 需要理解研究依据时查看 [研究计划](research/RESEARCH_PLAN.md)、
   [决策记录](research/DECISIONS.md) 和 `research/stages/` 中的阶段协议。
6. 历史重新验收情况见 [历史实验验收总表](research/HISTORICAL_ACCEPTANCE_CN.md)。

## 目录地图

| 路径 | 职责 |
|---|---|
| `core/`, `lib/`, `det/` | 上游和稳定核心代码 |
| `configs/` | 模型、训练与评估配置 |
| `research/stages/` | 阶段问题、冻结协议和 gate |
| `research/experiments/` | 具体实验记录和紧凑 metadata |
| `research/EXPERIMENT_INDEX.md` | 全部实验的统一索引 |
| `research/experiment_system/` | 新实验身份、资产、指标与产物基础设施 |
| `research/managed_runtime/` | 受管训练、封存与精简日志执行器 |
| `docker/l40/` | L40 可复现环境与服务器控制脚本 |
| `output/` | Git 忽略的本地原始产物，历史目录默认只读 |
| `.local/` | Git 忽略的机器路径 profile，不得保存凭据 |

## 状态和证据的关系

- `research/STATUS_CN.md`：当前状态和下一步的中文摘要。
- `research/EXPERIMENT_INDEX.md`：全部实验的统一索引。
- `research/HISTORICAL_ACCEPTANCE_CN.md`：EXP000～008 重新验证状态。
- `research/RESEARCH_PLAN.md`：长期研究路线。
- `research/DECISIONS.md`：协议变更及原因。
- `research/stages/`：阶段级冻结协议。
- `research/experiments/<experiment_id>/`：具体实验的客观记录与结论。
- `research/SERVER_RUNTIME_STATUS_CN.md`：带日期的服务器动态事实。

若记录冲突，以原始产物及哈希为先，其次是 run manifest/标准化指标、实验
RECORD、stage 协议，最后才是状态摘要。

README 不保存“当前实验编号”、Git HEAD/worktree 或服务器实时状态；这些动态事实
分别由实验索引、Git 本身和服务器状态文档承担。

## 基本安全规则

- 修改前检查 Git 状态并保留未提交内容。
- 本地 Python 命令使用 Conda `pytorch22`；服务器训练使用项目 Docker。
- 不覆盖历史 output，不把 dataset、checkpoint、完整日志或 secrets 提交到 Git。
- 当前正式实验完成前不改变其 config、镜像、容器或运行脚本。
- `research/experiments/` 保存身份、协议和结果记录；公共训练代码仍在 `core/`，
  实验差异由 `configs/` 和统一启动器表达，不为每个 EXP 复制训练循环。

上游英文用法仍见 [README.md](README.md)。Agent 的长期工作规则见
[AGENTS.md](AGENTS.md)。
