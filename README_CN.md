# GDRNPP-RGBD 中文入口

本仓库以 GDRNPP/ConvNeXt-Base 为基线，研究遮挡场景中 dense correspondence
到直接姿态头的信息利用。当前代码主线是 EXP012、EXP013 与 EXP017；早期诊断和失败方案
只保留实验记录，不再保留专用执行框架。

## 阅读顺序

1. `research/STATUS_CN.md`：当前结论和下一步。
2. `research/EXPERIMENT_INDEX.md`：全部实验状态及 RECORD 链接。
3. `research/RESEARCH_PLAN.md`：当前研究问题和后续候选。
4. `research/RUNBOOK_CN.md`：本地测试、服务器训练与评估命令。
5. `research/SERVER_SAFETY_CN.md`：服务器只读检查和隔离规则。

## 目录

| 路径 | 内容 |
|---|---|
| `core/`, `lib/`, `det/` | 上游 GDRNPP 与稳定核心 |
| `configs/` | 上游配置及当前研究配置 |
| `research/experiments/` | 每个实验唯一的 `RECORD.md` |
| `research/next_pose_head/` | EXP012 实现与 preflight |
| `research/exp013/`, `research/exp014/`, `research/exp017/` | 当前 pose-head 分枝与 preflight |
| `research/diagnostics/pose_structure/` | 当前低成本结构诊断 |
| `docker/l40/` | L40 镜像和单一安全启动器 |
| `output/`, `.local/` | Git 忽略的本机产物与路径资源 |

实验事实以 RECORD 为准；索引和 STATUS 只做导航与当前摘要。上游英文说明见
`README.md`，长期 Agent 规则见 `AGENTS.md`。
