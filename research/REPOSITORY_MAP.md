# 仓库地图

| 路径 | 作用 |
|---|---|
| `core/gdrn_modeling/models/GDRN_CAD.py` | EXP025 模型包装、初始化和 checkpoint 契约 |
| `core/gdrn_modeling/models/heads/` | EXP025 attention/head；历史稳定实现仍保留但不由 EXP025 导入 |
| `core/gdrn_modeling/cad/` | 通用 CAD hierarchy loader |
| `configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/` | 两条正式 arm 与本地 smoke 配置 |
| `research/exp025/` | preflight、server gate 所用真实 batch smoke、诊断和测试 |
| `research/cad_hierarchy/` | hierarchy 几何与完整性检查 |
| `research/cad_hierarchy/build_geometry_adaptive.py`、`compare_geometry_adaptive.py` | EXP026 三层 GA-HFPS 独立构造与离线比较 |
| `research/experiments/` | EXP000–026 的唯一叙述性 RECORD 与紧凑证据 |
| `docker/l40/experiment.sh` | 当前唯一服务器 check/create/gate/run/eval/status/logs 入口 |

历史实验配置、runner 与诊断已退出 HEAD；按 RECORD 的 source commit 使用独立 worktree
复现。`output/`、`.local/`、数据和权重均为机器本地内容。
