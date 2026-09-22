# 仓库地图

| 路径 | 作用 |
|---|---|
| `core/gdrn_modeling/models/GDRN_CAD.py` | EXP025/026 共享模型包装、初始化和 checkpoint 契约 |
| `core/gdrn_modeling/models/heads/` | EXP025/026 共享 attention/head；训练 residual mode 由配置选定 |
| `core/gdrn_modeling/cad/` | 通用 CAD hierarchy loader |
| `configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/` | 两条正式 arm 与本地 smoke 配置 |
| `research/exp025/` | preflight、server gate 所用真实 batch smoke、诊断和测试 |
| `research/exp026/` | 新两臂 artifact/CPU preflight、本地 CUDA 与 fixed-batch 验证 |
| `configs/gdrn/lmo_pbr/research/exp026_residual_aligned_sampling_ablation/` | EXP026 两臂 matched Full 配置；服务器放行开关保持关闭 |
| `research/exp025_checkpoint_diagnostics/` | EXP025 E15 完整权重审计、GT-box oracle 与推理干预（不改正式训练） |
| `research/cad_hierarchy/` | hierarchy 几何、完整性及实验 artifact 精确身份检查 |
| `core/gdrn_modeling/datasets/research_context.py` | CAD/PCC 共用 dataset metadata、object order 与 artifact 路径解析 |
| `configs/gdrn/lm/research/candidate_cad/`、`research/lm_candidate/` | LM13 非正式候选配置和本地检查；不属于 EXP025/026 formal |
| `research/cad_hierarchy/build_geometry_adaptive.py`、`compare_geometry_adaptive.py` | EXP026 三层 GA-HFPS 独立构造与离线比较 |
| `research/experiments/` | EXP000–026 的叙述性 RECORD、母实验内部诊断与紧凑证据 |
| `docker/l40/experiment.sh` | 当前唯一服务器 check/create/gate/run/eval/status/logs 入口 |

历史实验配置、runner 与诊断已退出 HEAD；按 RECORD 的 source commit 使用独立 worktree
复现。`output/`、`.local/`、数据和权重均为机器本地内容。
