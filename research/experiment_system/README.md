# 实验管理基础设施

本目录为后续新实验提供 opt-in 的身份、路径、指标和产物管理，不替换正在进行
的 B/C2 脚本，也不修改上游 GDRNPP 训练接口。

## 主要契约

- `EXPERIMENT.json`：受 Git 管理的科学实验身份和冻结协议。
- `meta/run_manifest.json`：实际运行时自动生成、不可覆盖的执行事实。
- `meta/run_state.json`：`PREPARED/RUNNING/COMPLETE/FAILED/INVALID/ARCHIVED`。
- `checkpoints/checkpoint_index.json`：checkpoint 与 experiment/run/config/commit 的映射。
- `evaluation_index.json`：BOP raw score 的相对路径和 SHA-256。
- `metrics.normalized.json`：使用明确 macro/micro 语义的统一指标。

## 只读检查

在 Conda `pytorch22` 中运行：

```bash
python -m research.experiment_system.cli registry --check
python -m research.experiment_system.cli metrics
python -m research.experiment_system.cli source
python -m research.experiment_system.cli audit
python -m research.experiment_system.cli verify-freeze
```

机器路径 profile 从 `path_profile.example.json` 复制到 Git 忽略的
`.local/path_profiles/<machine>.json`，不得在其中保存凭据。

## 新 run 准备

```bash
python -m research.experiment_system.cli prepare \
  --experiment research/experiments/EXP-.../EXPERIMENT.json \
  --config configs/gdrn/lmo_pbr/research/<experiment>/train.py \
  --mode smoke \
  --seed 20260731 \
  --profile .local/path_profiles/<machine>.json \
  --image gdrnpp:<tag> \
  --output-root output/experiments
```

formal 模式要求包括非忽略 untracked 文件在内的 Git 工作树完全干净。当前
B/C2 继续使用既有 `research/stage3c_runtime/` 和 Docker 控制脚本。

长时间训练仍由已有服务器脚本后台执行；统一 CLI 只登记步骤命令和状态，不
持有 SSH 或训练进程：

```bash
python -m research.experiment_system.cli step <run_dir> \
  --step-id train --kind train --command-line "<实际训练命令>"
python -m research.experiment_system.cli step <run_dir> \
  --step-id train --set RUNNING --message "后台任务已启动"
```
