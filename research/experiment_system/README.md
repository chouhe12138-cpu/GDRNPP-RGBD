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
python -m research.experiment_system.cli accept-history
```

`accept-history` 默认只读检查 EXP000～EXP008 的现有证据。只有在工作树干净且
确认 dry-run 无 `CONFLICT` 后才使用 `--write`；写入时为每个实验生成独立
`ACCEPTANCE.json`/`ACCEPTANCE_CN.md`，不修改原 `RECORD.md` 和历史 output。

机器路径 profile 从 `path_profile.example.json` 复制到 Git 忽略的
`.local/path_profiles/<machine>.json`，不得在其中保存凭据。

## 新 run 准备

服务器先从稳定 environment image 为 detached release 准备 native overlay：

```bash
docker/l40/prepare_release.sh lab0 <existing-environment-image>
```

```bash
python -m research.experiment_system.cli prepare \
  --experiment research/experiments/EXP-.../EXPERIMENT.json \
  --config configs/gdrn/lmo_pbr/research/<experiment>/train.py \
  --mode smoke \
  --seed 42 \
  --profile .local/path_profiles/<machine>.json \
  --environment-binding .local/environment_binding.json \
  --environment-image-id sha256:<actual-container-image-id> \
  --output-root output/experiments
```

新 manifest 分别记录 `source_git_commit`、resolved config hash、不可变
environment image ID、environment build-source commit、环境契约和 native
artifact identity；不要求 image build-source commit 等于 source commit。

formal 模式要求 detached release checkout 和包括非忽略 untracked 文件在内的
Git 工作树完全干净。当前
C2 继续使用既有冻结运行链；新的 EXP005/EXP009 使用
`docker/l40/managed_experiment.sh`。两者的新 run 均固定 seed `42`。

长时间训练仍由已有服务器脚本后台执行；统一 CLI 只登记步骤命令和状态，不
持有 SSH 或训练进程：

```bash
python -m research.experiment_system.cli step <run_dir> \
  --step-id train --kind train --command-line "<实际训练命令>"
python -m research.experiment_system.cli step <run_dir> \
  --step-id train --set RUNNING --message "后台任务已启动"
```

受管 run 还会生成 `train/epoch_summary.jsonl`、checkpoint index、evaluation
index 和去重后的 `summary/warnings.json`。控制台日志不是指标的唯一来源。
