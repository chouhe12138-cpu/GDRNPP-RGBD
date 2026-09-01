# 工作区操作规则

本文件只保存长期有效的工作方式和安全边界。实验结果放在各实验的
`RECORD.md`，当前进展放在 `research/STATUS_CN.md`。

## 开始工作

- 修改前运行 `git status --short --branch`，保留用户已有修改。
- 先读 `README_CN.md` 和 `research/STATUS_CN.md`；需要历史结果时再查
  `research/EXPERIMENT_INDEX.md` 与对应 `RECORD.md`。
- 涉及服务器、Docker 或 GPU 时先读 `research/SERVER_SAFETY_CN.md`，并由用户在
  服务器执行其中的只读检查。Agent 不主动 SSH。

## 代码与环境

- `core/`、`lib/`、`det/` 是上游和稳定核心，不因整理目录而大范围改写。
- 当前研究代码集中在 EXP012、EXP013/014 和 `research/diagnostics/pose_structure/`；
  实验差异优先由 `configs/gdrn/lmo_pbr/research/` 表达。
- 本地 Python、测试和实验命令先激活 Conda `pytorch22`；服务器深度学习任务只在
  项目 Docker 容器内运行，不修改宿主机 Python、CUDA 或全局包。
- dataset、checkpoint、完整日志、缓存和 secrets 不进入 Git；`output/` 与
  `.local/` 是机器本地内容。

## 轻量实验记录

- 一个科学问题使用唯一 `experiment_id`，一次执行使用唯一 `run_id`。
- 每个实验目录只维护一份 `RECORD.md`；中央状态只维护在
  `research/EXPERIMENT_INDEX.md`。
- 正式记录保留：experiment/run ID、配置路径、seed、源码 commit、checkpoint
  文件名与 epoch、关键指标、结论或失败原因。
- 不要求源码快照、checkpoint、日志、镜像或原生扩展哈希；怀疑文件损坏时可以
  临时校验，但不建立常规哈希链。
- 运行目录必须唯一，禁止覆盖。失败运行在 RECORD 中保留原因；大型失败产物可在
  结论收口后清理。
- train、eval、diagnostic 使用明确的配置和 checkpoint，不把 smoke 当正式结果。

## Git 与服务器

- 本地工作区是唯一代码修改来源；服务器只 checkout 确定 commit 并运行，不提交、
  不 push、不现场修代码。
- 普通 Python/config 变更复用稳定镜像；只有依赖、Dockerfile、C++/CUDA 或 ABI
  变化才重建镜像。
- 正式训练期间不 pull、不修改 checkout、不替换镜像。
- 未经用户明确要求，不 reset、clean、删除未列明数据、提交、合并或 push。

## 服务器安全边界

- 使用 `/usr/bin/docker`，不使用 `sudo docker`；不执行 Docker prune。
- `lab0` 只用物理 GPU 0，`lab1` 只用物理 GPU 1；启动前检查 GPU 占用。
- repo、dataset 和 weights 只读挂载，只有项目 output/cache/home 可写。
- 只操作带本项目 ownership label 的容器；不停止、重命名、删除或进入其他用户的
  容器，不修改其他账户、进程、GPU 分配、镜像、权限或数据。
- 新实验使用 `docker/l40/experiment.sh`。该脚本不提供 stop/remove/prune；需要
  破坏性服务器操作时必须另行确认精确目标。

详细运行命令见 `research/RUNBOOK_CN.md`。
