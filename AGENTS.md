# 工作区操作规则

本文件只保存长期有效的 Agent 路由、安全边界和实验治理规则。动态实验结果、
GPU 占用和容器状态不在这里重复维护。

## 开始工作

- 任何修改前先运行 `git status --short --branch`，保留全部未提交内容。
- 先读根目录 `README_CN.md`，再读 `research/STATUS_CN.md` 获取当前研究状态。
- 涉及服务器、Docker、GPU、账户、数据挂载或正在运行的实验时，先读
  `research/SERVER_RUNTIME_STATUS_CN.md`，并重新执行其中的只读检查。
- 只有追溯迁移、镜像构建或 C1 历史环境时才读
  `research/SERVER_MIGRATION_HANDOFF.md`。
- 研究路线、协议和历史事实分别从 `research/RESEARCH_PLAN.md`、
  `research/stages/`、`research/experiments/` 查找，不用 HANDOFF 摘要替代原始证据。

## 目录职责

- `core/`、`lib/`、`det/`：上游及稳定核心代码；不要为整理目录而大范围重写。
- `configs/`：模型和实验配置；现有 Stage 3C 配置是 B/C2 复现链的一部分。
- `research/experiment_system/`：新实验的身份、资产、指标和产物管理基础设施。
- `research/stages/`：阶段级研究问题、冻结协议和通过门槛。
- `research/experiments/`：具体实验的客观记录、结论和紧凑 metadata。
- `output/`：Git 忽略的实际运行产物；历史目录默认只读，不覆盖、不搬迁。
- `.local/`：Git 忽略的机器特定路径配置和本地状态，不存凭据。

事实冲突时按以下顺序核对：原始产物及哈希、run manifest/标准化指标、实验
RECORD、stage 协议、STATUS/HANDOFF 摘要。

## Python 和测试环境

- 本地 Python、PyTorch、CUDA、测试或实验命令必须先激活 Conda `pytorch22`。
- 非交互 shell 先加载 Conda shell hook，再执行 `conda activate pytorch22`。
- 不要在检查该环境前诊断系统 Python、包、CUDA runtime 或 GPU 缺失。
- 深度学习服务器任务必须在项目 Docker 容器中运行，不直接用宿主机 Conda 训练。

## 实验治理

- 科学协议使用唯一 `experiment_id`；一次实际执行使用唯一 `run_id`。
- smoke、audit、formal 是 run mode，不因运行规模不同自动创建新实验。
- formal 必须分别绑定确定的 source Git commit、resolved config、初始化权重哈希、
  environment image ID 和 native/environment identity；source commit 不要求等于
  image build-source commit。
- 禁止覆盖已有 run 目录；失败或无效运行保留证据并标记状态。
- train 只负责训练，eval 只评估明确 checkpoint，diagnostic 只读固定 checkpoint，
  summarize 读取标准化指标并执行预注册 gate。
- 当前 B/C2 及其配置、Docker 脚本、容器、镜像和输出在完成封存前保持不变。
- 每个正式实验使用预注册 seed；不要把只更换 seed 的重复自动加入计划。

## Git 和服务器同步

- 本地工作区是唯一代码修改来源；Gitee `origin/main` 是唯一长期代码主线。
- 服务器只 fetch/checkout 确定 commit、准备只读 release 并运行实验，不提交、
  不 push。普通 Python/config 更新复用稳定 environment image；只有 Dockerfile、
  requirements/vendor、C++/CUDA native 或 ABI 变化才重建镜像。
- 服务器发现本地修改时先保存只读 diff 和审计信息，再回本地修复、测试和提交；
  不在服务器覆盖或 reset。
- 正式实验启动后不 pull、不修改 release checkout、不重建或替换其环境镜像。
  checkpoint、dataset、output、
  完整日志和 secrets 不进入 Git。
- 用户在服务器终端执行命令并返回输出；Agent 不主动 SSH、不生成密钥、不保存
  Gitee 口令或 PAT。

## 安全边界

- 未经用户明确要求，不 reset、clean、覆盖、删除、提交、push、批量移动或重命名。
- 不修改其他用户的账户、进程、GPU 分配、容器、镜像、目录、权限或数据。
- 使用 `/usr/bin/docker`，不要使用需要管理员密码的 `sudo docker` 别名。
- `lab0` 只使用物理 GPU 0；`lab1` 只使用物理 GPU 1。容器内通常显示为逻辑 0。
- 新研究假设和候选结构仍写入 `/mnt/e/6D姿态估计的研究`；不要自动加载该目录。
- 客观结果、复现设置和运行状态可以记录在仓库中，但大型原始产物保持外置。

详细命令见 `research/RUNBOOK_CN.md`；动态服务器事实见
`research/SERVER_RUNTIME_STATUS_CN.md`。
