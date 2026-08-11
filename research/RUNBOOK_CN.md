# 实验、Git 与服务器运行手册

本文记录稳定操作流程。即时 GPU、容器和实验状态以
`research/SERVER_RUNTIME_STATUS_CN.md` 的重新检查结果为准。

## 本地开始工作

```bash
cd /home/wsluser/GDRNPP-RGBD
git status --short --branch
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch22
```

修改代码前先确认当前实验是否仍依赖相关文件。历史 `output/`、正式 config、
checkpoint 和服务器控制脚本默认只读。

## 代码同步模型

```text
本地开发/测试 → Git commit → Gitee origin/main
              → 服务器 fetch 确定 commit
              → 新建只读 release checkout
              → 复用稳定 environment image
              → manifest 分别记录 source commit 与 environment image ID
```

- `origin/main` 是唯一长期主线；功能分支只用于短期开发。
- 服务器不提交、不 push，不通过 rsync 覆盖代码。
- dataset、weights、checkpoint、output 和完整日志不进入 Git。
- 正式实验绑定完整 40 位 commit SHA，不依赖“当时 main 大概是什么”。
- 普通 Python、config 和实验入口变化不重建镜像。只有 Dockerfile、锁定依赖、
  vendor、C++/CUDA native/ABI 变化才允许重建 environment image。

服务器获取新版本前先执行：

```bash
git status --porcelain=v1 --untracked-files=all
git rev-parse HEAD
```

若有输出，停止同步并保存 diff/状态供本地审阅。干净时才执行：

```bash
git fetch origin
git switch --detach <40位commit>
git rev-parse HEAD
```

不得在正在运行旧实验的代码目录中 pull 或切换版本。EXP005/EXP009 使用
`releases/GDRNPP-RGBD-<short-commit>` 全新 clone，并 checkout 指定 commit；
旧 dirty repo 和运行目录保持不动。

## 正式实验冻结条件

- Git tracked 和非忽略 untracked 状态为空；
- release 为 detached HEAD，正式启动后保持只读；
- commit 已存在于 Gitee；
- resolved config 和初始化 checkpoint 哈希已记录；
- environment image ID、build-source commit、环境契约和 native artifact 哈希已记录；
- source commit 与 image build-source commit 独立，不要求相等；
- 数据、VOC、baseline 和 renderer 检查通过；
- smoke/audit gate 满足对应协议；
- 输出目录不存在；
- 已记录启动时 GPU 占用，且没有同容器/同角色的重复 formal；
- manifest 已记录 experiment ID、run ID、seed 和环境。

正式启动后不 pull、重建镜像、替换容器或修改 config。失败运行不删除，标记
`FAILED`；违反协议或误启动的运行标记 `INVALID`。

## Train / Eval / Diagnostic 职责

- `train`：训练并写 loss、事件和 checkpoint，不决定研究结论。
- `eval`：评估显式 checkpoint，保留 BOP raw score 并生成结果索引。
- `diagnose`：只读固定 checkpoint，不修改模型参数。
- `summarize`：读取标准化指标并执行预注册 gate。
- `verify`：校验 manifest、哈希、参数隔离和产物完整性。

C2 已完成 Epoch 40 并按历史最小证据归档；新的 EXP005/EXP009 run 使用受管
入口，不复用已删除的旧 C2 容器或输出目录。

## EXP005 / EXP009 受管入口

新 run 的随机种子统一固定为 `42`。run ID 的 UTC 时间只负责目录唯一性，不
参与随机数初始化。历史 run 保留原 seed。

每个新 release 在创建容器前先复用已构建镜像并准备 native overlay：

```bash
docker/l40/prepare_release.sh \
  lab0 \
  gdrnpp-research:torch220-cu121-sm89-35313ae3d413
```

该命令不构建镜像。它只解析不可变 image ID、验证环境契约，并把镜像中的
`.so`/uncertainty-PnP runtime libraries 按哈希复制到 Git 忽略区域。环境或
native 输入不兼容时直接拒绝运行。

宿主机用 Git 检查 detached/clean release，并生成 binding v2 tracked-source
snapshot；容器只验证该 snapshot、image ID 和 native artifacts，不要求镜像内
安装 Git。普通 Python/config commit 不触发镜像重建。

```bash
docker/l40/managed_experiment.sh lab0 EXP005 access
docker/l40/managed_experiment.sh lab0 EXP005 create
docker/l40/managed_experiment.sh lab0 EXP005 gate
docker/l40/managed_experiment.sh lab0 EXP005 smoke
docker/l40/managed_experiment.sh lab0 EXP005 audit48
docker/l40/managed_experiment.sh lab0 EXP005 launch
```

`smoke` 和 `audit48` 是后台 run，前一步显示 `COMPLETE` 后才能启动下一步：

```bash
docker/l40/managed_experiment.sh lab0 EXP005 status
```

EXP009 的 C2 完成、结果核验和旧 `lab1_chx` 清理门槛已经满足：

```bash
docker/l40/managed_experiment.sh lab1 EXP009 access
docker/l40/managed_experiment.sh lab1 EXP009 create
docker/l40/managed_experiment.sh lab1 EXP009 gate
docker/l40/managed_experiment.sh lab1 EXP009 smoke
docker/l40/managed_experiment.sh lab1 EXP009 audit48
docker/l40/managed_experiment.sh lab1 EXP009 launch
```

`launch` 每 15 分钟检查 formal 状态；只有明确 CUDA OOM 才在保留失败 attempt
后等待 10 分钟重试。其他错误停止重试。GPU 上存在其他进程只记录快照，不
自动杀进程，也不单凭“非空闲”取消实验。

成功训练后执行：

```bash
docker/l40/managed_experiment.sh <lab0|lab1> <EXP005|EXP009> finalize
```

正式日志只保存 setup、每 500 iteration/epoch 的关键 loss、checkpoint、评估、
warning 和异常。完整配置、环境、指标和 warning 计数分别保存为结构化文件；
不保存 TensorBoard、完整模型文本或 DEBUG 级冗余信息。

## 服务器安全

- 使用 `/usr/bin/docker`，不使用 `sudo docker` 别名。
- `lab0` 只使用物理 GPU 0；`lab1` 只使用物理 GPU 1。
- 不 kill 其他用户进程，不修改其他用户容器、镜像、目录或权限。
- Agent 不主动 SSH。用户在服务器执行命令并返回输出。
- Gitee HTTPS 凭据交互输入，不写入仓库、脚本或 manifest。

## 失败与恢复

- 不用 `git reset --hard` 或 clean 恢复整理失败。
- 每类修改独立提交；需要回退时切换已知 commit 或显式 revert。
- 旧 Docker image/container 保留到新 smoke 完整通过。
- resume 必须读取原 run manifest；科学字段变化时创建新 run。
- 服务器临时修复先保存 diff，再回本地重现、测试和提交。普通源码更新建立新
  release 并复用 environment image；只有环境契约变化才重建镜像。
