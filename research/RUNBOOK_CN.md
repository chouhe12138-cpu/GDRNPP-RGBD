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
              → 干净源码构建 Docker image
              → manifest 记录 commit + image ID
```

- `origin/main` 是唯一长期主线；功能分支只用于短期开发。
- 服务器不提交、不 push，不通过 rsync 覆盖代码。
- dataset、weights、checkpoint、output 和完整日志不进入 Git。
- 正式实验绑定完整 40 位 commit SHA，不依赖“当时 main 大概是什么”。

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

不得在正在运行旧实验的代码目录中 pull 或切换版本。需要临时并存时使用独立
`git worktree`，旧实验封存后再由用户确认是否移除。

## 正式实验冻结条件

- Git tracked 和非忽略 untracked 状态为空；
- commit 已存在于 Gitee；
- resolved config 和初始化 checkpoint 哈希已记录；
- Docker image revision 等于 Git commit；
- 数据、VOC、baseline 和 renderer 检查通过；
- smoke/audit gate 满足对应协议；
- 输出目录不存在；
- GPU 空闲且没有同角色重复 formal；
- manifest 已记录 experiment ID、run ID、seed 和环境。

正式启动后不 pull、重建镜像、替换容器或修改 config。失败运行不删除，标记
`FAILED`；违反协议或误启动的运行标记 `INVALID`。

## Train / Eval / Diagnostic 职责

- `train`：训练并写 loss、事件和 checkpoint，不决定研究结论。
- `eval`：评估显式 checkpoint，保留 BOP raw score 并生成结果索引。
- `diagnose`：只读固定 checkpoint，不修改模型参数。
- `summarize`：读取标准化指标并执行预注册 gate。
- `verify`：校验 manifest、哈希、参数隔离和产物完整性。

当前 B/C2 继续使用已经冻结的 `docker/l40/lab0_b.sh`、
`docker/l40/lab1_c2.sh` 和 `research/stage3c_runtime/`；新统一入口不接管未完成的
正式实验。

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
- 服务器临时修复先保存 diff，再回本地重现、测试、提交和重新构建镜像。
