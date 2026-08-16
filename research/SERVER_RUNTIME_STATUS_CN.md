# GDRNPP 服务器运行环境与实验进度交接

最后更新：2026-08-16

## 2026-08-16 更新

- EXP005/B formal run `RUN-20260811-063606-formal-s42-a01` 已自然完成固定
  Epoch 40：BOP AR `0.6919123414`、ADD(-S) `0.5065743945`。
- 外部 Windows checkpoint 副本仍在下载；完成后再与服务器原文件核对哈希，
  下载中的临时文件不解释为权重损坏。
- EXP010 已授权在 lab0 使用新的确定 Git release 启动。它与 lab1 的 EXP009
  可以并行；最终比较等待两者固定 Epoch 40。
- 启动 EXP010 前只替换项目自己的 `lab0_chx` 容器；EXP005 output、旧 release、
  数据、权重、稳定 environment image 和其他用户资源全部保留。

## 2026-08-11 当前状态

- C2 已自然完成 Epoch 40，固定 Epoch 40 BOP AR 为 `0.6930057670`；筛选结论
  为 `C2_SCREEN_FAIL`。Epoch 40 checkpoint、完整日志和 BOP JSON 已在本地与
  服务器完成 SHA-256 对照，历史 ADD(-S) 保持缺失。
- 当前正式运行使用同一个 detached、只读 source release：
  `652d7fd9d38f8ea5cea0c5a98cc9477b66623180`。`lab0_chx` 在物理 GPU 0
  运行 EXP005/B，`lab1_chx` 在物理 GPU 1 运行 EXP009/CPM。
- 两端 source snapshot、native、环境、registry、数据与角色 gate 均通过；有效
  smoke 与 audit 也已完成。用户已确认两个 formal 均在运行中，但尚未回传精确
  formal `run_id`、后台 PID 和当前 epoch。本文不伪造这些身份，训练结束后必须
  从 `latest_formal_run.path`、run manifest 和 run state 补齐。
- 两端第一次 managed smoke 均为无效基础设施运行：EXP005 run
  `RUN-20260811-052852-smoke-s42-a01`、EXP009 run
  `RUN-20260811-052906-smoke-s42-a01`。dataset cache 错误指向只读 source
  release，训练异常又被 Loguru decorator 吞掉，因而未生成 checkpoint，最后
  表面记录为 `POSTPROCESS_ERROR`。这不是 EXP005 或 CPM 的科学失败。
- 两个失败 run 必须保留。修复 commit `dcf6d57e694229f2e723f3389a171d5cf603dcfe`
  的第一次容器重建在实验进程启动前失败：Docker 无法在只读 source mount
  内自动创建原本不存在的 `/workspace/gdrnpp/.cache` 嵌套挂载点。未创建新的
  run，也不构成实验失败。最终 `652d7fd...` 修复在 `docker run` 前创建 Git
  忽略 mountpoint，已通过本地 `147 passed` 和两端实际容器 gate。
- 有效 EXP005 smoke/audit 分别为
  `RUN-20260811-061212-smoke-s42-a01`、
  `RUN-20260811-062719-audit-s42-a01`；有效 EXP009 smoke/audit 分别为
  `RUN-20260811-061226-smoke-s42-a01`、
  `RUN-20260811-062736-audit-s42-a01`。用户回传四者均完成；两个 smoke 均生成
  `checkpoints/model_epoch_001.pth`。
- 两端旧历史 outputs/logs/cache/audit/runtime/releases/code 已在重建前清理；
  lab1 的 `chx_old_20260801` 已删除。数据集、官方权重和 official baseline
  保留。
- lab0/lab1 baseline 现在统一位于各自
  `/data/labs/<lab>/docker_data/chx/baselines/official_gt`。
- 稳定 environment image 仍为
  `sha256:f3055cb660032bbb4c1b7cfd9b1840a6c98359d0562a3a4f0601f7238f7291ee`，
  build-source 为 `35313ae3d4139a559a97c01b2d3ee007dc16604c`。
- source commit 与 image build-source 分别记录，不要求相等。当前正式训练冻结
  release、容器和镜像；不得在运行目录 pull、修改文件或替换环境。

以下 2026-08-10 段落保留迁移背景；与本节冲突时以本节及重新执行的只读检查
为准。

## 2026-08-10 启动决策

- 新构建镜像保留并作为 lab0/lab1 共用的稳定 environment image；不删除、不因
  EXP010 等普通 Python/config 提交重复构建。
- source Git commit、resolved config 与 environment image/native identity 分别
  记录，不再要求 image build-source revision 等于实验 source commit。
- 新实验使用 detached、只读 release checkout；镜像 native artifacts 经环境
  契约验证后复制到 Git 忽略 overlay，再将 release 挂载为实际执行源码。
- Dockerfile、requirements/vendor 或 C++/CUDA native/ABI 变化时环境契约拒绝
  复用，届时才允许重建镜像。

- C2 运行链已完成并清理；新容器只使用后续确定的 source release。
- lab0 重建 `lab0_chx` 运行 EXP005/B，lab1 重建 `lab1_chx` 运行 EXP009/CPM。
- EXP005 和 EXP009 的所有新 run 固定 seed `42`。历史失败和 C2 不改 seed。
- 新入口为 `docker/l40/managed_experiment.sh`。GPU 上存在其他任务只记录占用，
  不据此自动取消；实际 CUDA OOM 才保留失败 attempt 并延迟重试。
- 以下旧 Stage 3C 状态段保留作历史运行链说明；实际启动以本节、Gitee release
  commit 和重新执行的只读检查为准。

本文是新对话处理服务器、Docker、GPU分配和Stage 3C实验运行时的首要
入口。它记录当前操作事实和下一步顺序；历史迁移、镜像构建和C1验证细节
仍保存在 `research/SERVER_MIGRATION_HANDOFF.md`。

实时状态会变化。启动、停止或判断实验前，必须重新运行本文的只读检查，
不能仅凭“最后观察”直接操作进程。

## 新对话从这里开始

本地仓库：

```text
/home/wsluser/GDRNPP-RGBD
```

依次执行或阅读：

```bash
cd /home/wsluser/GDRNPP-RGBD
git status --short --branch
```

```text
AGENTS.md
research/HANDOFF.md
research/SERVER_RUNTIME_STATUS_CN.md
```

只有需要追溯迁移、镜像构建或C1原始环境时，再读取：

```text
research/SERVER_MIGRATION_HANDOFF.md
```

旧服务器状态段曾以 `25fb5e3` 为本地基准；该信息已经过时。新的 EXP005/
EXP009 必须以本轮实际推送到 Gitee 的 full release commit 为准，后续对话仍须
重新运行 `git status` 和 `git rev-parse HEAD`。

## 实验室规则和安全边界

- 深度学习任务只能在Docker容器中运行，不应直接使用宿主机Conda环境训练。
- `lab0`账户只能使用物理GPU 0。
- `lab1`账户只能使用物理GPU 1。
- 两个账户应尽量使用满足任务的最少GPU数量。
- 使用 `/usr/bin/docker`，不要使用需要管理员密码的 `sudo docker` 别名。
- 容器采用实验室命名规则；本项目当前使用 `lab0_chx` 和 `lab1_chx`。
- 不得停止、杀死或修改其他用户的进程。
- 不得修改或删除其他用户的容器、镜像、目录、数据、挂载、权限或所有权。
- 不运行 `docker prune`，不运行Git reset/clean，不覆盖已有实验输出。
- 用户在服务器终端中执行命令并返回输出；助手不得主动SSH、生成SSH密钥或
  保存Gitee口令/PAT。
- 对不符合GPU或Docker规定的外部任务，只保存只读证据并联系管理员，不自行
  处置。

## 账户、GPU和Docker关系

| 账户 | 分配的物理GPU | 容器内可见设备 | 当前实验角色 |
|---|---:|---:|---|
| `lab0` | GPU 0 | 通常为CUDA device 0 | B：Patch-PnP-only |
| `lab1` | GPU 1 | 通常为CUDA device 0 | EXP009：CPM-Head |

服务器GPU为NVIDIA L40，每张约46,068 MiB。

`lab0`和`lab1`连接同一个宿主机Docker daemon。因此：

- 镜像在宿主机范围共享，不需要仅因切换账户而重复构建；
- 容器名称在宿主机范围全局唯一，不是每个Linux账户各有一套同名空间；
- `/usr/bin/docker ps -a`可能显示其他账户创建的容器；
- Docker没有可靠的“创建者账户”字段，归属需要结合容器名、进程用户、挂载
  路径和实验室登记判断；
- 宿主机 `nvidia-smi` 显示宿主机PID。仅凭Python可执行文件路径不能完全判断
  进程是否位于容器中，必须结合 `/proc/<PID>/cgroup` 或 `docker top`。

## 当前容器和镜像

| 用途 | 账户/GPU | 容器 | 镜像 | 状态 |
|---|---|---|---|---|
| EXP005/B | lab0/GPU0 | `lab0_chx` | 稳定 environment image | `652d7fd...` formal Epoch 40 完成；待替换为 EXP010 容器 |
| EXP009/CPM | lab1/GPU1 | `lab1_chx` | 稳定 environment image | `652d7fd...` gate/smoke/audit PASS；formal 运行中 |

B/C2镜像最后观察到的ID：

```text
sha256:04a68c9b5b54de9bd386e120d52dd9bf812c5d20ec9b24bf06e7829d4c983495
```

C1镜像ID：

```text
sha256:8e2ee36cae8c9916c6f98b2e29d7c0c9d8cde4d06daca31532f2f7ca47891a99
```

以服务器重新执行的 `docker inspect` 为最终依据。

## 服务器目录、数据和挂载

### lab1

工作根目录：

```text
/data/labs/lab1/docker_data/chx
```

代码：

```text
/data/labs/lab1/docker_data/chx/releases/GDRNPP-RGBD-652d7fd9d38f
```

主要资源：

```text
/data/labs/lab1/docker_data/chx/datasets/BOP_DATASETS/lm
/data/labs/lab1/docker_data/chx/datasets/BOP_DATASETS/lmo
/data/labs/lab1/docker_data/chx/datasets/VOC/VOC2012
/data/labs/lab1/docker_data/chx/weights/lmo_pbr/model_final_wo_optim.pth
/data/labs/lab1/docker_data/chx/baselines/official_gt
```

运行数据：

```text
/data/labs/lab1/docker_data/chx/outputs
/data/labs/lab1/docker_data/chx/logs
/data/labs/lab1/docker_data/chx/cache
/data/labs/lab1/docker_data/chx/audit
```

### lab0

工作根目录：

```text
/data/labs/lab0/docker_data/chx
```

代码：

```text
/data/labs/lab0/docker_data/chx/releases/GDRNPP-RGBD-652d7fd9d38f
```

已观察到 `lab0_chx` 使用lab0自有资源：

```text
/data/labs/lab0/docker_data/chx/datasets/BOP_DATASETS
/data/labs/lab0/docker_data/chx/datasets/VOC
/data/labs/lab0/docker_data/chx/weights
/data/labs/lab0/docker_data/chx/baselines/official_gt
/data/labs/lab0/docker_data/chx/outputs
/data/labs/lab0/docker_data/chx/logs
/data/labs/lab0/docker_data/chx/cache
```

数据集、VOC、权重和基线应只读挂载；outputs、logs和cache可写。不要通过修改
lab1目录权限来解决lab0访问问题。

官方初始化权重：

```text
file:   model_final_wo_optim.pth
size:   410708489 bytes
sha256: bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c
```

## Git和Gitee工作方式

```text
repository: https://gitee.com/Aa1156433279/gdrnpp-rgbd.git
branch:     main
```

- 个人WSL工作区是代码修改和提交的来源。
- 服务器只执行 `fetch`、确定 commit 的 detached checkout、只读 release 准备与
  实验，不在服务器提交或 push，也不在活动 release 中直接 pull。普通
  Python/config 更新复用稳定 environment image；只有环境依赖、native/ABI 或
  Dockerfile 变化才重新构建镜像。
- Gitee HTTPS凭据由用户交互输入，不写入脚本或仓库。
- 服务器存在未提交修改时先运行 `git status --short`；不得reset或覆盖。lab1
  曾观察到 `train_stage3c1.py` 有本地修改，后续必须重新核验其是否仍存在。
- 新终端不需要重新设置 `CHX_ROOT`、`ASSET_ROOT`、`GPU_ID` 或
  `CONTAINER_NAME`。账户脚本使用固定默认值，并主动清除旧终端遗留的覆盖值。

## 当前实验进度

### C1：质量/覆盖模块

```text
状态：FORMAL COMPLETE — C1_SCREEN_FAIL
固定Epoch 40 BOP AR：68.9742%
固定Epoch 40 ADD(-S)：50.57%
逐物体非负：4/8
```

C1容器、日志、输出和权重是不可变实验依据，不用于覆盖式续跑。

### B：Patch-PnP-only

- 历史 Stage 3C 链曾有一轮 smoke 成功，checkpoint 隔离验证通过；当前
  `652d7fd...` managed release 已重新完成独立验收。
- `RUN-20260811-052852-smoke-s42-a01` 因只读 cache 无效，只作为基础设施证据。
- 有效 smoke `RUN-20260811-061212-smoke-s42-a01` 与 audit
  `RUN-20260811-062719-audit-s42-a01` 已完成，smoke checkpoint 已生成。
- 曾误同时启动smoke和formal；该次过早formal已终止，退出码143。
- 无效输出已隔离到：

```text
/data/labs/lab0/docker_data/chx/outputs/stage3c/B_patch_pnp_INVALID_EARLY_20260805_234436
```

- 该次过早formal不计入实验结果。
- 用户已确认有效正式 B 训练正在 `lab0_chx` 中运行；正式 `run_id` 尚待从
  `latest_formal_run.path` 核验。正式结果固定使用 Epoch 40。

### EXP009：CPM-Head

- `RUN-20260811-052906-smoke-s42-a01` 因同一只读 cache 问题无效，不构成 CPM
  科学失败。
- 有效 smoke `RUN-20260811-061226-smoke-s42-a01` 与 audit
  `RUN-20260811-062736-audit-s42-a01` 已完成，smoke checkpoint 已生成。
- 用户已确认 formal 正在 `lab1_chx` 中运行；正式 `run_id` 尚待从
  `latest_formal_run.path` 核验。不得使用 LM-O 中间 checkpoint 选模。

### C2：Patch-PnP + 质量/覆盖联合训练

- 正式训练已于 2026-08-10 完成 40 epoch。
- 固定 Epoch 40 BOP AR 为 `0.6930057670`，未通过预注册筛选门槛。
- 最小归档完成后，旧容器和 raw server outputs 已删除；完整身份和证据边界见
  `research/experiments/EXP-20260805-008-stage3c2-joint-adaptation/`。

### 2026-08-06最后观察到的外部GPU进程

GPU1曾显示：

```text
/data/ipcom/M2023/zwg/home/miniconda3/envs/thu_ai_310/bin/python
```

GPU0曾显示：

```text
/data/ipcom/M2023/niyu/nyhome/miniconda3/envs/games/bin/python
```

这些路径不属于本项目。`thu_ai_310` 和 `games` 是Conda环境名，不能据此判断
具体研究任务，也不能仅凭路径断言是否位于Docker中。它们只是最后观察记录，
不得在未重新检查PID、用户和cgroup时作为当前结论。

## 当前受管入口

在各自确定的 detached release 目录中使用统一入口：

```bash
docker/l40/managed_experiment.sh lab0 EXP005 status
docker/l40/managed_experiment.sh lab1 EXP009 status
```

当前 formal 已启动，只允许只读查看状态和日志；不要重新执行 `launch`，不要在
release 目录 pull 或修改，不要替换容器或镜像。精确日志指针如下：

```text
/data/labs/lab0/docker_data/chx/logs/managed/EXP-20260731-005-pnp-only-control/latest_formal_run.path
/data/labs/lab1/docker_data/chx/logs/managed/EXP-20260809-009-cpm-head/latest_formal_run.path
```

每个 run 内以 `meta/run_manifest.json`、`meta/run_state.json`、
`meta/launcher_status.json`、`train/epoch_summary.jsonl`、`train/console.log` 和
`checkpoints/` 为正式依据。详细只读命令见 `research/RUNBOOK_CN.md`。

## 只读核验GPU、用户和Docker归属

查看账户分配GPU上的进程：

```bash
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

查看某个PID的Linux用户、命令和cgroup：

```bash
PID=替换为实际PID
ps -o user,uid,pid,ppid,lstart,etime,cmd -p "$PID"
readlink -f "/proc/$PID/exe"
cat "/proc/$PID/cgroup"
```

查看同一Docker daemon中的全部容器及镜像：

```bash
/usr/bin/docker ps -a --no-trunc \
  --format 'NAME={{.Names}} STATUS={{.Status}} IMAGE={{.Image}} ID={{.ID}}'
```

```bash
/usr/bin/docker images \
  --format 'REPOSITORY={{.Repository}} TAG={{.Tag}} ID={{.ID}} CREATED={{.CreatedSince}} SIZE={{.Size}}'
```

检查某个宿主机PID是否属于运行中的容器：

```bash
PID=替换为实际PID
for C in $(/usr/bin/docker ps -q); do
  /usr/bin/docker top "$C" -eo user,pid,ppid,etime,cmd 2>/dev/null |
    awk -v pid="$PID" -v c="$C" '$2 == pid {print "CONTAINER_ID=" c; print}'
done
```

找到容器ID后查看名称、镜像、用户和挂载：

```bash
/usr/bin/docker inspect 替换为容器ID \
  --format 'NAME={{.Name}} IMAGE={{.Config.Image}} USER={{.Config.User}} INIT_PID={{.State.Pid}}'
```

```bash
/usr/bin/docker inspect 替换为容器ID \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

上述命令均为只读。发现其他用户占用分配GPU时，保存带时间的
`nvidia-smi`、`ps`、`cgroup`输出并交给管理员，不执行kill。

## 下一步固定顺序

1. 本地提交并 push 已通过测试的 cache/home 与异常退出修复；不重建镜像。
2. lab0/lab1 分别 checkout 新 detached release，执行 source/environment
   binding；环境契约必须仍等于稳定镜像契约。
3. 保留两个失败 run，删除并重建本项目的 `lab0_chx`/`lab1_chx` 容器，使新
   home/cache 挂载生效；不删除数据、权重、baseline 或稳定镜像。
4. 两端重新通过 gate 后，各创建唯一 run 重试 smoke；先确认训练退出码、
   `model_epoch_001.pth` 和 checkpoint isolation 均通过。
5. smoke 成功后运行 batch-48 audit；成功后才启动 fixed-seed Epoch 40 formal。
6. GPU 占用只记录为非阻塞快照，不停止或修改其他用户进程；若本 run 实际
   CUDA OOM，则保留失败 attempt 并按受管规则重试。
7. 训练过程中只通过 status/watch 查看；VPN 断开不视为训练停止。
8. 两个正式实验完成后汇总 BOP AR、ADD(-S)、逐物体结果和权重 SHA-256。

## 正式启动前验收条件

- `git status` 已检查，服务器本地修改得到保留；
- 账户、物理GPU和容器名对应正确；
- 数据集、VOC、官方权重和基线路径可读；
- 官方权重SHA-256一致；
- 容器运行门通过；
- 对应角色smoke退出码为0；
- checkpoint隔离验证通过；
- NUM_WORKERS已由同batch协议测试并固定；
- 已记录物理 GPU 占用快照；外部占用不自动阻断，实际 CUDA OOM 按失败规则处理；
- 正式输出目录尚不存在，不会覆盖历史结果；
- 使用预注册固定种子，且没有同时启动同角色的第二个formal。
