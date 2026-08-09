# GDRNPP 服务器运行环境与实验进度交接

最后更新：2026-08-10

## 2026-08-10 启动决策

- 新构建镜像保留并作为 lab0/lab1 共用的稳定 environment image；不删除、不因
  EXP010 等普通 Python/config 提交重复构建。
- source Git commit、resolved config 与 environment image/native identity 分别
  记录，不再要求 image build-source revision 等于实验 source commit。
- 新实验使用 detached、只读 release checkout；镜像 native artifacts 经环境
  契约验证后复制到 Git 忽略 overlay，再将 release 挂载为实际执行源码。
- Dockerfile、requirements/vendor 或 C++/CUDA native/ABI 变化时环境契约拒绝
  复用，届时才允许重建镜像。

- 用户提供的最新只读检查确认旧 `lab1_chx` 中 C2 仍在运行；保持原容器、
  镜像、代码和输出不变。
- 先在 lab0 准备新的 `lab0_chx`，运行 EXP005/B；旧同名容器须先只读检查并
  以 legacy 名称保存，不删除。
- C2 Epoch 40 完成并封存后，再以确定 Git commit 和同一 environment image ID
  重建
  `lab1_chx`，运行 EXP009/CPM。
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
| `lab1` | GPU 1 | 通常为CUDA device 0 | C2：Patch-PnP + 质量/覆盖联合训练 |

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
| 已完成C1证据 | lab1/GPU1 | `lab1_chx_stage3c1` | `gdrnpp-stage3c1:torch220-cu121-sm89-v1` | 保留，不修改 |
| B实验 | lab0/GPU0 | `lab0_chx` | `gdrnpp-stage3bc2:torch220-cu121-sm89-v2` | smoke已通过，正式训练未有效启动 |
| C2实验 | lab1/GPU1 | `lab1_chx` | `gdrnpp-stage3bc2:torch220-cu121-sm89-v2` | 容器已准备，流水线等待GPU1 |

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
/data/labs/lab1/docker_data/chx/code/GDRNPP-RGBD
```

主要资源：

```text
/data/labs/lab1/docker_data/chx/datasets/BOP_DATASETS/lm
/data/labs/lab1/docker_data/chx/datasets/BOP_DATASETS/lmo
/data/labs/lab1/docker_data/chx/datasets/VOC/VOC2012
/data/labs/lab1/docker_data/chx/weights/lmo_pbr/model_final_wo_optim.pth
/data/labs/lab1/docker_data/chx/outputs/EXP-20260731-006/official_gt
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
/data/labs/lab0/docker_data/chx/code/GDRNPP-RGBD
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
- 服务器只执行pull、Docker构建和实验，不在服务器提交或push。
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

- 一轮smoke已成功完成，退出码为0。
- checkpoint隔离验证通过：只允许的Patch-PnP张量发生变化，冻结张量未变化。
- 曾误同时启动smoke和formal；该次过早formal已终止，退出码143。
- 无效输出已隔离到：

```text
/data/labs/lab0/docker_data/chx/outputs/stage3c/B_patch_pnp_INVALID_EARLY_20260805_234436
```

- 该次过早formal不计入实验结果。
- 有效正式B训练尚未开始。
- NUM_WORKERS benchmark尚未产生可采用的推荐结果；GPU0最后观察到被其他用户
  进程占用。重新开始前先检查是否仍有旧的后台等待器。

### C2：Patch-PnP + 质量/覆盖联合训练

- 已运行 `docker/l40/lab1_c2.sh benchmark-workers`。
- 一键流水线最后状态为 `PREPARING/WAITING_GPU1`。
- 等待期间尚未开始C2 smoke、NUM_WORKERS实测或正式训练。
- 流水线使用nohup后台运行；VPN或VS Code SSH断开不会自动终止。
- GPU1释放后，流水线设计为自动继续完成gate、smoke、隔离验证和worker测试。

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

## 一键脚本和常用命令

在lab0中执行B：

```bash
cd /data/labs/lab0/docker_data/chx/code/GDRNPP-RGBD
docker/l40/lab0_b.sh benchmark-status
docker/l40/lab0_b.sh benchmark-watch
```

需要重新启动worker测试时：

```bash
docker/l40/lab0_b.sh benchmark-workers
```

在lab1中执行C2：

```bash
cd /data/labs/lab1/docker_data/chx/code/GDRNPP-RGBD
docker/l40/lab1_c2.sh benchmark-status
docker/l40/lab1_c2.sh benchmark-watch
```

`benchmark-watch` 中按Ctrl-C只停止查看日志，不停止后台流水线。

worker测试完成并正式冻结NUM_WORKERS之前，不运行：

```bash
docker/l40/lab0_b.sh formal
docker/l40/lab1_c2.sh formal
```

正式启动后查看：

```bash
docker/l40/lab0_b.sh status
docker/l40/lab0_b.sh watch
```

```bash
docker/l40/lab1_c2.sh status
docker/l40/lab1_c2.sh watch
```

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

1. 分别重新检查物理GPU0和GPU1的实时占用及进程归属。
2. 若分配GPU仍被其他用户占用，保留证据并联系管理员；不要并发争抢资源。
3. GPU1释放后，优先让现有C2流水线完成gate、smoke、隔离验证和NUM_WORKERS
   benchmark。
4. 根据benchmark结果固定C2的NUM_WORKERS；不改变batch size 48，不改变正式
   实验的预注册种子。
5. GPU0释放后完成B的同协议NUM_WORKERS benchmark并冻结其设置。
6. 确认B与C2各自smoke通过、输出目录不存在冲突、GPU空闲后，才分别启动正式
   40轮训练。
7. 每个正式实验只使用一个固定种子，不进行随机种子重复，不临时启动其他实验。
8. 训练过程中只通过status/watch查看；VPN断开不视为训练停止。
9. 两个正式实验完成后再汇总BOP AR、ADD(-S)、逐物体结果和权重SHA-256。
10. ADD结果路径匹配错误和重复警告过多的问题已知，但等当前正式训练结束后再
    修复，避免中途改变运行代码。

## 正式启动前验收条件

- `git status` 已检查，服务器本地修改得到保留；
- 账户、物理GPU和容器名对应正确；
- 数据集、VOC、官方权重和基线路径可读；
- 官方权重SHA-256一致；
- 容器运行门通过；
- 对应角色smoke退出码为0；
- checkpoint隔离验证通过；
- NUM_WORKERS已由同batch协议测试并固定；
- 物理GPU没有其他计算进程；
- 正式输出目录尚不存在，不会覆盖历史结果；
- 使用预注册固定种子，且没有同时启动同角色的第二个formal。
