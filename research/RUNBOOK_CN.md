# 实验、Git 与服务器运行手册

## EXP013 A/B 与 C revision 2

GitHub 上确定的 40 位 commit 建立 detached、只读 release 后，lab0 对 `EXP013A`、lab1 对 `EXP013B` 分别执行：

```bash
docker/l40/managed_experiment.sh <lab0|lab1> <EXP013A|EXP013B> access
docker/l40/managed_experiment.sh <lab0|lab1> <EXP013A|EXP013B> create
docker/l40/managed_experiment.sh <lab0|lab1> <EXP013A|EXP013B> gate
docker/l40/managed_experiment.sh <lab0|lab1> <EXP013A|EXP013B> smoke
docker/l40/managed_experiment.sh <lab0|lab1> <EXP013A|EXP013B> audit48
docker/l40/managed_experiment.sh <lab0|lab1> <EXP013A|EXP013B> launch
docker/l40/managed_experiment.sh <lab0|lab1> <EXP013A|EXP013B> finalize
```

两端必须绑定同一 source commit、官方 checkpoint SHA、environment image ID 和 seed 42。E40 是正式比较点；E5–E35 只观察轨迹，不提前停止。

### 服务器无法连接 GitHub/Gitee 时的 bundle release

本地对已经验证并推送的确定 commit 创建完整 Git bundle，上传到对应账户可写的
`/data/labs/<lab>/docker_data/chx/transfer/`。服务器从 bundle clone 到新的
`releases/GDRNPP-RGBD-<short-commit>`，再执行 detached checkout、clean 检查和
`prepare_release.sh`。bundle 只替代网络传输，不改变 source commit、release
只读、环境镜像和实验 gate 约束。整个流程使用 `lab0`/`lab1` 普通账户，不使用
`sudo`，也不向系统目录写文件。

### 固定容器名冲突的保留流程

受管容器固定使用 `lab0_chx`/`lab1_chx`。新 release 执行 `create` 时若提示容器名
已存在，先执行：

```bash
docker/l40/managed_experiment.sh <lab0|lab1> <EXP别名> preserve
docker/l40/managed_experiment.sh <lab0|lab1> <EXP别名> create
```

`preserve` 会检查所有受管 formal supervisor 和旧容器进程。存在训练任务时直接
拒绝；旧容器空闲时只把它重命名为 `<lab>_chx_legacy_<UTC时间>`，不停止、不删除，
随后才允许新容器占用固定名称。不得用 `sudo`、`docker rm` 或 `docker stop` 绕过
保护。该问题不写入 `AGENTS.md`：后者只保存长期 Agent 路由与安全规则，服务器
操作故障和恢复步骤统一记录在本运行手册。

原始 B-based C 的条件是 A、B 均通过相对 EXP012 E40 的门槛；由于 B 严格 BOP gate 失败，该版本没有运行。2026-08-26 在任何 C formal run 之前，C 明确修订为继承已通过 gate 的 A，并完成新的本地 CPU/CUDA preflight 与完整 1 epoch smoke。用户审阅后授权 revision 2，因此 metadata 为 `AUTHORIZED`；lab0 必须从授权后的新 commit 建立 release，并对 `EXP013C` 重新执行完整 `access→create→gate→smoke→audit48→launch→finalize` 序列，不能复用 A 的 run 或跳过门槛。

A/B E40 完成后用 `python -m research.exp013.diagnostics` 运行 1,445 个 LM-O GT-bbox targets、五个 XYZ alpha 和 fixed-pred/synced/region0 三条路径。诊断不更新模型状态，也不替代正式精度 gate。

## EXP013D（ImageNet 全量端到端训练）

分支 `exp014-d-fulltrain`，服务器离线时同样走 bundle 流程（上传
`/data/labs/lab1/docker_data/chx/transfer/` → clone 到
`releases/GDRNPP-RGBD-<short>` → detached checkout 40 位 commit → clean 检查 →
`docker/l40/prepare_release.sh lab1 <image sha256:f3055cb6…>`）。lab1 对
`EXP013D` 执行完整序列：

```bash
docker/l40/managed_experiment.sh lab1 EXP013D access
docker/l40/managed_experiment.sh lab1 EXP013D create
docker/l40/managed_experiment.sh lab1 EXP013D gate
docker/l40/managed_experiment.sh lab1 EXP013D smoke
docker/l40/managed_experiment.sh lab1 EXP013D audit48
docker/l40/managed_experiment.sh lab1 EXP013D launch
docker/l40/managed_experiment.sh lab1 EXP013D finalize
```

与 EXP013 的关键差异：无官方 ckpt（`MODEL.WEIGHTS=""`），主干/几何头/姿态头全部
解冻，几何监督开启（训练渲染器 egl），WARMUP_ITERS=1000。服务器首道 gate 是 egl
渲染器验证——失败则报告，按预注册规则降级 cpp 并在 RECORD 记录。timm 的
`convnext_base_1k_224_ema.pth`（~330MB）随 bundle 交付到
`E:\6D姿态估计\EXP-014\`，容器内放入 `$TORCH_HOME/hub/checkpoints/`（或 HF 缓存），
避免离线训练启动时下载失败。

## EXP013E（官方头随机初始化冻结对照）

分支 `exp013e-official-random`，目标 lab0，bundle 流程与 EXP013D 相同。lab0 对
`EXP013E` 执行完整序列：

```bash
docker/l40/managed_experiment.sh lab0 EXP013E access
docker/l40/managed_experiment.sh lab0 EXP013E create
docker/l40/managed_experiment.sh lab0 EXP013E gate
docker/l40/managed_experiment.sh lab0 EXP013E smoke
docker/l40/managed_experiment.sh lab0 EXP013E audit48
docker/l40/managed_experiment.sh lab0 EXP013E launch
docker/l40/managed_experiment.sh lab0 EXP013E finalize
```

与 EXP013A/B/C 的关键差异：头为官方 `ConvPnPNet` **随机初始化**；`MODEL.WEIGHTS`
指向官方 ckpt 的 pnp 剥离派生文件 `model_final_wo_optim_wo_pnp.pth`——gate 步骤
会先执行 `python -m research.exp013.e_prep` 从 SHA 校验的官方文件确定性生成
（官方 pnp 键与重建头同名，直接加载原始 ckpt 会覆盖随机初始化）。训练渲染器为
**关闭**状态（冻结几何监督关闭，引擎不构造 CPP/EGL 渲染器）；决策点固定
epoch_040；预注册读数为诊断型（reS ≥0.54 / 0.52–0.54 / <0.52 三档解释规则）。

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
- lab0 与 lab1 必须从 Gitee 获取同一个确定 commit，分别建立独立、detached、
  只读 release。两台机器执行代码内容相同，差别只允许来自明确的 EXP 身份、
  config、物理 GPU 和各自非覆盖 output；不得维护 lab0/lab1 两套源码。
- dataset、初始化权重、checkpoint、output、cache 和机器挂载路径保留在各服务器
  Git 之外。它们由 path profile、manifest 和哈希绑定，不因源码一致而复制进 Git。
- 已启动的旧 run 永远绑定其原 release commit；新 commit 建新 release，不在旧
  release 上 pull、checkout 或覆盖。不同 commit 的实验可以同时存在，但每个 run
  必须能从 manifest 唯一还原自己的 source 与 environment 身份。

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

不得在正在运行旧实验的代码目录中 pull 或切换版本。受管实验使用
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

C2 已完成 Epoch 40 并按历史最小证据归档；新的 EXP005/EXP009/EXP010 run 使用受管
入口，不复用已删除的旧 C2 容器或输出目录。

## EXP005 / EXP009 / EXP010 受管入口

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

受管容器使用 `${root}/home` 作为账户本地可写 HOME，使用 `${root}/cache`
作为可写 cache。为兼容仍默认写 `PROJ_ROOT/.cache` 的上游 dataset/evaluator，
`${root}/cache/gdrnpp_datasets` 还会嵌套挂载到
`/workspace/gdrnpp/.cache`。该位置只保存 Git 忽略的运行缓存，不改变只读
release 中任何 tracked source 或 native artifact。gate 必须同时验证这些路径
可写。launcher 会在 `docker run` 前创建 release 侧 Git 忽略的空 `.cache`
挂载点，避免 Docker 尝试在只读父挂载中创建它；不能通过取消 source 只读
挂载来解决 cache 权限问题。

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
docker/l40/managed_experiment.sh <lab0|lab1> <EXP005|EXP009|EXP010> finalize
```

EXP010 metadata 状态为 `AUTHORIZED`。EXP005/B 固定 Epoch 40 已完成；EXP010
可在 lab0 与 lab1 的 EXP009 并行，最终比较仍等待两者固定 Epoch 40。它必须
建立新的确定 commit release，复用同一稳定 environment image；不修改或续用
EXP009 的 release、checkpoint 或 output。

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
