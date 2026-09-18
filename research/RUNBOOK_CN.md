# 实验运行手册

## 本地检查

```bash
source /home/wsluser/miniconda3/etc/profile.d/conda.sh
conda activate pytorch22
PYTHONPATH="$PWD" pytest -q research/tests research/next_pose_head/tests \
  research/exp013/tests research/diagnostics/pose_structure/tests
python -m research.next_pose_head.preflight --device cpu
python -m research.exp013.preflight --variant A --device cpu
python -m research.exp017.preflight --device cpu
python -m research.exp017.real_smoke --device cuda:0 --batch-size 2 --num-workers 0
```

其他 EXP013 变体把 `A` 改为 `B/C/E/F`。D 仍暂停，不运行 formal。
EXP017 的 `real_smoke` 只执行一个真实小批次和三个受限 optimizer step；服务器一 epoch
smoke 使用 EXP017 的 `smoke.py`，不得把本地 smoke 当作 formal。

长期 PnP-only matched control 使用：

```bash
docker/l40/experiment.sh lab0 run EXP-20260731-005-pnp-only-control \
  configs/gdrn/lmo_pbr/research/controls/pnp_only/smoke.py smoke
```

正式训练时把配置换为 `train.py`。该入口按当前代码体系维护，用于未来实验的统一
matched comparison，不替代历史 EXP005 的精确复现。

## 历史实验精确复现

已退出 HEAD 的历史实验不要把旧 config 单独复制回当前 core。先从对应
`research/experiments/*/RECORD.md` 读取 source commit，再用独立 worktree 恢复
当时代码和配置，例如：

```bash
git worktree add ../gdrnpp-exp009 652d7fd9d38f8ea5cea0c5a98cc9477b66623180
cd ../gdrnpp-exp009
```

复现完成后删除 worktree，不把历史执行框架重新并入当前 main。

## 服务器

先读 `research/SERVER_SAFETY_CN.md`，确认代码 checkout、数据、权重、镜像和 GPU。

### Bundle release 固定流程

本地统一使用下面的入口创建 bundle；脚本要求 HEAD 位于已附着分支、working tree
（包括 untracked 文件）为空、目标 bundle 不存在，并在创建后执行完整验证。bundle
只包含当前分支及其可达历史，不要求切换到 `main`；因此可以保持 `main` 干净，并
直接从当前研究分支生成 release：

```bash
docker/l40/create_bundle.sh
```

bundle 默认写入 `.local/release/GDRNPP-RGBD-<short-sha>.bundle`。不要直接用裸
`git bundle create` 绕过 clean-tree gate。

服务器不要依赖当前工作目录解析 `transfer/...` 或 `releases/...`。`git bundle verify` 需要 Git
仓库上下文，因此先创建一次性 bare verification repo；它位于 `/tmp`，不作为 release。
下面以 `lab0`、short SHA `3cfbceb` 为例，实际使用时只替换四个标量。整个流程放在
subshell 中并启用失败即停止，前一步失败后不会继续 clone 或 checkout：

```bash
(
set -Eeuo pipefail

machine=lab0
owner=chx
short_sha=3cfbceb
full_sha=3cfbceb94252c1fc35b9f81350d2b6a0c068d97a

root="/data/labs/${machine}/docker_data/${owner}"
bundle="${root}/transfer/GDRNPP-RGBD-${short_sha}.bundle"
release="${root}/releases/GDRNPP-RGBD-${short_sha}"

test "$(id -un)" = "${machine}"
test -f "${bundle}"
test ! -e "${release}"

verify_repo="$(mktemp -d "/tmp/gdrnpp-bundle-verify-${short_sha}.XXXXXX")"
git -c init.defaultBranch=main -C "${verify_repo}" init --bare --quiet
git -C "${verify_repo}" bundle verify "${bundle}"

git clone --no-checkout "${bundle}" "${release}"
git -C "${release}" checkout --detach "${full_sha}"
test "$(git -C "${release}" rev-parse HEAD)" = "${full_sha}"
test -z "$(git -C "${release}" status --short)"

git -C "${release}" status --short --branch
echo "RELEASE_CHECK PASS commit=${full_sha}"
)
```

预期最后两行包含 `HEAD (no branch)` 与 `RELEASE_CHECK PASS`。目标 release 已存在时
命令会停止，不覆盖、不删除也不复用。release 创建后必须从该目录使用当前统一入口；
图示历史中的 `prepare_release.sh` 和 `managed_experiment.sh` 不再使用。

### 已有容器的受控替换

`create` 不覆盖同名容器。仅在用户明确授权替换精确容器、只读检查确认标签归属，并且
容器内没有 `main_gdrn.py` 后，才执行下面的受控替换。示例对应 EXP017 的 `lab0`
release；旧 mount 必须与已检查到的值精确一致，否则停止，不猜测也不删除：

```bash
(
set -Eeuo pipefail

machine=lab0
container=gdrnpp_chx_lab0
expected_old_repo=/data/labs/lab0/docker_data/chx/releases/GDRNPP-RGBD-c1e0dfa
new_repo=/data/labs/lab0/docker_data/chx/releases/GDRNPP-RGBD-3cfbceb
image_ref=gdrnpp-research:torch220-cu121-sm89-c0be1ade7ea9

test "$(id -un)" = "${machine}"
test "$(git -C "${new_repo}" rev-parse HEAD)" = \
  3cfbceb94252c1fc35b9f81350d2b6a0c068d97a
test -z "$(git -C "${new_repo}" status --short)"
test "$(/usr/bin/docker inspect "${container}" \
  --format '{{index .Config.Labels "gdrnpp.project"}}')" = GDRNPP-RGBD
test "$(/usr/bin/docker inspect "${container}" \
  --format '{{index .Config.Labels "gdrnpp.machine"}}')" = "${machine}"

mounted_repo="$(/usr/bin/docker inspect "${container}" \
  --format '{{range .Mounts}}{{if eq .Destination "/workspace/gdrnpp"}}{{.Source}}{{end}}{{end}}')"
test "${mounted_repo}" = "${expected_old_repo}"

active="$(/usr/bin/docker top "${container}" -eo pid,args | awk '
  NR > 1 {
    pid = $1
    $1 = ""
    sub(/^[[:space:]]+/, "", $0)
    if ($0 != "sleep infinity") print pid "\t" $0
  }
')"
if [[ -n "${active}" ]]; then
  printf 'REFUSE: active process in %s:\n%s\n' "${container}" "${active}" >&2
  exit 1
fi

/usr/bin/docker stop "${container}"
/usr/bin/docker rm "${container}"

cd "${new_repo}"
docker/l40/experiment.sh "${machine}" create "${image_ref}"

mounted_repo="$(/usr/bin/docker inspect "${container}" \
  --format '{{range .Mounts}}{{if eq .Destination "/workspace/gdrnpp"}}{{.Source}}{{end}}{{end}}')"
mounted_rw="$(/usr/bin/docker inspect "${container}" \
  --format '{{range .Mounts}}{{if eq .Destination "/workspace/gdrnpp"}}{{.RW}}{{end}}{{end}}')"
test "${mounted_repo}" = "${new_repo}"
test "${mounted_rw}" = false

docker/l40/experiment.sh "${machine}" check
docker/l40/experiment.sh "${machine}" status
echo "CONTAINER_REPLACEMENT PASS container=${container} release=${new_repo}"
)
```

替换只影响经过上述 gate 的项目容器，不操作宿主机其他 Python 进程。若 GPU 上有其他
任务，仍由 launcher 的 free-memory gate 决定是否允许启动。先运行独立 smoke；拿到
`RUN_ID` 后使用 `status`/`logs` 检查，formal 必须再次获得明确授权：

```bash
cd /data/labs/lab0/docker_data/chx/releases/GDRNPP-RGBD-3cfbceb
docker/l40/experiment.sh lab0 run \
  EXP-20260902-017-support-aware-rotation-residual \
  configs/gdrn/lmo_pbr/research/exp017/support_aware_rotation_residual/smoke.py smoke
docker/l40/experiment.sh lab0 status
# 将上一条 run 命令返回的 RUN_ID 代入：
docker/l40/experiment.sh lab0 logs \
  EXP-20260902-017-support-aware-rotation-residual/RUN-...
```

```bash
docker/l40/experiment.sh lab0 check
docker/l40/experiment.sh lab0 create gdrnpp-research:torch220-cu121-sm89-c0be1ade7ea9
docker/l40/experiment.sh lab0 run EXP-... configs/.../smoke.py smoke
docker/l40/experiment.sh lab0 run EXP-... configs/.../train.py formal
docker/l40/experiment.sh lab0 status
docker/l40/experiment.sh lab0 logs EXP-.../RUN-...
```

`run`/`eval` 在创建运行目录前再次强制 working tree clean，并加载 effective config
执行共享 contract：smoke 必须是 1 epoch、batch 不超过 8、无 periodic evaluation；
formal 必须是 seed 42、LM-PBR、LM-O GT-box、40 epoch、batch 48、每 5 epoch checkpoint
与 evaluation。带 `RESEARCH_PROTOCOL.SCHEDULE="configurable"` 的新实验（当前只有
EXP023 LM13）走另一支 formal 规则：只要求 `FORMAL_READY=True`、train/test split 非空、
`EVAL_PERIOD` 在 `(0, TOTAL_EPOCHS]` 内、GT box 与 evaluation renderer；协议数值由
`research/exp022/preflight.py` 的 `check_lm13_gdrn_protocol` 单独硬校验。每次 run 的根目录
写入 `run_metadata.json`，保存完整 source commit、
image ID、image build revision、config、mode 与 run ID。

idle gate 通过 `docker top` 检查容器内除 PID 1 `sleep infinity` 外的全部进程；不能
依赖 `main_gdrn.py` 字符串，因为训练启动后会用 `setproctitle` 改名为配置名和时间戳。

给远端终端的命令优先写成可整体复制的多行 subshell：集中定义 `release`、experiment、
config 等短变量，一行执行一个命令，不使用超长单行或反斜杠续行。简单的单步
`status/logs` 也使用短行和 release 绝对路径。shell 提示符显示的当前 release 不代表
现有容器使用同一源码；每次 `run` 前由 launcher 的 mount gate 核对
`/workspace/gdrnpp` source。若报
`mount /workspace/gdrnpp has source ..., expected ...`，说明容器仍绑定旧 release，
必须按“已有容器的受控替换”检查并重建，不能绕过 gate 或在服务器内修改源码。

### EXP023 LM13 server profile

EXP023 主协议（`TRAIN_PROTOCOL.NAME=lm13_gdrn`）训练 LM real 加 DeepIM 渲染图，并让
ConvNeXt backbone 从 ImageNet 权重初始化，因此比旧的 LMO/PBR 协议多两项服务器资源。
以 `lab0` 为例，`root=/data/labs/lab0/docker_data/chx`；lab1 只替换 `lab0 -> lab1`：

| host | container | 说明 |
|---|---|---|
| `${root}/datasets/lm_imgn` | `/workspace/gdrnpp/datasets/lm_imgn` | DeepIM 渲染图，独立只读 mount |
| `${root}/weights/convnext/convnext_base_1k_224_ema.pth` | `/workspace/gdrnpp/pretrained_models/convnext/convnext_base_1k_224_ema.pth` | 复用 `${root}/weights` 只读 mount |
| `${root}/cache/gdrnpp_datasets/exp022/lm13/independent_v2.npz` | `/home/gdrn/.cache/gdrnpp_datasets/exp022/lm13/independent_v2.npz` | LM13 CAD 层级 |

`create` 自动注入，不需要手工准备：

```bash
--env GDRN_CONVNEXT_BASE_WEIGHTS=/workspace/gdrnpp/pretrained_models/convnext/convnext_base_1k_224_ema.pth
--mount "type=bind,src=${root}/datasets/lm_imgn,dst=/workspace/gdrnpp/datasets/lm_imgn,readonly"
```

配置与命令里不得出现 `/data/labs/...` host 路径；容器内路径与该 env 一律由 launcher 提供。
`lm/test/xyz_crop` 与 `lm_imgn/xyz_crop_imgn` 都**不**需要：两个 LM13 real split 与渲染
split 都用 `require_xyz=False`，评估也不需要 GT XYZ。

上线前在 host 上确认（旧 LMO 机器不会自动具备这些）：

```bash
(
set -Eeuo pipefail
machine=lab0
root="/data/labs/${machine}/docker_data/chx"
test -d "${root}/datasets/BOP_DATASETS/lm/test"
test -d "${root}/datasets/BOP_DATASETS/lm/image_set"
test -d "${root}/datasets/BOP_DATASETS/lm/models"
test -d "${root}/datasets/lm_imgn/image_set"
test -d "${root}/datasets/lm_imgn/imgn"
test -d "${root}/datasets/VOC/VOC2012/JPEGImages"
test -r "${root}/weights/convnext/convnext_base_1k_224_ema.pth"
test -r "${root}/cache/gdrnpp_datasets/exp022/lm13/independent_v2.npz"
echo "LM13_HOST_RESOURCES PASS"
)
```

LM13 CAD 层级 `independent_v2.npz` 不是数据，也不能由训练入口自动生成：缺失时 formal 必须被
资源门拒绝。它必须先在服务器就位，且**不要在 host 的 release 目录里直接跑项目 Python** ——
服务器约定是项目代码一律经 Docker 执行，host 侧 release 目录可能没有 conda 环境，也可能
拿到与镜像不一致的依赖。两条都可行：

方案 A，复制本地已验证的 artifact（推荐，与本地结果逐位一致）。本地文件在仓库的 ignored
cache 里：`<repo>/.local/dataset_cache/exp022/lm13/independent_v2.npz`（约 1.5 MB），
先传到 host，再落到 cache mount：

```bash
# 在本地开发机
scp .local/dataset_cache/exp022/lm13/independent_v2.npz <server>:/tmp/independent_v2.npz

# 在服务器
install -D -m 644 /tmp/independent_v2.npz \
  "${root}/cache/gdrnpp_datasets/exp022/lm13/independent_v2.npz"
```

方案 B，建好容器后在容器内生成（先 `create`，再按上面的 host 检查复核）：

```bash
docker exec -w /workspace/gdrnpp -e PYTHONPATH=/workspace/gdrnpp gdrnpp_chx_lab0 \
  python -m research.exp022.build_hierarchy --mode independent \
  --config configs/gdrn/research/exp022_progressive_pcc/train_lm13_gdrn.py
```

方案 B 生成后必须重跑下面的 `server_preflight` 与 `preflight`：层级是新的 artifact，先前的
检查结果对它不再有效。

#### profile-specific runtime gate

`run`/`eval` 在创建 run 目录前，先把容器内 `mmcv.Config.fromfile` 的
`TRAIN_PROTOCOL.NAME` 读回来选择资源门（不按文件名或 EXP 编号猜）：

| `TRAIN_PROTOCOL.NAME` | profile | 资源门 |
|---|---|---|
| `lm13_gdrn` | `lm13` | LM real + `lm_imgn` + VOC + ConvNeXt + LM13 hierarchy + `server_preflight` |
| `lm13_real_only` | `lm13` | 同上，但按配置实际的 `DATASETS.TRAIN` **不要求** `lm_imgn` |
| `lm13_pbr` | `lm13_pbr` | LM PBR + LM real test + VOC + ConvNeXt + LM13 hierarchy + `server_preflight`；不要求 `lm_imgn` |
| `lmo_full_imagenet` | `lmo_full_imagenet` | LM PBR + LM-O test + VOC + ConvNeXt ImageNet 权重 + `reused_v1.npz` + 容器内 EXP022 `preflight`；不要求官方 LM-O 训练权重 |
| 缺失（全部旧 LMO、PBR 与 EXP013/017/020/021 配置） | `legacy_lmo` | `lm/train_pbr` + `lmo/test` + VOC + `weights/lmo_pbr/model_final_wo_optim.pth` |
| 其他非空名称 | — | 直接 fail：`unknown TRAIN_PROTOCOL.NAME`，不再回落到 legacy |

`lm13` profile 下是否需要 `lm_imgn` 由容器内读回的 `DATASETS.TRAIN` 决定（存在任一
`lm_imgn*` split 才要求），因此 `lm13_real_only` 这条消融臂不需要服务器准备 DeepIM 渲染图。
新增非空 `TRAIN_PROTOCOL.NAME` 时必须同时更新 launcher 的映射表：未列出的名字会被拒绝，
不会静默套用旧 LMO 的资源清单。

EXP024 的 `train_lmo_full_imagenet.py` 在 lab1 使用独立实验 ID。准备版 `a357017`
保持 `FORMAL_READY=False`，其容器内 CPU preflight 和 EGL/AMP 真实 batch4 smoke
已通过；本地随后开启 `FORMAL_READY=True`。正式训练须用第二段 clean commit/release，
由 launcher 的 `run ... formal` 启动。40 epoch 的 E5–E40 checkpoint 与评估口径见
EXP024 RECORD。不要在服务器
release 中直接修改配置，也不要把 EXP024 的运行目录放进 EXP022。

`check_host()` 只保留 user/Docker/GPU 与 `${root}` 基础路径；`${root}` 下的
`datasets/weights/outputs/cache/home` 由 `create` 负责建立（`--mount type=bind` 不接受不存在的
来源），数据集与权重内容改由上面的 profile 门在容器内检查。缺任一项时 `run` 在创建输出目录
之前就失败。

新 release 的 `verify_required_mounts` 要求 `datasets/lm_imgn` 这一条 mount，而旧容器没有
它（legacy profile 也走同一个 mount 门，只是不查 `lm_imgn` 的数据内容）。因此从本次修改
起，**已有容器必须按“已有容器的受控替换”重建一次**才能继续 `run`；空目录是允许的，
数据缺失只会让 LM13 gate 拒绝，不会影响旧 LMO 实验。

容器内两项手工复核（formal 前各跑一次）：

```bash
docker exec -w /workspace/gdrnpp -e PYTHONPATH=/workspace/gdrnpp gdrnpp_chx_lab0 \
  python -m research.exp022.server_preflight \
  --config configs/gdrn/research/exp022_progressive_pcc/train_lm13_gdrn.py
docker exec -w /workspace/gdrnpp -e PYTHONPATH=/workspace/gdrnpp gdrnpp_chx_lab0 \
  python -m research.exp022.preflight \
  --config configs/gdrn/research/exp022_progressive_pcc/train_lm13_gdrn.py
```

`server_preflight` 报告 profile、train/test split、hierarchy、ConvNeXt、VOC 以及各 split
记录数（应读到 LM real train `2375`、`lm_imgn` `13000`、LM test `13425`）。
`preflight` 另外核对 340 个 ImageNet 张量、hierarchy 对象顺序、可训练参数集合、
forward/backward/optimizer step 与推理输出形状。两者都不允许为通过而放宽或吞异常。

#### EGL smoke 与两段 release

release 只读，不要在服务器上编辑 config。第一段 release 保持
`RESEARCH_PROTOCOL.FORMAL_READY=False`，只跑 smoke：

```bash
docker/l40/experiment.sh lab0 run \
  EXP-20260918-023-lm13-progressive-pcc-fulltrain \
  configs/gdrn/research/exp022_progressive_pcc/smoke_lm13_gdrn.py \
  smoke
```

smoke 必须真的使用 `XYZ_RENDERER=egl`、AMP、ConvNeXt checkpoint 与两个训练域，并检查
`forward` / 有限 loss / `backward` / 有限 grad / `optimizer.step` / AMP 无跳步 /
checkpoint 写入 / `exit_code=0`。本仓库当前本地开发机的 EGL 抛
`RuntimeError: Bindless Textures not supported`，所以这一步只能在服务器完成；不要在本地
用 CPP 结果代替它。

EGL smoke PASS 后：回本地打开 `train_lm13_gdrn.py` 里注释掉的
`RESEARCH_PROTOCOL = dict(SCHEDULE="configurable", FORMAL_READY=True)`，commit，再
`docker/l40/create_bundle.sh`，按“已有容器的受控替换”换到新 release，重跑 `check` 与
runtime gate，然后启动 formal。`FORMAL_READY=False` 时
`research.run_contract --mode formal` 必然拒绝，这是有意的安全锁。

若最终决定改用 CPP，把 `MODEL.POSE_NET.XYZ_RENDERER` 固定为 `cpp`，重新
preflight → CUDA+CPP smoke → commit → bundle → release；160 epoch 中不得再切回 EGL，
并把 renderer 选择写进 RECORD。

### Renderer 配置边界

- `MODEL.POSE_NET.GEO_HEAD.TRAIN_SUPERVISION` 和
  `MODEL.POSE_NET.XYZ_RENDERER` 只控制训练阶段 rendered GT geometry。允许值为
  `cpp`、`egl`；禁用统一写 `None`，解析器也接受 `False`、`none`、`false`、
  `disabled`。geometry head 冻结时必须同时关闭 supervision 和训练 renderer。
- `VAL.RENDERER_TYPE` 只控制 BOP evaluation，允许 `cpp` 或 `egl`。关闭训练 renderer
  不得清空这个字段。当前 research screening 与独立 evaluation 统一使用 `cpp`。
- 本地单测/preflight 不创建 renderer；pose-head-only real smoke 使用真实 LM-PBR batch，
  但不生成无梯度用途的 rendered GT geometry。服务器 lab0/lab1 从镜像提供
  `/opt/bop_renderer/build`，launcher 统一注入并验证 `BOP_RENDERER_PATH`。
- 需要端到端训练 geometry head 的实验必须显式设置
  `TRAIN_SUPERVISION=True`，并从 `cpp`/`egl` 中明确选择训练 renderer。

训练 loop 在每个预定 evaluation 点先保存 epoch checkpoint，再运行 BOP evaluation；
因此 evaluation 异常不会抹掉已经完成的 epoch 状态。

新建正式研究训练配置默认显式设置 `SOLVER.AMP.ENABLED=True`；历史实验配置保持原值。
启用 AMP 的实验必须让 CUDA smoke/profile 使用与 formal 相同的 autocast 和 GradScaler，
不能用 FP32 诊断代替 AMP 证据。因算子或数值问题关闭 AMP 时，在对应 RECORD 中记录
smoke 输出与例外范围。

正式流程为 bundle/只读 release checkout → `create IMAGE_REF` → `run`/`eval`。
`create` 会核对 image revision 与当前 native/环境输入，并自动补齐 Git ignored
native artifacts，无需手工复制 `.so`。output、home、XDG runtime cache 与 dataset
cache 使用外部可写挂载；GPU 可共享，默认要求至少 `12000 MiB` 空闲显存，可用
`GDRN_MIN_FREE_GPU_MB` 显式覆盖。

独立评估使用：

```bash
docker/l40/experiment.sh lab0 eval EXP-... configs/.../eval.py \
  /data/labs/lab0/docker_data/chx/outputs/experiments/EXP-.../RUN-.../model_epoch_040.pth
```

`run`/`eval` 自动建立唯一输出目录，后台执行并写 `console.log`、`exit_code`。启动后
不要修改服务器 checkout 或镜像。完成后把关键指标、checkpoint 文件名/epoch、
run ID、源码 commit 和结论写入对应 RECORD；不要提交 checkpoint 或完整日志。
smoke/profile 启动后至少间隔 10–15 分钟再检查；formal 首次在 15–30 分钟确认数值、
显存和吞吐，之后按约 6 小时或固定 checkpoint 节点检查，避免持续 `tail -f` 和频繁轮询。
