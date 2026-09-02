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

本地统一使用下面的入口创建 bundle；脚本强制当前分支为 `main`、working tree（包括
untracked 文件）为空、目标 bundle 不存在，并在创建后执行完整验证：

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

if /usr/bin/docker exec "${container}" pgrep -f '[m]ain_gdrn.py' >/dev/null 2>&1; then
  echo "REFUSE: main_gdrn.py is active in ${container}" >&2
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
与 evaluation。每次 run 的根目录写入 `run_metadata.json`，保存完整 source commit、
image ID、image build revision、config、mode 与 run ID。

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
