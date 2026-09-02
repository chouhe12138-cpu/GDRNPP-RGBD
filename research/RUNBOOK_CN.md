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

不要依赖当前工作目录解析 `transfer/...` 或 `releases/...`。`git bundle verify` 需要 Git
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

```bash
docker/l40/experiment.sh lab0 check
docker/l40/experiment.sh lab0 create gdrnpp-research:torch220-cu121-sm89-c0be1ade7ea9
docker/l40/experiment.sh lab0 run EXP-... configs/.../smoke.py smoke
docker/l40/experiment.sh lab0 run EXP-... configs/.../train.py formal
docker/l40/experiment.sh lab0 status
docker/l40/experiment.sh lab0 logs EXP-.../RUN-...
```

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
