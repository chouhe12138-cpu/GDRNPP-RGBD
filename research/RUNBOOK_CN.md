# EXP025 运行手册

本手册当前只开放 LM-O。LM13 虽已有本地配置与验证产物，但 `exp025_lm13` 尚未加入 launcher；
不要为 LM13 创建服务器 run，待 LM-O 正式实验完成后再扩展本手册。

## 本地验证与 bundle

```bash
(
set -Eeuo pipefail

cd /home/wsluser/GDRNPP-RGBD
source /home/wsluser/miniconda3/etc/profile.d/conda.sh
conda activate pytorch22
export GDRN_CONVNEXT_BASE_WEIGHTS=/home/wsluser/.cache/torch/hub/checkpoints/convnext_base_1k_224_ema.pth

PYTHONPATH="$PWD" pytest -q research/exp025/tests research/tests research/cad_hierarchy/tests
python -m research.exp025.preflight \
  --config configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_official_frozen.py
python -m research.exp025.preflight \
  --config configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_imagenet_full.py
git diff --check
docker/l40/create_bundle.sh
)
```

bundle 位于 `.local/release/GDRNPP-RGBD-<short-sha>.bundle`。传到两台服务器各自的
`/data/labs/<machine>/docker_data/chx/transfer/`。服务器不连接远端 Git。

## 建立只读 release

以下代码块在 lab0/lab1 分别执行，只修改 `machine`、`short_sha` 和 `full_sha`：

```bash
(
set -Eeuo pipefail

machine=lab0
short_sha=REPLACE_SHORT_SHA
full_sha=REPLACE_FULL_SHA
root="/data/labs/${machine}/docker_data/chx"
bundle="${root}/transfer/GDRNPP-RGBD-${short_sha}.bundle"
release="${root}/releases/GDRNPP-RGBD-${short_sha}"

test "$(id -un)" = "${machine}"
test -f "${bundle}"
test ! -e "${release}"

verify_repo="$(mktemp -d "/tmp/gdrnpp-bundle-${short_sha}.XXXXXX")"
git -c init.defaultBranch=main -C "${verify_repo}" init --bare --quiet
git -C "${verify_repo}" bundle verify "${bundle}"

git clone --no-checkout "${bundle}" "${release}"
git -C "${release}" checkout --detach "${full_sha}"
test "$(git -C "${release}" rev-parse HEAD)" = "${full_sha}"
test -z "$(git -C "${release}" status --short)"
git -C "${release}" status --short --branch
)
```

## 安装 EXP025 hierarchy

把本地 `.local/dataset_cache/exp025/consistent_v3.npz` 传到两台服务器的 transfer 目录，
然后分别执行：

```bash
(
set -Eeuo pipefail

machine=lab0
root="/data/labs/${machine}/docker_data/chx"
source_file="${root}/transfer/consistent_v3.npz"
target="${root}/cache/gdrnpp_datasets/exp025/consistent_v3.npz"
expected=02ce090949bc40b2732417fec23984f3f748431098c5f67c853839d10ff1a373

test "$(id -un)" = "${machine}"
test -r "${source_file}"
install -D -m 644 "${source_file}" "${target}"
test "$(sha256sum "${target}" | awk '{print $1}')" = "${expected}"
)
```

## 受控替换项目容器

本轮更新了 Dockerfile 的构建期测试集合与镜像环境验证入口，因此必须从新 release 重建
项目镜像。依赖、C++/CUDA 和 ABI 没有变化；重建用于让镜像 revision 与当前验证契约一致。

```bash
(
set -Eeuo pipefail

machine=lab0
short_sha=REPLACE_SHORT_SHA
release="/data/labs/${machine}/docker_data/chx/releases/GDRNPP-RGBD-${short_sha}"

test "$(id -un)" = "${machine}"
test -z "$(git -C "${release}" status --short)"
cd "${release}"
docker/l40/build_image.sh
)
```

记录 `build_image.sh` 最后输出的 `image=...`，在下面填写为 `image_ref`。lab1 使用相同
commit 构建；两台镜像都应记录同一个 source revision。

先运行 `check` 和 `docker inspect`，确认精确容器标签、旧 repo mount，且容器内除
`sleep infinity` 外无进程。只有满足这些条件才停止并删除该项目容器，再从新 release
执行 `create`。不要操作其他容器。

```bash
(
set -Eeuo pipefail

machine=lab0
short_sha=REPLACE_SHORT_SHA
expected_old_repo=REPLACE_EXACT_OLD_RELEASE
image_ref=REPLACE_IMAGE_FROM_BUILD_OUTPUT
root="/data/labs/${machine}/docker_data/chx"
release="${root}/releases/GDRNPP-RGBD-${short_sha}"
container="gdrnpp_chx_${machine}"

test "$(id -un)" = "${machine}"
test -z "$(git -C "${release}" status --short)"
test "$(/usr/bin/docker inspect "${container}" --format '{{index .Config.Labels "gdrnpp.project"}}')" = GDRNPP-RGBD
test "$(/usr/bin/docker inspect "${container}" --format '{{index .Config.Labels "gdrnpp.machine"}}')" = "${machine}"
/usr/bin/docker image inspect "${image_ref}" >/dev/null

mounted_repo="$(/usr/bin/docker inspect "${container}" --format '{{range .Mounts}}{{if eq .Destination "/workspace/gdrnpp"}}{{.Source}}{{end}}{{end}}')"
test "${mounted_repo}" = "${expected_old_repo}"

active="$(/usr/bin/docker top "${container}" -eo pid,args | awk 'NR>1 {$1=""; sub(/^[[:space:]]+/,""); if ($0!="sleep infinity") print}')"
test -z "${active}"

/usr/bin/docker stop "${container}"
/usr/bin/docker rm "${container}"
cd "${release}"
docker/l40/experiment.sh "${machine}" create "${image_ref}"
docker/l40/experiment.sh "${machine}" check
docker/l40/experiment.sh "${machine}" status
)
```

lab1 使用同一代码块并把 `machine=lab1`；两台都必须绑定相同 source commit 的 release。

## 真实 batch48 gate

第一段 release 的两配置保持 `FORMAL_READY=False`。lab0：

```bash
(
set -Eeuo pipefail

release=/data/labs/lab0/docker_data/chx/releases/GDRNPP-RGBD-REPLACE_SHORT_SHA
experiment=EXP-20260920-025-hierarchical-cad-attention
config=configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_official_frozen.py

cd "${release}"
docker/l40/experiment.sh lab0 gate "${experiment}" "${config}" 65536
)
```

lab1：

```bash
(
set -Eeuo pipefail

release=/data/labs/lab1/docker_data/chx/releases/GDRNPP-RGBD-REPLACE_SHORT_SHA
experiment=EXP-20260920-025-hierarchical-cad-attention
config=configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_imagenet_full.py

cd "${release}"
docker/l40/experiment.sh lab1 gate "${experiment}" "${config}" 65536
)
```

若某臂仅因 AMP 非有限失败，使用新的唯一 run 依次重跑 32768、16384。选择两臂共同通过的
最高 scale。gate 必须报告 EGL、batch48、8 步、0 skipped step、有限 loss/gradient、正确的
backbone 冻结或更新、全部 Image-SA stage 更新、checkpoint roundtrip 和峰值显存。batch48
OOM 或 16384 仍失败时停止，不启动 formal，不自行改梯度累积。

gate 通过后把两份紧凑 `report.json` 同步回本地，记录进 EXP025 RECORD；随后在两个正式配置
中设置相同 `SOLVER.AMP.INIT_SCALE` 和 `FORMAL_READY=True`，提交、push、tag，再生成第二段
bundle/release并按上节替换容器。

## 正式训练

lab0：

```bash
(
set -Eeuo pipefail

release=/data/labs/lab0/docker_data/chx/releases/GDRNPP-RGBD-REPLACE_FORMAL_SHA
experiment=EXP-20260920-025-hierarchical-cad-attention
config=configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_official_frozen.py

cd "${release}"
docker/l40/experiment.sh lab0 run "${experiment}" "${config}" formal
)
```

lab1 使用 `train_imagenet_full.py` 和 `experiment.sh lab1`。启动后 15–30 分钟首次检查数值、
显存和吞吐，之后约每 6 小时或固定 checkpoint 节点检查。使用短命令：

```bash
docker/l40/experiment.sh lab0 status
docker/l40/experiment.sh lab0 logs EXP-20260920-025-hierarchical-cad-attention/RUN-...
```

正式训练期间不 pull、不修改 release、不替换镜像。每个 E5–E40 点记录聚合指标；完成后记录
run ID、source commit、checkpoint、exit code 和结论，不提交 checkpoint 或完整日志。
