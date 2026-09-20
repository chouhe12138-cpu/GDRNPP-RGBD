# EXP025 当前运行手册

本手册只描述从当前状态继续工作所需的操作。EXP025 的 formal 前 batch48/EGL gate 已完成，
canonical 结果见对应 [RECORD](experiments/EXP-20260920-025-hierarchical-cad-attention/RECORD.md)。
本手册覆盖同一 source release 的两条 LM-O formal arm；LM13 服务器 profile 尚未开放。

所有服务器代码块都可在 lab0、lab1 原样执行，不含需要手改的机器编号、SHA、release、镜像、
旧容器名或 run ID。机械对象无法唯一确定时命令会列出候选并停止。

## 1. 本地验证与 bundle

```bash
(
set -Eeuo pipefail

cd /home/wsluser/GDRNPP-RGBD
source /home/wsluser/miniconda3/etc/profile.d/conda.sh
conda activate pytorch22
export GDRN_CONVNEXT_BASE_WEIGHTS=/home/wsluser/.cache/torch/hub/checkpoints/convnext_base_1k_224_ema.pth
frozen_config=configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_official_frozen.py
full_config=configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_imagenet_full.py

PYTHONPATH="$PWD" pytest -q research/exp025/tests research/tests research/cad_hierarchy/tests
python -m research.exp025.preflight --config "${frozen_config}"
python -m research.exp025.preflight --config "${full_config}"
git diff --check
docker/l40/create_bundle.sh
)
```

`create_bundle.sh` 会打印 bundle 的完整本地路径。把该文件分别放入两台服务器当前账户的
`/data/labs/<登录账户>/docker_data/chx/transfer/`；服务器不连接远端 Git。用户只需传文件，
不用抄写文件名中的 SHA。

## 2. 自动验证 bundle 并建立 release

以下代码块自动寻找 transfer 中唯一一个尚未建立 release 的 GDRNPP bundle，从 bundle 自身
解析 commit。若没有新 bundle，则核对并复用服务器记录的当前 release；若有多个候选则停止。

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
case "${machine}" in
    lab0|lab1) ;;
    *) printf 'FAIL: unsupported account: %s\n' "${machine}" >&2; exit 1 ;;
esac

root="/data/labs/${machine}/docker_data/chx"
marker="${root}/transfer/GDRNPP-RGBD-current-release.txt"
mapfile -t bundles < <(
    find "${root}/transfer" -maxdepth 1 -type f -name 'GDRNPP-RGBD-*.bundle' -print | sort
)

candidates=()
for bundle in "${bundles[@]}"; do
    full_sha="$(git bundle list-heads "${bundle}" | awk '$2 ~ /^refs\/heads\// {print $1}')"
    [[ "${full_sha}" =~ ^[0-9a-f]{40}$ ]] || {
        printf 'FAIL: bundle must expose exactly one branch head: %s\n' "${bundle}" >&2
        exit 1
    }
    short_sha="${full_sha:0:7}"
    release="${root}/releases/GDRNPP-RGBD-${short_sha}"
    [[ -e "${release}" ]] || candidates+=("${bundle}")
done

if ((${#candidates[@]} == 0)); then
    if ! test -s "${marker}"; then
        printf 'FAIL: no new bundle and no current-release marker\n' >&2
        printf 'TRANSFER_BUNDLE %s\n' "${bundles[@]}" >&2
        exit 1
    fi
    release="$(<"${marker}")"
    [[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
    test -d "${release}/.git"
    test -z "$(git -C "${release}" status --short)"
    printf 'RELEASE_ALREADY_READY path=%s commit=%s\n' "${release}" "$(git -C "${release}" rev-parse HEAD)"
    exit 0
fi

if ((${#candidates[@]} != 1)); then
    printf 'FAIL: expected one new bundle, found %d\n' "${#candidates[@]}" >&2
    printf 'CANDIDATE %s\n' "${candidates[@]}" >&2
    exit 1
fi

bundle="${candidates[0]}"
full_sha="$(git bundle list-heads "${bundle}" | awk '$2 ~ /^refs\/heads\// {print $1}')"
short_sha="${full_sha:0:7}"
release="${root}/releases/GDRNPP-RGBD-${short_sha}"
test ! -e "${release}"

verify_repo="$(mktemp -d "/tmp/gdrnpp-bundle-${short_sha}.XXXXXX")"
trap 'test -n "${verify_repo:-}" && rm -rf -- "${verify_repo}"' EXIT
git -c init.defaultBranch=main -C "${verify_repo}" init --bare --quiet
git -C "${verify_repo}" bundle verify "${bundle}"

git clone --no-checkout "${bundle}" "${release}"
git -C "${release}" checkout --detach "${full_sha}"
test "$(git -C "${release}" rev-parse HEAD)" = "${full_sha}"
test -z "$(git -C "${release}" status --short)"
printf '%s\n' "${release}" > "${marker}"
printf 'RELEASE_READY path=%s commit=%s\n' "${release}" "${full_sha}"
)
```

## 3. 镜像策略

普通 Python、config 和文档变化复用稳定镜像。只有 Dockerfile、locked requirements、vendor、
C++/CUDA 或 ABI 输入变化才重建。当前文档整理不需要重建镜像。

若后续只读预检明确报告 native/environment 不兼容，再在对应服务器原样执行：

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
case "${machine}" in
    lab0|lab1) ;;
    *) printf 'FAIL: unsupported account: %s\n' "${machine}" >&2; exit 1 ;;
esac

root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
test -d "${release}/.git"
test -z "$(git -C "${release}" status --short)"

cd "${release}"
docker/l40/build_image.sh
image="gdrnpp-research:torch220-cu121-sm89-$(git rev-parse --short=12 HEAD)"
/usr/bin/docker image inspect "${image}" >/dev/null
printf '%s\n' "${image}" > "${root}/transfer/GDRNPP-RGBD-current-image.txt"
printf 'IMAGE_READY image=%s\n' "${image}"
)
```

## 4. 容器替换前只读预检

先运行本节，只读解析当前 release、受管容器、挂载和镜像。不要把输出中的名称抄到下一段；
下一段会重新解析和校验。若当前容器不存在，则必须已有上节生成的 image marker。

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
case "${machine}" in
    lab0) gpu=0 ;;
    lab1) gpu=1 ;;
    *) printf 'FAIL: unsupported account: %s\n' "${machine}" >&2; exit 1 ;;
esac

root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
container="gdrnpp_chx_${machine}"
image_marker="${root}/transfer/GDRNPP-RGBD-current-image.txt"
native_inputs=(
    docker/l40/Dockerfile
    docker/l40/requirements.lock
    docker/l40/build_native.sh
    docker/l40/verify_environment.py
    docker/l40/verify_native.sh
    docker/l40/vendor
    core/csrc
    lib/egl_renderer
)

test -d "${release}/.git"
test -z "$(git -C "${release}" status --short)"
/usr/bin/docker info >/dev/null
nvidia-smi -i "${gpu}"

if /usr/bin/docker container inspect "${container}" >/dev/null 2>&1; then
    test "$(/usr/bin/docker inspect "${container}" --format '{{index .Config.Labels "gdrnpp.project"}}')" = GDRNPP-RGBD
    test "$(/usr/bin/docker inspect "${container}" --format '{{index .Config.Labels "gdrnpp.machine"}}')" = "${machine}"
    mounted_repo="$(/usr/bin/docker inspect "${container}" --format '{{range .Mounts}}{{if eq .Destination "/workspace/gdrnpp"}}{{.Source}}{{end}}{{end}}')"
    [[ "${mounted_repo}" == "${root}/releases/GDRNPP-RGBD-"* ]]
    active="$(/usr/bin/docker top "${container}" -eo pid,args | awk 'NR>1 {$1=""; sub(/^[[:space:]]+/,""); if ($0!="sleep infinity") print}')"
    test -z "${active}"
    image="$(/usr/bin/docker inspect "${container}" --format '{{.Config.Image}}')"
else
    test -s "${image_marker}"
    image="$(<"${image_marker}")"
    mounted_repo=MISSING
fi

if test -s "${image_marker}"; then
    image="$(<"${image_marker}")"
fi
/usr/bin/docker image inspect "${image}" >/dev/null
revision="$(/usr/bin/docker image inspect "${image}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
git -C "${release}" cat-file -e "${revision}^{commit}"
git -C "${release}" merge-base --is-ancestor "${revision}" HEAD
changed="$(git -C "${release}" diff --name-only "${revision}..HEAD" -- "${native_inputs[@]}")"
test -z "${changed}"

printf 'REPLACEMENT_PRECHECK PASS\n'
printf 'machine=%s gpu=%s\n' "${machine}" "${gpu}"
printf 'release=%s\n' "${release}"
printf 'container=%s\n' "${container}"
printf 'current_mount=%s\n' "${mounted_repo}"
printf 'image=%s\n' "${image}"
)
```

## 5. 受控替换项目容器

这是破坏性操作。只有用户核对上节输出并明确同意后才执行。代码块自动重新解析目标；如果容器
已经挂载目标 release，则只运行检查，不重复删除。

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
case "${machine}" in
    lab0|lab1) ;;
    *) printf 'FAIL: unsupported account: %s\n' "${machine}" >&2; exit 1 ;;
esac

root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
container="gdrnpp_chx_${machine}"
image_marker="${root}/transfer/GDRNPP-RGBD-current-image.txt"
native_inputs=(
    docker/l40/Dockerfile
    docker/l40/requirements.lock
    docker/l40/build_native.sh
    docker/l40/verify_environment.py
    docker/l40/verify_native.sh
    docker/l40/vendor
    core/csrc
    lib/egl_renderer
)

test -d "${release}/.git"
test -z "$(git -C "${release}" status --short)"

if /usr/bin/docker container inspect "${container}" >/dev/null 2>&1; then
    test "$(/usr/bin/docker inspect "${container}" --format '{{index .Config.Labels "gdrnpp.project"}}')" = GDRNPP-RGBD
    test "$(/usr/bin/docker inspect "${container}" --format '{{index .Config.Labels "gdrnpp.machine"}}')" = "${machine}"
    mounted_repo="$(/usr/bin/docker inspect "${container}" --format '{{range .Mounts}}{{if eq .Destination "/workspace/gdrnpp"}}{{.Source}}{{end}}{{end}}')"
    [[ "${mounted_repo}" == "${root}/releases/GDRNPP-RGBD-"* ]]
    active="$(/usr/bin/docker top "${container}" -eo pid,args | awk 'NR>1 {$1=""; sub(/^[[:space:]]+/,""); if ($0!="sleep infinity") print}')"
    test -z "${active}"
    image="$(/usr/bin/docker inspect "${container}" --format '{{.Config.Image}}')"
else
    mounted_repo=MISSING
    test -s "${image_marker}"
    image="$(<"${image_marker}")"
fi

if test -s "${image_marker}"; then
    image="$(<"${image_marker}")"
fi
/usr/bin/docker image inspect "${image}" >/dev/null
revision="$(/usr/bin/docker image inspect "${image}" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
git -C "${release}" cat-file -e "${revision}^{commit}"
git -C "${release}" merge-base --is-ancestor "${revision}" HEAD
changed="$(git -C "${release}" diff --name-only "${revision}..HEAD" -- "${native_inputs[@]}")"
test -z "${changed}"

if test "${mounted_repo}" = "${release}"; then
    cd "${release}"
    docker/l40/experiment.sh "${machine}" check
    docker/l40/experiment.sh "${machine}" status
    exit 0
fi

if test "${mounted_repo}" != MISSING; then
    /usr/bin/docker stop "${container}"
    /usr/bin/docker rm "${container}"
fi

cd "${release}"
docker/l40/experiment.sh "${machine}" create "${image}"
docker/l40/experiment.sh "${machine}" check
docker/l40/experiment.sh "${machine}" status
)
```

## 6. 启动 formal

在两台服务器分别原样执行同一个代码块。登录账户决定 arm；launcher 仍会再次校验映射并自动
生成唯一 run ID。

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
case "${machine}" in
    lab0) config=configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_official_frozen.py ;;
    lab1) config=configs/gdrn/lmo_pbr/research/exp025_hierarchical_cad_attention/train_imagenet_full.py ;;
    *) printf 'FAIL: unsupported account: %s\n' "${machine}" >&2; exit 1 ;;
esac

root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
experiment=EXP-20260920-025-hierarchical-cad-attention

test -d "${release}/.git"
test -z "$(git -C "${release}" status --short)"
cd "${release}"
docker/l40/experiment.sh "${machine}" run "${experiment}" "${config}" formal
)
```

启动后 15–30 分钟首次检查数值、显存和吞吐；之后约每 6 小时或固定 checkpoint 节点检查。
正式训练期间不 pull、不修改 release、不替换镜像。

## 7. 状态与最新日志

状态检查可在任一服务器原样执行：

```bash
(
set -Eeuo pipefail
machine="$(id -un)"
case "${machine}" in
    lab0|lab1) ;;
    *) printf 'FAIL: unsupported account: %s\n' "${machine}" >&2; exit 1 ;;
esac
root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
cd "${release}"
docker/l40/experiment.sh "${machine}" status
)
```

查看该服务器最新 EXP025 run 的日志，无需记 run ID：

```bash
(
set -Eeuo pipefail
machine="$(id -un)"
case "${machine}" in
    lab0|lab1) ;;
    *) printf 'FAIL: unsupported account: %s\n' "${machine}" >&2; exit 1 ;;
esac
root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
experiment=EXP-20260920-025-hierarchical-cad-attention
run_root="${root}/outputs/experiments/${experiment}"
mapfile -t runs < <(find "${run_root}" -mindepth 1 -maxdepth 1 -type d -printf '%T@\t%f\n' | sort -nr)
test "${#runs[@]}" -ge 1
latest="${runs[0]#*$'\t'}"
cd "${release}"
docker/l40/experiment.sh "${machine}" logs "${experiment}/${latest}"
)
```

若需要手工评价某个 checkpoint，checkpoint 是科学目标而非机械标识；Agent 应先根据固定评价点
和 RECORD 选定目标，再给用户一段已经填好精确路径的完整命令，不能让用户自行回忆或拼接。

## 8. 完成后的记录

每个 E5–E40 点记录聚合指标；完成后记录 run ID、source commit、image revision、checkpoint
文件名与 epoch、exit code、全部预定指标和结论。完整日志、checkpoint 与 BOP 输出继续外置，
Git 只保存 RECORD 和直接支撑结论的紧凑 evidence。
