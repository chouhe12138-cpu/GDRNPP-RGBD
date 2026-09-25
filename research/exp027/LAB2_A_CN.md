# EXP027-A：lab2 启动步骤

本流程只用于 `lab2` / 物理 GPU 2，配置为
`configs/gdrn/lmo_pbr/research/exp027_multiscale_cad/train_a_lab2.py`。lab0/lab1 的
EXP026 正式训练不受影响。EXP027-B 仍由原配置阻断。

资源位于当前账户的 `/data/labs/lab2/docker_data/chx/`：BOP、VOC、ConvNeXt 权重和
`cache/gdrnpp_datasets/exp026/RUN-20260921-ga-hfps-s20260919-a01/adaptive_512_l1.npz`。
本地配置的正式 AMP 初始 scale 为 4096；它只通过 batch4 本地验证，必须先通过
lab2 真实 batch48/EGL gate，才能启动 formal。

## 1. 本地提交后制作 bundle

按 `research/RUNBOOK_CN.md` 的本地规则完成提交。只在 clean tree 下运行：

```bash
(
set -Eeuo pipefail

cd /home/wsluser/GDRNPP-RGBD
docker/l40/create_bundle.sh
)
```

把输出的 `GDRNPP-RGBD-<short-sha>.bundle` 上传到 lab2 当前账户的
`/data/labs/lab2/docker_data/chx/transfer/`。不要上传本机 `.local/` 中的旧 bundle。

## 2. lab2 建立唯一 release

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
test "${machine}" = lab2
root="/data/labs/${machine}/docker_data/chx"
mapfile -t bundles < <(find "${root}/transfer" -maxdepth 1 -type f -name 'GDRNPP-RGBD-*.bundle' -print | sort)
printf 'BUNDLE_CANDIDATE %s\n' "${bundles[@]}"
test "${#bundles[@]}" -eq 1
bundle="${bundles[0]}"
mapfile -t heads < <(git bundle list-heads "${bundle}" | awk '$2 ~ /^refs\/heads\// {print $1}')
test "${#heads[@]}" -eq 1
commit="${heads[0]}"
[[ "${commit}" =~ ^[0-9a-f]{40}$ ]]
release="${root}/releases/GDRNPP-RGBD-${commit:0:7}"
test ! -e "${release}"

verify_repo="$(mktemp -d /tmp/gdrnpp-lab2-bundle.XXXXXX)"
trap 'rm -rf -- "${verify_repo}"' EXIT
git -c init.defaultBranch=main -C "${verify_repo}" init --bare --quiet
git -C "${verify_repo}" bundle verify "${bundle}"
git clone --no-checkout "${bundle}" "${release}"
git -C "${release}" checkout --detach "${commit}"
test -z "$(git -C "${release}" status --short)"
printf '%s\n' "${release}" > "${root}/transfer/GDRNPP-RGBD-current-release.txt"
printf 'RELEASE_READY %s %s\n' "${release}" "${commit}"
)
```

## 3. 选用已有稳定镜像并建立 lab2 项目容器

这一步不构建镜像；只有一个兼容的稳定镜像候选时才继续。若候选为零个或多个，输出
候选并停止，待确认实际镜像后再处理。

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
test "${machine}" = lab2
root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
test -z "$(git -C "${release}" status --short)"
mapfile -t images < <(/usr/bin/docker image ls --format '{{.Repository}}:{{.Tag}}' | awk '/^gdrnpp-research:torch220-cu121-sm89-/')
printf 'IMAGE_CANDIDATE %s\n' "${images[@]}"
test "${#images[@]}" -eq 1
image="${images[0]}"

nvidia-smi -i 2
/usr/bin/docker info >/dev/null
cd "${release}"
docker/l40/experiment.sh "${machine}" check
docker/l40/experiment.sh "${machine}" create "${image}"
docker/l40/experiment.sh "${machine}" check
)
```

`create` 会核对镜像源码兼容性、lab2 ownership 和挂载，只创建
`gdrnpp_chx_lab2`；若该容器已经存在，会停止而不替换。

## 4. 真实 batch48/EGL gate

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
test "${machine}" = lab2
root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
experiment=EXP-20260924-027-multiscale-cad-interaction
config=configs/gdrn/lmo_pbr/research/exp027_multiscale_cad/train_a_lab2.py

cd "${release}"
docker/l40/experiment.sh "${machine}" gate "${experiment}" "${config}" 4096
)
```

gate 异步运行。其 run 在
`/data/labs/lab2/docker_data/chx/outputs/experiments/EXP-20260924-027-multiscale-cad-interaction/`
下；`gate/report.json` 与 `exit_code` 给出结果。只有真实 batch48、8 步、0 skipped、
0 non-finite、checkpoint roundtrip PASS 才可进入 formal。formal launcher 会按 source
commit、配置、scale 和 hierarchy 身份寻找唯一通过的 gate，缺失或歧义会停止。

查看唯一 gate run 的简短输出：

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
test "${machine}" = lab2
root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
experiment=EXP-20260924-027-multiscale-cad-interaction
run_root="${root}/outputs/experiments/${experiment}"
mapfile -t runs < <(find "${run_root}" -mindepth 1 -maxdepth 1 -type d -name 'RUN-*-gate-s42-*' -print | sort)
printf 'GATE_CANDIDATE %s\n' "${runs[@]}"
test "${#runs[@]}" -eq 1

cd "${release}"
docker/l40/experiment.sh "${machine}" status
tail -n 40 "${runs[0]}/console.log"
if test -f "${runs[0]}/exit_code"; then
    cat "${runs[0]}/exit_code"
fi
if test -f "${runs[0]}/gate/report.json"; then
    cat "${runs[0]}/gate/report.json"
fi
)
```

## 5. 启动 formal

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
test "${machine}" = lab2
root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
experiment=EXP-20260924-027-multiscale-cad-interaction
config=configs/gdrn/lmo_pbr/research/exp027_multiscale_cad/train_a_lab2.py

cd "${release}"
docker/l40/experiment.sh "${machine}" run "${experiment}" "${config}" formal
)
```

正式训练期间不修改 lab2 release 或镜像。不能把本地 batch4 smoke 或未完成的 gate
当作正式训练依据。
