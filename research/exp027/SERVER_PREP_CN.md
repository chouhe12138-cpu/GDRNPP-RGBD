# EXP027 原 lab0/lab1 准备草案（当前不要执行）

EXP027-A 当前改在 lab2 / GPU 2 准备独立 release，操作步骤见
[lab2 A 启动步骤](LAB2_A_CN.md)。以下为 2026-09-24 的旧 lab0/lab1 方案，
不适用于本次 lab2 启动。EXP027-B 仍保持阻断。

EXP026 正式训练期间不修改服务器 checkout、容器或镜像。当前 EXP027 配置的
`SERVER_RELEASE_ALLOWED=False`、`FORMAL_READY=False`，服务器 gate/formal 会 fail-closed。
待 EXP026 两臂结束、用户授权 EXP027 release 后，在本地把两项 flag 打开并提交；该同一提交
须用于 bundle、两台服务器 release、各自 gate 和 formal。两臂共同 AMP 初始 scale 候选 4096
仅通过本地 batch4，不能代替真实 batch48/EGL gate。无依赖或 native 改动，按
[RUNBOOK](../RUNBOOK_CN.md)复用稳定镜像。

## 先完成 release

在本地按 RUNBOOK 第 1 节创建 bundle；用户将其分别放入各服务器当前账户 `transfer/`。
在每台服务器依次使用 RUNBOOK 第 2、4、5 节建立只读 release、核对 GPU/受管容器和镜像，
仅在精确目标确认并获授权时执行第 5 节容器替换。下列块均独立解析当前 release 和本机映射。

## 每臂真实 batch48/EGL gate

两台服务器各执行一次。此块只在 release flag 已放行且当前项目容器挂载该 release 时有效；
launcher 会检查资源、hierarchy、ImageNet 权重和 clean tree，并生成唯一 gate run。

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
case "${machine}" in
    lab0) gpu=0; config=configs/gdrn/lmo_pbr/research/exp027_multiscale_cad/train_a_multiscale_fpn.py ;;
    lab1) gpu=1; config=configs/gdrn/lmo_pbr/research/exp027_multiscale_cad/train_b_cad_region_query.py ;;
    *) printf 'FAIL: unsupported account: %s\n' "${machine}" >&2; exit 1 ;;
esac

root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
test -z "$(git -C "${release}" status --short)"
commit="$(git -C "${release}" rev-parse HEAD)"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-${commit:0:7}" ]]
container="gdrnpp_chx_${machine}"
test "$(/usr/bin/docker inspect "${container}" --format '{{index .Config.Labels "gdrnpp.project"}}')" = GDRNPP-RGBD
test "$(/usr/bin/docker inspect "${container}" --format '{{index .Config.Labels "gdrnpp.machine"}}')" = "${machine}"
image="$(/usr/bin/docker inspect "${container}" --format '{{.Config.Image}}')"
/usr/bin/docker image inspect "${image}" >/dev/null
nvidia-smi -i "${gpu}"

experiment=EXP-20260924-027-multiscale-cad-interaction
cd "${release}"
docker/l40/experiment.sh "${machine}" check
docker/l40/experiment.sh "${machine}" gate "${experiment}" "${config}" 4096
)
```

gate 在容器中异步执行。用以下独立块检查唯一匹配本 release、config 和 scale 的 gate；
若零个或多个匹配则列出候选并停止。此检查不代表用户授权启动 formal。

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
case "${machine}" in
    lab0) gpu=0; config=configs/gdrn/lmo_pbr/research/exp027_multiscale_cad/train_a_multiscale_fpn.py ;;
    lab1) gpu=1; config=configs/gdrn/lmo_pbr/research/exp027_multiscale_cad/train_b_cad_region_query.py ;;
    *) printf 'FAIL: unsupported account: %s\n' "${machine}" >&2; exit 1 ;;
esac

root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
test -z "$(git -C "${release}" status --short)"
commit="$(git -C "${release}" rev-parse HEAD)"
container="gdrnpp_chx_${machine}"
test "$(/usr/bin/docker inspect "${container}" --format '{{index .Config.Labels "gdrnpp.project"}}')" = GDRNPP-RGBD
image="$(/usr/bin/docker inspect "${container}" --format '{{.Config.Image}}')"
/usr/bin/docker image inspect "${image}" >/dev/null
nvidia-smi -i "${gpu}"

experiment=EXP-20260924-027-multiscale-cad-interaction
run_root="${root}/outputs/experiments/${experiment}"
mapfile -t runs < <(find "${run_root}" -mindepth 1 -maxdepth 1 -type d -name 'RUN-*-gate-s42-*' -print | sort)
printf 'GATE_CANDIDATE %s\n' "${runs[@]}"
cd "${release}"
docker/l40/experiment.sh "${machine}" status
sha=7ac75daa756ed7ae59463614737ccc46cd746173452cd1943a4a024d0817c631
PYTHONPATH="${release}" python3 -m research.exp027.gate_evidence "${run_root}" "${commit}" "${config}" 4096 "${sha}"
)
```

两臂各自 PASS 后，用户若授权启动 formal，再按相同 release 从头启动各自 40 epoch。
`docker/l40/experiment.sh` 在 EXP027 formal 前会再次核对唯一、同 source commit/config/scale/
hierarchy SHA 的 batch48 gate PASS，缺失或歧义时阻断。启动后约 30 分钟检查一次数值与资源；
其后约每 12 小时或固定评价点检查，避免重复轮询。正式训练期间不变更 release 或镜像。

```bash
(
set -Eeuo pipefail

machine="$(id -un)"
case "${machine}" in
    lab0) gpu=0; config=configs/gdrn/lmo_pbr/research/exp027_multiscale_cad/train_a_multiscale_fpn.py ;;
    lab1) gpu=1; config=configs/gdrn/lmo_pbr/research/exp027_multiscale_cad/train_b_cad_region_query.py ;;
    *) printf 'FAIL: unsupported account: %s\n' "${machine}" >&2; exit 1 ;;
esac

root="/data/labs/${machine}/docker_data/chx"
release="$(<"${root}/transfer/GDRNPP-RGBD-current-release.txt")"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-"* ]]
test -z "$(git -C "${release}" status --short)"
commit="$(git -C "${release}" rev-parse HEAD)"
[[ "${release}" == "${root}/releases/GDRNPP-RGBD-${commit:0:7}" ]]
container="gdrnpp_chx_${machine}"
test "$(/usr/bin/docker inspect "${container}" --format '{{index .Config.Labels "gdrnpp.project"}}')" = GDRNPP-RGBD
test "$(/usr/bin/docker inspect "${container}" --format '{{index .Config.Labels "gdrnpp.machine"}}')" = "${machine}"
image="$(/usr/bin/docker inspect "${container}" --format '{{.Config.Image}}')"
/usr/bin/docker image inspect "${image}" >/dev/null
nvidia-smi -i "${gpu}"

experiment=EXP-20260924-027-multiscale-cad-interaction
cd "${release}"
docker/l40/experiment.sh "${machine}" run "${experiment}" "${config}" formal
)
```
