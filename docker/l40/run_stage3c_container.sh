#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 B|C2" >&2
    exit 2
fi

role="$1"
case "${role}" in
    B)
        gpu_id="${GPU_ID:-0}"
        chx_root="${CHX_ROOT:-/data/labs/lab0/docker_data/chx}"
        container="${CONTAINER_NAME:-lab0_chx}"
        baseline_root="${BASELINE_ROOT:-${chx_root}/baselines/official_gt}"
        ;;
    C2)
        gpu_id="${GPU_ID:-1}"
        chx_root="${CHX_ROOT:-/data/labs/lab1/docker_data/chx}"
        container="${CONTAINER_NAME:-lab1_chx}"
        baseline_root="${BASELINE_ROOT:-${chx_root}/outputs/EXP-20260731-006/official_gt}"
        ;;
    *)
        echo "usage: $0 B|C2" >&2
        exit 2
        ;;
esac

asset_root="${ASSET_ROOT:-${chx_root}}"
image="${IMAGE:-gdrnpp-stage3bc2:torch220-cu121-sm89-v2}"
docker_bin="${DOCKER_BIN:-/usr/bin/docker}"

for directory in outputs logs cache audit; do
    mkdir -p "${chx_root}/${directory}"
done

test -d "${asset_root}/datasets/BOP_DATASETS/lm/train_pbr"
test -d "${asset_root}/datasets/BOP_DATASETS/lmo/test"
test -d "${asset_root}/datasets/VOC/VOC2012/JPEGImages"
test -f "${asset_root}/weights/lmo_pbr/model_final_wo_optim.pth"
test -d "${baseline_root}"
test -z "$("${docker_bin}" ps -aq --filter "name=^/${container}$")"
"${docker_bin}" image inspect "${image}" >/dev/null
image_id="$("${docker_bin}" image inspect "${image}" --format '{{.Id}}')"
gpu_uuid="$(nvidia-smi -i "${gpu_id}" --query-gpu=uuid --format=csv,noheader)"

"${docker_bin}" run -d \
    --gpus "device=${gpu_id}" \
    --user "$(id -u):$(id -g)" \
    --name "${container}" \
    --hostname "${container}" \
    --shm-size=16g \
    --label "gdrnpp.stage3c.role=${role}" \
    --env "HOME=/home/gdrn" \
    --env "XDG_CACHE_HOME=/home/gdrn/.cache" \
    --env "MPLCONFIGDIR=/home/gdrn/.cache/matplotlib" \
    --env "GDRN_IMAGE_ID=${image_id}" \
    --env "GDRN_PHYSICAL_GPU=${gpu_id}" \
    --env "GDRN_GPU_UUID=${gpu_uuid}" \
    --env "GDRN_STAGE3C_ROLE=${role}" \
    --mount "type=bind,src=${asset_root}/datasets/BOP_DATASETS,dst=/workspace/gdrnpp/datasets/BOP_DATASETS,readonly" \
    --mount "type=bind,src=${asset_root}/datasets/VOC,dst=/workspace/gdrnpp/datasets/VOCdevkit,readonly" \
    --mount "type=bind,src=${asset_root}/weights,dst=/workspace/gdrnpp/pretrained_models,readonly" \
    --mount "type=bind,src=${baseline_root},dst=/workspace/baselines/stage3c/official_gt,readonly" \
    --mount "type=bind,src=${chx_root}/outputs,dst=/workspace/gdrnpp/output" \
    --mount "type=bind,src=${chx_root}/cache,dst=/home/gdrn/.cache" \
    --mount "type=bind,src=${chx_root}/logs,dst=/workspace/logs" \
    "${image}" sleep infinity

echo "CONTAINER_CREATED=${container}"
echo "PHYSICAL_GPU=${gpu_id}"
echo "GPU_UUID=${gpu_uuid}"
echo "IMAGE_ID=${image_id}"
echo "ASSET_ROOT=${asset_root}"
echo "BASELINE_ROOT=${baseline_root}"
echo "WRITE_ROOT=${chx_root}"
