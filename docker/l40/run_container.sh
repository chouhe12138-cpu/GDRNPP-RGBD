#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 GPU_ID" >&2
    exit 2
fi

gpu_id="$1"
case "${gpu_id}" in
    0|1|2|3) ;;
    *) echo "GPU_ID must be 0, 1, 2, or 3" >&2; exit 2 ;;
esac

chx_root="/data/labs/lab1/docker_data/chx"
container="lab1_chx_stage3c1"
image="${IMAGE:-gdrnpp-stage3c1:torch220-cu121-sm89-v1}"

test -d "${chx_root}/datasets/BOP_DATASETS/lm/train_pbr"
test -d "${chx_root}/datasets/BOP_DATASETS/lmo/test"
test -d "${chx_root}/datasets/VOC/VOC2012/JPEGImages"
test -f "${chx_root}/weights/lmo_pbr/model_final_wo_optim.pth"
test -z "$(/usr/bin/docker ps -aq --filter "name=^/${container}$")"

/usr/bin/docker run --gpus "device=${gpu_id}" --name "${container}" --hostname "${container}" --shm-size=16g -it \
    --mount "type=bind,src=${chx_root}/datasets/BOP_DATASETS,dst=/workspace/gdrnpp/datasets/BOP_DATASETS,readonly" \
    --mount "type=bind,src=${chx_root}/datasets/VOC,dst=/workspace/gdrnpp/datasets/VOCdevkit,readonly" \
    --mount "type=bind,src=${chx_root}/weights,dst=/workspace/gdrnpp/pretrained_models,readonly" \
    --mount "type=bind,src=${chx_root}/outputs,dst=/workspace/gdrnpp/output" \
    --mount "type=bind,src=${chx_root}/cache,dst=/home/gdrn/.cache" \
    --mount "type=bind,src=${chx_root}/logs,dst=/workspace/logs" \
    "${image}" bash
