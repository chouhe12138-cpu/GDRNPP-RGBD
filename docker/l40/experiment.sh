#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat >&2 <<'EOF'
usage:
  experiment.sh lab0|lab1 check
  experiment.sh lab0|lab1 create IMAGE_REF
  experiment.sh lab0|lab1 run EXPERIMENT_ID CONFIG smoke|formal
  experiment.sh lab0|lab1 eval EXPERIMENT_ID CONFIG CHECKPOINT
  experiment.sh lab0|lab1 status
  experiment.sh lab0|lab1 logs EXPERIMENT_ID/RUN_ID
EOF
    exit 2
}

[[ $# -ge 2 ]] || usage
machine="$1"
action="$2"
shift 2

case "${machine}" in
    lab0) gpu_id=0 ;;
    lab1) gpu_id=1 ;;
    *) usage ;;
esac

owner="chx"
root="/data/labs/${machine}/docker_data/${owner}"
container="gdrnpp_${owner}_${machine}"
docker_bin="/usr/bin/docker"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
output_root="${root}/outputs/experiments"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

check_host() {
    [[ "$(id -un)" == "${machine}" ]] || fail "run this profile as ${machine}"
    [[ -x "${docker_bin}" ]] || fail "missing ${docker_bin}"
    "${docker_bin}" info >/dev/null
    [[ -d "${root}/datasets/BOP_DATASETS/lm/train_pbr" ]] || fail "missing LM PBR data"
    [[ -d "${root}/datasets/BOP_DATASETS/lmo/test" ]] || fail "missing LM-O test data"
    [[ -d "${root}/datasets/VOC/VOC2012/JPEGImages" ]] || fail "missing VOC data"
    [[ -r "${root}/weights/lmo_pbr/model_final_wo_optim.pth" ]] || fail "missing official checkpoint"
    nvidia-smi -i "${gpu_id}" \
        --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader
}

container_exists() {
    "${docker_bin}" container inspect "${container}" >/dev/null 2>&1
}

require_owned_container() {
    container_exists || fail "container ${container} does not exist; run create first"
    local project_label machine_label
    project_label="$("${docker_bin}" inspect "${container}" --format '{{index .Config.Labels "gdrnpp.project"}}')"
    machine_label="$("${docker_bin}" inspect "${container}" --format '{{index .Config.Labels "gdrnpp.machine"}}')"
    [[ "${project_label}" == "GDRNPP-RGBD" && "${machine_label}" == "${machine}" ]] || \
        fail "container ${container} is not owned by this project/profile"
}

start_owned_container() {
    require_owned_container
    if [[ "$("${docker_bin}" inspect "${container}" --format '{{.State.Running}}')" != "true" ]]; then
        "${docker_bin}" start "${container}" >/dev/null
    fi
}

require_idle_gpu() {
    local active
    active="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
    [[ -z "${active}" ]] || fail "GPU ${gpu_id} already has active compute processes: ${active}"
}

require_idle_container() {
    if "${docker_bin}" exec "${container}" pgrep -f '[m]ain_gdrn.py' >/dev/null 2>&1; then
        fail "a GDRN training/evaluation process is already active in ${container}"
    fi
}

validate_experiment_id() {
    [[ "$1" =~ ^EXP-[A-Za-z0-9._-]+$ ]] || fail "invalid experiment ID: $1"
}

resolve_config() {
    local requested="$1" resolved
    [[ "${requested}" != /* ]] || fail "config must be relative to the repository"
    resolved="$(realpath -e -- "${repo_root}/${requested}")" || fail "config does not exist: ${requested}"
    [[ "${resolved}" == "${repo_root}/configs/"* ]] || fail "config must be under configs/"
    printf '%s\n' "${resolved#${repo_root}/}"
}

next_run_id() {
    local mode="$1" stamp attempt candidate
    stamp="$(date -u +%Y%m%d-%H%M%S)"
    for attempt in $(seq 1 99); do
        printf -v candidate 'RUN-%s-%s-s42-a%02d' "${stamp}" "${mode}" "${attempt}"
        [[ -e "${output_root}/${experiment_id}/${candidate}" ]] || {
            printf '%s\n' "${candidate}"
            return
        }
    done
    fail "no unused run ID remains for ${stamp}"
}

launch() {
    local mode="$1" config="$2" checkpoint_container="${3:-}" run_id run_host run_container commit image_ref command
    check_host
    start_owned_container
    require_idle_gpu
    require_idle_container
    run_id="$(next_run_id "${mode}")"
    run_host="${output_root}/${experiment_id}/${run_id}"
    run_container="/workspace/gdrnpp/output/experiments/${experiment_id}/${run_id}"
    mkdir -p "${run_host}"
    commit="$(git -C "${repo_root}" rev-parse --short=12 HEAD)"
    image_ref="$("${docker_bin}" inspect "${container}" --format '{{.Config.Image}}')"
    printf 'RUN_INFO experiment=%s run=%s mode=%s config=%s seed=42 commit=%s image=%s\n' \
        "${experiment_id}" "${run_id}" "${mode}" "${config}" "${commit}" "${image_ref}" \
        > "${run_host}/console.log"
    if [[ "${mode}" == "eval" ]]; then
        command="python core/gdrn_modeling/main_gdrn.py --config-file ${config} --num-gpus 1 --eval-only MODEL.WEIGHTS ${checkpoint_container} OUTPUT_DIR ${run_container}"
    else
        command="core/gdrn_modeling/train_gdrn.sh ${config} 0 OUTPUT_DIR ${run_container}"
    fi
    "${docker_bin}" exec -d "${container}" bash -lc \
        "cd /workspace/gdrnpp && { ${command}; code=\$?; printf '%s\\n' \"\${code}\" > ${run_container}/exit_code; exit \"\${code}\"; } >> ${run_container}/console.log 2>&1"
    echo "RUN_LAUNCHED experiment=${experiment_id} run=${run_id} output=${run_host}"
}

case "${action}" in
    check)
        [[ $# -eq 0 ]] || usage
        check_host
        if container_exists; then
            require_owned_container
            "${docker_bin}" inspect "${container}" --format 'container={{.Name}} state={{.State.Status}} image={{.Config.Image}}'
        else
            echo "container=${container} state=MISSING"
        fi
        nvidia-smi -i "${gpu_id}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
        ;;
    create)
        [[ $# -eq 1 ]] || usage
        image_ref="$1"
        check_host
        container_exists && fail "container name already exists: ${container}"
        "${docker_bin}" image inspect "${image_ref}" >/dev/null 2>&1 || fail "image not found: ${image_ref}"
        mkdir -p "${root}/outputs" "${root}/cache" "${root}/home"
        "${docker_bin}" run -d \
            --gpus "device=${gpu_id}" \
            --user "$(id -u):$(id -g)" \
            --name "${container}" --hostname "${container}" --shm-size=16g \
            --label "gdrnpp.project=GDRNPP-RGBD" --label "gdrnpp.machine=${machine}" \
            --env HOME=/home/gdrn --env CUDA_VISIBLE_DEVICES=0 \
            --env XDG_CACHE_HOME=/home/gdrn/.cache \
            --mount "type=bind,src=${repo_root},dst=/workspace/gdrnpp,readonly" \
            --mount "type=bind,src=${root}/datasets/BOP_DATASETS,dst=/workspace/gdrnpp/datasets/BOP_DATASETS,readonly" \
            --mount "type=bind,src=${root}/datasets/VOC,dst=/workspace/gdrnpp/datasets/VOCdevkit,readonly" \
            --mount "type=bind,src=${root}/weights,dst=/workspace/gdrnpp/pretrained_models,readonly" \
            --mount "type=bind,src=${root}/outputs,dst=/workspace/gdrnpp/output" \
            --mount "type=bind,src=${root}/cache,dst=/home/gdrn/.cache" \
            --mount "type=bind,src=${root}/home,dst=/home/gdrn" \
            "${image_ref}" sleep infinity >/dev/null
        echo "CONTAINER_CREATED name=${container} physical_gpu=${gpu_id} image=${image_ref}"
        ;;
    run)
        [[ $# -eq 3 ]] || usage
        experiment_id="$1"
        validate_experiment_id "${experiment_id}"
        config="$(resolve_config "$2")"
        case "$3" in smoke|formal) mode="$3" ;; *) usage ;; esac
        launch "${mode}" "${config}"
        ;;
    eval)
        [[ $# -eq 3 ]] || usage
        experiment_id="$1"
        validate_experiment_id "${experiment_id}"
        config="$(resolve_config "$2")"
        checkpoint_host="$(realpath -e -- "$3")" || fail "checkpoint does not exist: $3"
        case "${checkpoint_host}" in
            "${root}/weights/"*) checkpoint_container="/workspace/gdrnpp/pretrained_models/${checkpoint_host#${root}/weights/}" ;;
            "${root}/outputs/"*) checkpoint_container="/workspace/gdrnpp/output/${checkpoint_host#${root}/outputs/}" ;;
            *) fail "checkpoint must be under ${root}/weights or ${root}/outputs" ;;
        esac
        launch eval "${config}" "${checkpoint_container}"
        ;;
    status)
        [[ $# -eq 0 ]] || usage
        check_host
        require_owned_container
        "${docker_bin}" inspect "${container}" --format 'container={{.Name}} state={{.State.Status}} image={{.Config.Image}}'
        "${docker_bin}" top "${container}" -eo pid,args
        ;;
    logs)
        [[ $# -eq 1 ]] || usage
        requested="$(realpath -m -- "${output_root}/$1")"
        [[ "${requested}" == "${output_root}/"* ]] || fail "run directory escapes ${output_root}"
        [[ -f "${requested}/console.log" ]] || fail "missing console.log: ${requested}"
        tail -f "${requested}/console.log"
        ;;
    *) usage ;;
esac
