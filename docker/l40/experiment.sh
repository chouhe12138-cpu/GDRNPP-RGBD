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

owner="chx"
docker_bin="/usr/bin/docker"

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

build_train_command() {
    local config="$1" run_container="$2"
    printf 'core/gdrn_modeling/train_gdrn.sh %q 0 --opts OUTPUT_DIR %q' \
        "${config}" "${run_container}"
}

build_eval_command() {
    local config="$1" checkpoint_container="$2" run_container="$3"
    printf 'python core/gdrn_modeling/main_gdrn.py --config-file %q --num-gpus 1 --eval-only --opts MODEL.WEIGHTS %q OUTPUT_DIR %q' \
        "${config}" "${checkpoint_container}" "${run_container}"
}

require_mount() {
    local expected_source="$1" expected_destination="$2" expected_rw="$3"
    local actual_source="" actual_destination="" actual_rw=""
    while IFS=$'\t' read -r actual_source actual_destination actual_rw; do
        if [[ "${actual_destination}" == "${expected_destination}" ]]; then
            [[ "${actual_source}" == "$(realpath -e -- "${expected_source}")" ]] || \
                fail "mount ${expected_destination} has source ${actual_source}, expected ${expected_source}"
            [[ "${actual_rw}" == "${expected_rw}" ]] || \
                fail "mount ${expected_destination} rw=${actual_rw}, expected ${expected_rw}"
            return
        fi
    done < <("${docker_bin}" inspect "${container}" \
        --format '{{range .Mounts}}{{printf "%s\t%s\t%t\n" .Source .Destination .RW}}{{end}}')
    fail "missing required mount: ${expected_destination}"
}

verify_required_mounts() {
    require_mount "${repo_root}" /workspace/gdrnpp false
    require_mount "${root}/datasets/BOP_DATASETS" /workspace/gdrnpp/datasets/BOP_DATASETS false
    require_mount "${root}/datasets/VOC" /workspace/gdrnpp/datasets/VOCdevkit false
    require_mount "${root}/weights" /workspace/gdrnpp/pretrained_models false
    require_mount "${root}/outputs" /workspace/gdrnpp/output true
    require_mount "${root}/cache" /home/gdrn/.cache true
    require_mount "${root}/home" /home/gdrn true
}

require_writable_output() {
    "${docker_bin}" exec "${container}" test -w /workspace/gdrnpp/output || \
        fail "container output mount is not writable"
}

require_cuda() {
    "${docker_bin}" exec "${container}" python -c \
        'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.device_count() == 1, torch.cuda.device_count()' || \
        fail "container CUDA gate failed"
}

verify_environment() {
    "${docker_bin}" exec "${container}" /usr/local/bin/verify-gdrn-environment || \
        fail "verify-gdrn-environment failed"
}

verify_native() {
    "${docker_bin}" exec "${container}" /usr/local/bin/verify-gdrn-native || \
        fail "verify-gdrn-native failed"
}

load_runtime_config() {
    local config="$1"
    "${docker_bin}" exec -w /workspace/gdrnpp -e PYTHONPATH=/workspace/gdrnpp \
        "${container}" python -c \
        'import sys; from mmcv import Config; Config.fromfile(sys.argv[1])' \
        "/workspace/gdrnpp/${config}" || fail "container config load failed: ${config}"
}

runtime_gate() {
    local config="$1"
    require_owned_container
    verify_required_mounts
    require_writable_output
    require_cuda
    verify_environment
    verify_native
    load_runtime_config "${config}"
    echo "RUNTIME_GATE PASS container=${container} config=${config}"
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
    runtime_gate "${config}"
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
        command="$(build_eval_command "${config}" "${checkpoint_container}" "${run_container}")"
    else
        command="$(build_train_command "${config}" "${run_container}")"
    fi
    "${docker_bin}" exec -d "${container}" bash -lc \
        "cd /workspace/gdrnpp && { ${command}; code=\$?; printf '%s\\n' \"\${code}\" > ${run_container}/exit_code; exit \"\${code}\"; } >> ${run_container}/console.log 2>&1"
    echo "RUN_LAUNCHED experiment=${experiment_id} run=${run_id} output=${run_host}"
}

main() {
    [[ $# -ge 2 ]] || usage
    machine="$1"
    action="$2"
    shift 2

    case "${machine}" in
        lab0) gpu_id=0 ;;
        lab1) gpu_id=1 ;;
        *) usage ;;
    esac

    root="/data/labs/${machine}/docker_data/${owner}"
    container="gdrnpp_${owner}_${machine}"
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd -- "${script_dir}/../.." && pwd)"
    output_root="${root}/outputs/experiments"

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
        mkdir -p \
            "${repo_root}/datasets/BOP_DATASETS" \
            "${repo_root}/datasets/VOCdevkit" \
            "${repo_root}/pretrained_models" \
            "${repo_root}/output" \
            "${root}/outputs" \
            "${root}/cache" \
            "${root}/home/.cache"
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
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
