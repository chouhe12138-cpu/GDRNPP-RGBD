#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "usage: $0 lab0|lab1 EXP005|EXP009|EXP010|EXP012|EXP013A|EXP013B|EXP013C|EXP013D|EXP013E|EXP013F access|preserve|create|gate|smoke|audit48|launch|status|watch|finalize" >&2
    exit 2
}

[[ $# -eq 3 ]] || usage
machine="$1"
experiment="$2"
command="$3"

case "${machine}" in
    lab0)
        root="/data/labs/lab0/docker_data/chx"
        gpu_id=0
        container="lab0_chx"
        baseline_root="${root}/baselines/official_gt"
        ;;
    lab1)
        root="/data/labs/lab1/docker_data/chx"
        gpu_id=1
        container="lab1_chx"
        baseline_root="${root}/baselines/official_gt"
        ;;
    *) usage ;;
esac

case "${experiment}" in
    EXP005)
        experiment_id="EXP-20260731-005-pnp-only-control"
        config_root="configs/gdrn/lmo_pbr/research/exp005_pnp_control"
        isolation_role="B"
        ;;
    EXP009)
        experiment_id="EXP-20260809-009-cpm-head"
        config_root="configs/gdrn/lmo_pbr/research/exp009_cpm_head"
        isolation_role="CPM"
        ;;
    EXP010)
        experiment_id="EXP-20260816-010-cpm-official-lr-control"
        config_root="configs/gdrn/lmo_pbr/research/exp010_cpm_official_lr_control"
        isolation_role="CPM"
        ;;
    EXP012)
        experiment_id="EXP-20260817-012-hierarchical-correspondence-head"
        config_root="configs/gdrn/lmo_pbr/research/exp012_hierarchical_corr_head"
        isolation_role="PNP_REPLACEMENT"
        ;;
    EXP013A)
        experiment_id="EXP-20260822-013-a-xyz-residual-bypass"
        config_root="configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual"
        isolation_role="PNP_REPLACEMENT"
        exp013_variant="A"
        ;;
    EXP013B)
        experiment_id="EXP-20260822-013-b-geometry-attention-residual"
        config_root="configs/gdrn/lmo_pbr/research/exp013/b_geometry_attention"
        isolation_role="PNP_REPLACEMENT"
        exp013_variant="B"
        ;;
    EXP013C)
        experiment_id="EXP-20260822-013-c-rt-decoupled-fusion"
        config_root="configs/gdrn/lmo_pbr/research/exp013/c_rt_decoupled"
        isolation_role="PNP_REPLACEMENT"
        exp013_variant="C"
        ;;
    EXP013D)
        experiment_id="EXP-20260827-014-d-fulltrain-imagenet"
        config_root="configs/gdrn/lmo_pbr/research/exp013/d_fulltrain"
        isolation_role="FULL_TRAIN"
        ;;
    EXP013E)
        experiment_id="EXP-20260829-015-e-official-head-random"
        config_root="configs/gdrn/lmo_pbr/research/exp013/e_official_head_random"
        isolation_role="PNP_REPLACEMENT"
        exp013_variant="E"
        ;;
    EXP013F)
        experiment_id="EXP-20260829-016-f-glm-pose-l-screening"
        config_root="configs/gdrn/lmo_pbr/research/exp013/f_glm_pose_l"
        isolation_role="PNP_REPLACEMENT"
        exp013_variant="F"
        ;;
    *) usage ;;
esac

case "${experiment}:${machine}" in
    EXP013A:lab0|EXP013B:lab1|EXP013C:lab0) ;;
    EXP013D:lab1) ;;
    EXP013E:lab0) ;;
    EXP013F:lab1) ;;
    EXP013A:*|EXP013B:*|EXP013C:*|EXP013E:*|EXP013F:*)
        echo "FAIL: fixed server mapping is EXP013A=lab0, EXP013B=lab1, EXP013C=lab0, EXP013E=lab0, EXP013F=lab1" >&2
        exit 1
        ;;
esac

docker_bin="/usr/bin/docker"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"
commit="$(git -C "${repo_root}" rev-parse HEAD)"
environment_binding_host="${repo_root}/.local/environment_binding.json"
environment_binding_container="/workspace/gdrnpp/.local/environment_binding.json"
[[ -f "${environment_binding_host}" ]] || {
    echo "FAIL: missing ${environment_binding_host}; run docker/l40/prepare_release.sh first" >&2
    exit 1
}
image_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment"]["environment_image_id"])' "${environment_binding_host}")"
image_ref="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment"]["environment_image_ref"])' "${environment_binding_host}")"
environment_build_source="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment"]["environment_build_source_commit"])' "${environment_binding_host}")"
runtime_dir="${root}/runtime"
home_dir="${root}/home"
profile_host="${runtime_dir}/path_profile.json"
profile_container="/workspace/runtime/path_profile.json"
output_root_host="${root}/outputs/experiments"
output_root_container="/workspace/gdrnpp/output/experiments"
log_root="${root}/logs/managed/${experiment_id}"
official_container="/workspace/gdrnpp/pretrained_models/lmo_pbr/model_final_wo_optim.pth"
self="${repo_root}/docker/l40/managed_experiment.sh"
experiment_metadata="${repo_root}/research/experiments/${experiment_id}/EXPERIMENT.json"

require_run_authorization() {
    local experiment_status
    experiment_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${experiment_metadata}")"
    case "${experiment_status}" in
        AUTHORIZED|RUNNING) ;;
        *)
            echo "FAIL: ${experiment_id} status is ${experiment_status}; mutating/run command requires AUTHORIZED or RUNNING" >&2
            exit 1
            ;;
    esac
}

container_exists() {
    "${docker_bin}" container inspect "${container}" >/dev/null 2>&1
}

container_running() {
    [[ "$("${docker_bin}" inspect "${container}" --format '{{.State.Running}}')" == "true" ]]
}

preserve_legacy_container() {
    local pid_file pid supervisor_args state active_processes timestamp legacy_container container_exists_by_name
    [[ "$(id -un)" == "${machine}" ]] || {
        echo "FAIL: ${machine} profile must be run from the ${machine} account" >&2
        exit 1
    }
    if ! container_exists; then
        echo "LEGACY_CONTAINER=NONE name=${container}"
        return
    fi
    for pid_file in "${root}"/logs/managed/*/formal_supervisor.pid; do
        [[ -f "${pid_file}" ]] || continue
        pid="$(cat "${pid_file}")"
        supervisor_args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
        if [[ "${supervisor_args}" == *"managed_experiment.sh"* && \
              "${supervisor_args}" == *" ${machine} "* && \
              "${supervisor_args}" == *"_supervise-formal"* ]]; then
            echo "FAIL: active formal supervisor must finish before preserving ${container}: pid=${pid} args=${supervisor_args}" >&2
            exit 1
        fi
    done
    state="$("${docker_bin}" inspect "${container}" --format '{{.State.Status}}')"
    if [[ "${state}" == "running" ]]; then
        active_processes="$("${docker_bin}" top "${container}" -eo pid,args | tail -n +2 | grep -vE '[[:space:]]sleep infinity$' || true)"
        if [[ -n "${active_processes}" ]]; then
            echo "FAIL: active processes must finish before preserving ${container}:" >&2
            echo "${active_processes}" >&2
            exit 1
        fi
    fi
    timestamp="$(date -u +%Y%m%d_%H%M%S)"
    legacy_container="${container}_legacy_${timestamp}"
    container_exists_by_name="$("${docker_bin}" container inspect "${legacy_container}" --format '{{.Name}}' 2>/dev/null || true)"
    [[ -z "${container_exists_by_name}" ]] || {
        echo "FAIL: generated legacy container name already exists: ${legacy_container}" >&2
        exit 1
    }
    "${docker_bin}" inspect "${container}" \
        --format 'PRESERVING_CONTAINER name={{.Name}} state={{.State.Status}} image={{.Image}} source={{index .Config.Labels "gdrnpp.source.commit"}}'
    "${docker_bin}" rename "${container}" "${legacy_container}"
    echo "LEGACY_CONTAINER=PRESERVED old=${container} new=${legacy_container} state=${state}"
}

start_container() {
    container_exists || {
        echo "FAIL: container ${container} does not exist; run create after preserving any legacy container" >&2
        exit 1
    }
    if ! container_running; then
        "${docker_bin}" start "${container}" >/dev/null
    fi
}

image_identity() {
    "${docker_bin}" image inspect "${image_id}" --format '{{.Id}} {{index .Config.Labels "org.opencontainers.image.revision"}}'
}

check_image_identity() {
    local identity actual_image_id image_build_source
    python3 -m research.experiment_system.environment verify-host \
        --repo-root "${repo_root}" \
        --binding "${environment_binding_host}" \
        --image-id "${image_id}" >/dev/null
    identity="$(image_identity)"
    actual_image_id="${identity%% *}"
    image_build_source="${identity##* }"
    [[ "${actual_image_id}" == "${image_id}" ]] || {
        echo "FAIL: resolved image ID ${actual_image_id} != binding ${image_id}" >&2
        exit 1
    }
    [[ "${image_build_source}" == "${environment_build_source}" ]] || {
        echo "FAIL: image build-source ${image_build_source} != binding ${environment_build_source}" >&2
        exit 1
    }
    echo "ENVIRONMENT_IMAGE_REF=${image_ref} IMAGE_ID=${image_id} BUILD_SOURCE=${environment_build_source} SOURCE_COMMIT=${commit}"
}

check_container_identity() {
    local expected_image actual_image
    expected_image="${image_id}"
    actual_image="$("${docker_bin}" inspect "${container}" --format '{{.Image}}')"
    [[ "${actual_image}" == "${expected_image}" ]] || {
        echo "FAIL: container ${container} uses ${actual_image}, expected ${expected_image}" >&2
        exit 1
    }
}

access_check() {
    [[ "$(id -un)" == "${machine}" ]] || {
        echo "FAIL: ${machine} profile must be run from the ${machine} account" >&2
        exit 1
    }
    "${docker_bin}" info >/dev/null
    for directory in \
        "${root}/datasets/BOP_DATASETS/lm/train_pbr" \
        "${root}/datasets/BOP_DATASETS/lmo/test" \
        "${root}/datasets/VOC/VOC2012/JPEGImages" \
        "${baseline_root}"; do
        [[ -d "${directory}" && -r "${directory}" ]] || {
            echo "FAIL: missing or unreadable directory ${directory}" >&2
            exit 1
        }
    done
    checkpoint="${root}/weights/lmo_pbr/model_final_wo_optim.pth"
    [[ -r "${checkpoint}" ]] || {
        echo "FAIL: missing official checkpoint ${checkpoint}" >&2
        exit 1
    }
    actual_sha="$(sha256sum "${checkpoint}" | cut -d' ' -f1)"
    [[ "${actual_sha}" == "bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c" ]] || {
        echo "FAIL: official checkpoint SHA-256 mismatch: ${actual_sha}" >&2
        exit 1
    }
    check_image_identity
    echo "ACCESS=PASS machine=${machine} experiment=${experiment_id} seed=42 source=${commit} environment=${image_id}"
    nvidia-smi -i "${gpu_id}" \
        --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader
    active="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "${active}" ]]; then
        echo "GPU_OCCUPANCY_SNAPSHOT=ACTIVE_NON_BLOCKING"
        echo "${active}"
    else
        echo "GPU_OCCUPANCY_SNAPSHOT=IDLE"
    fi
}

write_profile() {
    mkdir -p "${runtime_dir}"
    temporary="${profile_host}.tmp.$$"
    printf '%s\n' \
        '{' \
        '  "schema_version": 1,' \
        "  \"profile_id\": \"${machine}-managed-container\"," \
        '  "assets": {' \
        '    "bop_datasets_root": "/workspace/gdrnpp/datasets/BOP_DATASETS",' \
        '    "official_gt_baseline": "/workspace/baselines/official_gt",' \
        '    "official_lmo_checkpoint": "/workspace/gdrnpp/pretrained_models/lmo_pbr/model_final_wo_optim.pth",' \
        '    "output_root": "/workspace/gdrnpp/output/experiments",' \
        '    "voc2012_root": "/workspace/gdrnpp/datasets/VOCdevkit/VOC2012"' \
        '  }' \
        '}' > "${temporary}"
    if [[ -e "${profile_host}" ]] && ! cmp -s "${temporary}" "${profile_host}"; then
        rm -f "${temporary}"
        echo "FAIL: existing runtime profile differs: ${profile_host}" >&2
        exit 1
    fi
    if [[ ! -e "${profile_host}" ]]; then
        mv "${temporary}" "${profile_host}"
    else
        rm -f "${temporary}"
    fi
}

create_container() {
    access_check
    container_exists && {
        echo "FAIL: container name ${container} already exists; run ${self} ${machine} ${experiment} preserve first" >&2
        exit 1
    }
    write_profile
    mkdir -p \
        "${root}/outputs" \
        "${root}/logs" \
        "${root}/cache/matplotlib" \
        "${root}/cache/gdrnpp_datasets" \
        "${root}/audit" \
        "${home_dir}/.cache" \
        "${repo_root}/.cache" \
        "${output_root_host}"
    gpu_uuid="$(nvidia-smi -i "${gpu_id}" --query-gpu=uuid --format=csv,noheader)"
    "${docker_bin}" run -d \
        --gpus "device=${gpu_id}" \
        --user "$(id -u):$(id -g)" \
        --name "${container}" \
        --hostname "${container}" \
        --shm-size=16g \
        --label "gdrnpp.managed=true" \
        --label "gdrnpp.machine=${machine}" \
        --label "gdrnpp.source.commit=${commit}" \
        --label "gdrnpp.environment.image_id=${image_id}" \
        --label "gdrnpp.environment.build_source=${environment_build_source}" \
        --env "HOME=/home/gdrn" \
        --env "XDG_CACHE_HOME=/home/gdrn/.cache" \
        --env "MPLCONFIGDIR=/home/gdrn/.cache/matplotlib" \
        --env "GDRN_DATASET_CACHE_DIR=/home/gdrn/.cache/gdrnpp_datasets" \
        --env "BOP_RENDERER_PATH=/opt/bop_renderer/build" \
        --env "GDRN_IMAGE_ID=${image_id}" \
        --env "GDRN_SOURCE_COMMIT=${commit}" \
        --env "GDRN_ENVIRONMENT_BUILD_SOURCE=${environment_build_source}" \
        --env "GDRN_PHYSICAL_GPU=${gpu_id}" \
        --env "GDRN_GPU_UUID=${gpu_uuid}" \
        --mount "type=bind,src=${repo_root},dst=/workspace/gdrnpp,readonly" \
        --mount "type=bind,src=${root}/datasets/BOP_DATASETS,dst=/workspace/gdrnpp/datasets/BOP_DATASETS,readonly" \
        --mount "type=bind,src=${root}/datasets/VOC,dst=/workspace/gdrnpp/datasets/VOCdevkit,readonly" \
        --mount "type=bind,src=${root}/weights,dst=/workspace/gdrnpp/pretrained_models,readonly" \
        --mount "type=bind,src=${baseline_root},dst=/workspace/baselines/official_gt,readonly" \
        --mount "type=bind,src=${root}/outputs,dst=/workspace/gdrnpp/output" \
        --mount "type=bind,src=${root}/cache/gdrnpp_datasets,dst=/workspace/gdrnpp/.cache" \
        --mount "type=bind,src=${home_dir},dst=/home/gdrn" \
        --mount "type=bind,src=${root}/cache,dst=/home/gdrn/.cache" \
        --mount "type=bind,src=${runtime_dir},dst=/workspace/runtime,readonly" \
        "${image_id}" sleep infinity >/dev/null
    echo "CONTAINER_CREATED=${container} source=${commit} environment_image_id=${image_id} environment_build_source=${environment_build_source} physical_gpu=${gpu_id}"
}

gate() {
    start_container
    check_image_identity
    check_container_identity
    "${docker_bin}" exec "${container}" bash -lc \
        "cd /workspace/gdrnpp && test -d /opt/bop_renderer/build && test -w /home/gdrn && test -w /home/gdrn/.cache && test -w \${GDRN_DATASET_CACHE_DIR} && test -w /workspace/gdrnpp/.cache && python -m research.experiment_system.environment verify-runtime --repo-root /workspace/gdrnpp --binding ${environment_binding_container} --image-id ${image_id} && /usr/local/bin/verify-gdrn-environment && /usr/local/bin/verify-gdrn-native && python -m research.experiment_system.cli registry --check"
    if [[ "${experiment}" == "EXP005" ]]; then
        "${docker_bin}" exec "${container}" bash -lc \
            "cd /workspace/gdrnpp && python -m research.pnp_control.preflight --config ${config_root}/train.py --weights ${official_container} --expected-seed 42"
    elif [[ "${experiment}" == "EXP012" ]]; then
        "${docker_bin}" exec "${container}" bash -lc \
            "cd /workspace/gdrnpp && python -m research.next_pose_head.preflight --config ${config_root}/train.py --weights ${official_container} --device cuda --skip-round-trip"
    elif [[ "${experiment}" == "EXP013D" ]]; then
        "${docker_bin}" exec "${container}" bash -lc \
            "cd /workspace/gdrnpp && python -m research.exp014.preflight --config ${config_root}/train.py --device cuda --skip-round-trip"
    elif [[ "${experiment}" == EXP013* ]]; then
        "${docker_bin}" exec "${container}" bash -lc \
            "cd /workspace/gdrnpp && python -m research.exp013.preflight --variant ${exp013_variant} --config ${config_root}/train.py --weights ${official_container} --device cuda --skip-round-trip"
    else
        "${docker_bin}" exec "${container}" bash -lc \
            "cd /workspace/gdrnpp && python -m research.cpm_head.preflight --config ${config_root}/train.py --weights ${official_container} --device cuda --skip-round-trip"
    fi
    echo "GATE=PASS experiment=${experiment_id} seed=42"
}

next_run_id() {
    local mode="$1" timestamp attempt candidate
    timestamp="$(date -u +%Y%m%d-%H%M%S)"
    if [[ -n "${MANAGED_ATTEMPT:-}" ]]; then
        printf -v candidate 'RUN-%s-%s-s42-a%02d' "${timestamp}" "${mode}" "${MANAGED_ATTEMPT}"
        if [[ -e "${output_root_host}/${experiment_id}/${candidate}" ]]; then
            echo "FAIL: managed retry run already exists: ${candidate}" >&2
            exit 1
        fi
        echo "${candidate}"
        return
    fi
    for attempt in $(seq 1 99); do
        printf -v candidate 'RUN-%s-%s-s42-a%02d' "${timestamp}" "${mode}" "${attempt}"
        if [[ ! -e "${output_root_host}/${experiment_id}/${candidate}" ]]; then
            echo "${candidate}"
            return
        fi
    done
    echo "FAIL: no unused attempt remains for ${timestamp}/${mode}" >&2
    exit 1
}

mode_config() {
    case "$1" in
        smoke) echo "${config_root}/smoke.py" ;;
        audit) echo "${config_root}/audit48.py" ;;
        formal) echo "${config_root}/train.py" ;;
        *) return 1 ;;
    esac
}

launch_once() {
    local mode="$1" run_id config run_host run_container
    start_container
    check_image_identity
    check_container_identity
    if "${docker_bin}" exec "${container}" pgrep -f '[r]esearch.managed_runtime.run' >/dev/null 2>&1; then
        echo "FAIL: another managed training attempt is active in ${container}" >&2
        exit 1
    fi
    run_id="$(next_run_id "${mode}")"
    config="$(mode_config "${mode}")"
    "${docker_bin}" exec "${container}" bash -lc \
        "cd /workspace/gdrnpp && python -m research.experiment_system.cli prepare --experiment research/experiments/${experiment_id}/EXPERIMENT.json --config ${config} --mode ${mode} --seed 42 --run-id ${run_id} --output-root ${output_root_container} --profile ${profile_container} --environment-binding ${environment_binding_container} --environment-image-id ${image_id}"
    run_host="${output_root_host}/${experiment_id}/${run_id}"
    run_container="${output_root_container}/${experiment_id}/${run_id}"
    mkdir -p "${log_root}"
    printf '%s\n' "${run_host}" > "${log_root}/latest_${mode}_run.path"
    nvidia-smi -i "${gpu_id}" \
        --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader > "${run_host}/meta/gpu_before_launch.csv"
    nvidia-smi -i "${gpu_id}" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader > "${run_host}/meta/gpu_processes_before_launch.csv" 2>/dev/null || true
    "${docker_bin}" exec -d "${container}" bash -lc \
        "cd /workspace/gdrnpp && python -m research.managed_runtime.run --run-dir ${run_container} --config ${run_container}/meta/resolved_config.py --mode ${mode} --isolation-role ${isolation_role} --official ${official_container} --cuda-device 0"
    echo "RUN_LAUNCHED mode=${mode} run_id=${run_id} run_dir_host=${run_host}"
}

latest_run() {
    local mode="$1" pointer="${log_root}/latest_${mode}_run.path"
    [[ -f "${pointer}" ]] || {
        echo "FAIL: no ${mode} run pointer for ${experiment_id}" >&2
        exit 1
    }
    cat "${pointer}"
}

require_complete() {
    local mode="$1" run state container_image
    run="$(latest_run "${mode}")"
    state="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${run}/meta/run_state.json")"
    [[ "${state}" == "COMPLETE" ]] || {
        echo "FAIL: required ${mode} run is ${state}: ${run}" >&2
        exit 1
    }
    container_image="$("${docker_bin}" inspect "${container}" --format '{{.Image}}')"
    python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); expected=(sys.argv[2],sys.argv[3],sys.argv[4]); actual=(m["source"]["source_git_commit"],m["environment"]["environment_image_id"],str(m["seed"])); assert actual == expected, (actual, expected)' \
        "${run}/meta/run_manifest.json" "${commit}" "${container_image}" "42" || {
        echo "FAIL: ${mode} gate was produced by a different commit/image/seed: ${run}" >&2
        exit 1
    }
}

supervise_formal() {
    local output run failure exit_code missing_status_checks
    for attempt in $(seq 1 99); do
        output="$(MANAGED_ATTEMPT="${attempt}" "${self}" "${machine}" "${experiment}" _launch-formal-once)"
        echo "${output}"
        run="${output##*run_dir_host=}"
        missing_status_checks=0
        while [[ ! -f "${run}/meta/launcher_status.json" ]]; do
            sleep 600
            missing_status_checks=$((missing_status_checks + 1))
            if (( missing_status_checks >= 3 )); then
                echo "SUPERVISOR=FAILED launcher status was not created: ${run}" >&2
                exit 1
            fi
        done
        while true; do
            state="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "${run}/meta/launcher_status.json")"
            [[ "${state}" == "FINISHED" ]] && break
            sleep 900
        done
        exit_code="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["exit_code"])' "${run}/meta/launcher_status.json")"
        failure="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("failure_kind"))' "${run}/meta/launcher_status.json")"
        if [[ "${exit_code}" == "0" ]]; then
            echo "SUPERVISOR=TRAINING_COMPLETE run=${run}"
            exit 0
        fi
        if [[ "${failure}" != "CUDA_OOM" ]]; then
            echo "SUPERVISOR=FAILED_NON_OOM run=${run} failure=${failure}" >&2
            exit 1
        fi
        echo "SUPERVISOR=CUDA_OOM_RETRY_IN_600_SECONDS run=${run}"
        sleep 600
    done
    echo "SUPERVISOR=FAILED attempt limit reached" >&2
    exit 1
}

start_supervisor() {
    if [[ "${experiment}" != "EXP012" ]]; then
        require_complete smoke
        require_complete audit
    fi
    mkdir -p "${log_root}"
    supervisor_pid_file="${log_root}/formal_supervisor.pid"
    if [[ -f "${supervisor_pid_file}" ]]; then
        old_pid="$(cat "${supervisor_pid_file}")"
        if kill -0 "${old_pid}" 2>/dev/null; then
            echo "FAIL: formal supervisor is already running: ${old_pid}" >&2
            exit 1
        fi
    fi
    stamp="$(date +%Y%m%d_%H%M%S)"
    supervisor_log="${log_root}/formal_supervisor_${stamp}.log"
    printf '%s\n' "${supervisor_log}" > "${log_root}/latest_formal_supervisor.path"
    nohup "${self}" "${machine}" "${experiment}" _supervise-formal > "${supervisor_log}" 2>&1 </dev/null &
    printf '%s\n' "$!" > "${supervisor_pid_file}"
    echo "FORMAL_SUPERVISOR=STARTED pid=$! log=${supervisor_log} poll_seconds=900 oom_retry_seconds=600"
}

show_status() {
    echo "experiment=${experiment_id} machine=${machine} container=${container} seed=42"
    if container_exists; then
        "${docker_bin}" inspect "${container}" --format 'container_state={{.State.Status}} image_id={{.Image}}'
    else
        echo "container_state=MISSING"
    fi
    for mode in smoke audit formal; do
        pointer="${log_root}/latest_${mode}_run.path"
        [[ -f "${pointer}" ]] || continue
        run="$(cat "${pointer}")"
        echo "${mode}_run=${run}"
        [[ ! -f "${run}/meta/run_state.json" ]] || cat "${run}/meta/run_state.json"
        [[ ! -f "${run}/meta/launcher_status.json" ]] || cat "${run}/meta/launcher_status.json"
        [[ ! -f "${run}/train/epoch_summary.jsonl" ]] || tail -n 1 "${run}/train/epoch_summary.jsonl"
    done
    nvidia-smi -i "${gpu_id}" \
        --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader
}

watch_latest() {
    local mode="formal" run
    if [[ ! -f "${log_root}/latest_formal_run.path" ]]; then
        mode="audit"
    fi
    if [[ ! -f "${log_root}/latest_${mode}_run.path" ]]; then
        mode="smoke"
    fi
    run="$(latest_run "${mode}")"
    echo "Following ${run}/train/console.log; Ctrl-C stops viewing only."
    tail -f "${run}/train/console.log"
}

finalize_run() {
    local run run_container
    start_container
    run="$(latest_run formal)"
    run_container="${output_root_container}/${experiment_id}/$(basename "${run}")"
    "${docker_bin}" exec "${container}" bash -lc \
        "cd /workspace/gdrnpp && python -m research.managed_runtime.finalize ${run_container} && python -m research.experiment_system.cli verify-run ${run_container}"
}

case "${command}" in
    access) access_check ;;
    preserve) preserve_legacy_container ;;
    create) require_run_authorization; create_container ;;
    gate) require_run_authorization; gate ;;
    smoke) require_run_authorization; launch_once smoke ;;
    audit48)
        require_run_authorization
        require_complete smoke
        launch_once audit
        ;;
    launch) require_run_authorization; start_supervisor ;;
    status) show_status ;;
    watch) watch_latest ;;
    finalize) require_run_authorization; finalize_run ;;
    _launch-formal-once) require_run_authorization; launch_once formal ;;
    _supervise-formal) require_run_authorization; supervise_formal ;;
    *) usage ;;
esac
