#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

docker_bin="${DOCKER_BIN:-/usr/bin/docker}"
chx_root="${CHX_ROOT:-/data/labs/lab1/docker_data/chx}"
container="${CONTAINER_NAME:-lab1_chx_stage3c1}"
gpu_uuid="${GPU_UUID:-GPU-90265a5c-6348-51b7-c829-c60dd351c289}"
smoke_output="${chx_root}/outputs/EXP-20260731-006/quality_coverage_local"
log_dir="${chx_root}/logs"

usage() {
    cat <<EOF
usage: $0 COMMAND

commands:
  start      start the existing container
  gate       run the container runtime gate
  smoke      start the one-epoch smoke test in the background
  status     show the latest smoke-test status (default)
  watch      follow the latest smoke-test log; Ctrl-C only stops viewing
  validate   validate the completed smoke checkpoint and parameter isolation
EOF
}

require_docker() {
    [[ -x "${docker_bin}" ]] || {
        echo "FAIL: Docker client not found at ${docker_bin}" >&2
        exit 1
    }
}

container_exists() {
    "${docker_bin}" container inspect "${container}" >/dev/null 2>&1
}

container_running() {
    [[ "$( "${docker_bin}" inspect "${container}" --format '{{.State.Running}}' )" == "true" ]]
}

start_container() {
    require_docker
    container_exists || {
        echo "FAIL: container ${container} does not exist" >&2
        echo "Create it once with: docker/l40/run_container.sh 1" >&2
        exit 1
    }
    if container_running; then
        echo "PASS: ${container} is already running"
    else
        "${docker_bin}" start "${container}" >/dev/null
        container_running || {
            echo "FAIL: ${container} did not start" >&2
            exit 1
        }
        echo "PASS: ${container} started"
    fi
}

latest_smoke_log() {
    local files=()
    shopt -s nullglob
    files=("${log_dir}"/stage3c1_smoke_*.log)
    shopt -u nullglob
    ((${#files[@]} > 0)) || return 1
    ls -1t "${files[@]}" | head -n 1
}

show_status() {
    local log status progress
    log="$(latest_smoke_log)" || {
        echo "SMOKE_STATUS=NOT_STARTED"
        return 0
    }
    status="${log%.log}.status"
    echo "log=${log}"
    if [[ -f "${status}" ]]; then
        cat "${status}"
    else
        echo "state=UNKNOWN"
    fi
    progress="$(grep ' iter: ' "${log}" | tail -n 1 || true)"
    [[ -z "${progress}" ]] || echo "${progress}"
    if container_running; then
        "${docker_bin}" exec "${container}" nvidia-smi \
            --query-gpu=index,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader
    fi
}

run_gate() {
    local timestamp log rc
    start_container
    timestamp="$(date +%Y%m%d_%H%M%S)"
    log="${log_dir}/runtime_gate_${timestamp}.log"
    set +e
    "${docker_bin}" exec "${container}" bash -lc \
        'cd /workspace/gdrnpp && docker/l40/verify_runtime.sh' \
        2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
    set -e
    if ((rc == 0)); then
        echo "CONTAINER_RUNTIME_GATE=PASS"
    else
        echo "CONTAINER_RUNTIME_GATE=FAIL"
    fi
    echo "log=${log}"
    return "${rc}"
}

active_gpu_processes() {
    nvidia-smi \
        --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader |
        grep "${gpu_uuid}" || true
}

start_smoke() {
    local active timestamp log_name status_name
    start_container
    [[ ! -e "${smoke_output}" ]] || {
        echo "FAIL: smoke output already exists: ${smoke_output}" >&2
        echo "Use '$0 status' or '$0 validate'; existing output will not be overwritten." >&2
        exit 1
    }
    active="$(active_gpu_processes)"
    [[ -z "${active}" ]] || {
        echo "FAIL: GPU 1 already has an active compute process:" >&2
        echo "${active}" >&2
        exit 1
    }

    timestamp="$(date +%Y%m%d_%H%M%S)"
    log_name="stage3c1_smoke_${timestamp}.log"
    status_name="stage3c1_smoke_${timestamp}.status"
    "${docker_bin}" exec -d \
        -e CUDA_DEVICE=0 \
        -e SMOKE_LOG="/workspace/logs/${log_name}" \
        -e SMOKE_STATUS="/workspace/logs/${status_name}" \
        "${container}" bash -lc '
            printf "state=RUNNING\nstarted=%s\n" "$(date --iso-8601=seconds)" > "${SMOKE_STATUS}"
            cd /workspace/gdrnpp
            set +e
            research/quality_coverage/run_local.sh > "${SMOKE_LOG}" 2>&1
            rc=$?
            set -e
            printf "state=FINISHED\nexit=%s\nfinished=%s\n" \
                "${rc}" "$(date --iso-8601=seconds)" > "${SMOKE_STATUS}"
            exit "${rc}"
        '
    echo "SMOKE_LAUNCH=STARTED"
    echo "Run '$0 status' to check progress."
}

watch_smoke() {
    local log
    log="$(latest_smoke_log)" || {
        echo "FAIL: no smoke log found" >&2
        exit 1
    }
    echo "Following ${log}; Ctrl-C stops viewing, not training."
    tail -f "${log}"
}

validate_smoke() {
    local log status final_checkpoint metrics rc
    log="$(latest_smoke_log)" || {
        echo "FAIL: no smoke log found" >&2
        exit 1
    }
    status="${log%.log}.status"
    grep -qx 'state=FINISHED' "${status}" || {
        echo "FAIL: smoke test is not finished" >&2
        show_status
        exit 1
    }
    grep -qx 'exit=0' "${status}" || {
        echo "FAIL: smoke process did not exit successfully" >&2
        cat "${status}"
        exit 1
    }
    final_checkpoint="${smoke_output}/model_final.pth"
    metrics="${smoke_output}/metrics.json"
    [[ -f "${final_checkpoint}" && -f "${metrics}" ]] || {
        echo "FAIL: final checkpoint or metrics.json is missing" >&2
        exit 1
    }
    start_container
    set +e
    "${docker_bin}" exec -i -w /workspace/gdrnpp \
        -e PYTHONWARNINGS=ignore \
        -e MPLCONFIGDIR=/tmp/gdrnpp-matplotlib \
        -e TF_CPP_MIN_LOG_LEVEL=2 \
        "${container}" \
        python - \
        /workspace/gdrnpp/pretrained_models/lmo_pbr/model_final_wo_optim.pth \
        /workspace/gdrnpp/output/EXP-20260731-006/quality_coverage_local/model_final.pth \
        /workspace/gdrnpp/output/EXP-20260731-006/quality_coverage_local/metrics.json \
        "/workspace/logs/$(basename "${log}")" \
        < "${repo_root}/research/quality_coverage/validate_smoke.py"
    rc=$?
    set -e
    return "${rc}"
}

command="${1:-status}"
case "${command}" in
    start) start_container ;;
    gate) run_gate ;;
    smoke) start_smoke ;;
    status) require_docker; show_status ;;
    watch) require_docker; watch_smoke ;;
    validate) require_docker; validate_smoke ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
esac
