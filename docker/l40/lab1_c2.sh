#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 prepare|benchmark-workers|benchmark-status|benchmark-watch|access|gate|smoke|validate|formal|finalize|status|watch" >&2
    exit 2
fi

# Never inherit stale overrides from an older SSH/VS Code terminal.
unset CHX_ROOT ASSET_ROOT BASELINE_ROOT GPU_ID CONTAINER_NAME

command="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
controller="${script_dir}/stage3bc2.sh"
benchmark="${script_dir}/benchmark_stage3c_workers.sh"
container="lab1_chx"
root="/data/labs/lab1/docker_data/chx"
log_root="${root}/logs/stage3c/C2_joint"
smoke_output="${root}/outputs/stage3c_smoke/C2_joint"
latest_root_file="${log_root}/latest_worker_benchmark.path"
latest_launcher_file="${log_root}/latest_worker_benchmark_launcher.path"

container_exists() {
    /usr/bin/docker container inspect "${container}" >/dev/null 2>&1
}

latest_smoke_status() {
    local files=()
    shopt -s nullglob
    files=("${log_root}"/smoke_*.status.json)
    shopt -u nullglob
    ((${#files[@]} > 0)) || return 1
    ls -1t "${files[@]}" | head -n 1
}

wait_gpu_idle() {
    local waited=0
    local limit="${GPU_WAIT_SECONDS:-86400}"
    local poll="${GPU_POLL_SECONDS:-60}"
    local active
    while true; do
        active="$(nvidia-smi -i 1 \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>/dev/null || true)"
        [[ -n "${active}" ]] || break
        echo "WAITING_GPU1 elapsed_seconds=${waited} limit_seconds=${limit}"
        echo "${active}"
        ((waited < limit)) || {
            echo "FAIL: GPU1 did not become idle before timeout" >&2
            exit 1
        }
        sleep "${poll}"
        waited="$((waited + poll))"
    done
    echo "GPU1_IDLE waited_seconds=${waited}"
}

prepare_and_validate_smoke() {
    local status state
    "${controller}" C2 access
    if ! container_exists; then
        "${controller}" C2 create
    fi
    wait_gpu_idle
    "${controller}" C2 gate

    status="$(latest_smoke_status 2>/dev/null || true)"
    if [[ ! -e "${smoke_output}" ]]; then
        "${controller}" C2 smoke
        for _ in {1..30}; do
            status="$(latest_smoke_status 2>/dev/null || true)"
            [[ -n "${status}" ]] && break
            sleep 1
        done
    fi
    [[ -n "${status}" ]] || {
        echo "FAIL: C2 smoke output/status is inconsistent" >&2
        exit 1
    }

    while true; do
        state="$(cat "${status}")"
        if grep -q '"state":"FINISHED"' "${status}"; then
            break
        fi
        echo "C2_SMOKE_WAIT status=${status}"
        echo "${state}"
        sleep 30
    done
    grep -q '"exit":0' "${status}" || {
        echo "FAIL: C2 smoke failed: ${status}" >&2
        cat "${status}" >&2
        exit 1
    }
    "${controller}" C2 validate
    echo "LAB1_C2_PREPARE=PASS"
}

latest_path() {
    local pointer="$1"
    [[ -f "${pointer}" ]] || {
        echo "FAIL: no C2 benchmark record found: ${pointer}" >&2
        exit 1
    }
    cat "${pointer}"
}

run_background_pipeline() {
    local bench_id="$1"
    local bench_root="$2"
    printf 'PREPARING\n' > "${bench_root}/state.txt"
    trap 'printf "FAILED\n" > "${bench_root}/state.txt"' ERR
    prepare_and_validate_smoke
    trap - ERR
    exec env \
        STAGE3C_ROLE=C2 \
        BENCH_ID="${bench_id}" \
        WORKER_CANDIDATES="${WORKER_CANDIDATES:-4 8 12 16 24}" \
        "${benchmark}"
}

case "${command}" in
    prepare)
        prepare_and_validate_smoke
        ;;
    benchmark-workers)
        mkdir -p "${log_root}"
        bench_id="$(date +%Y%m%d_%H%M%S)"
        bench_root="${log_root}/worker_benchmark/${bench_id}"
        launcher="${log_root}/worker_benchmark_launcher_${bench_id}.log"
        mkdir -p "${bench_root}"
        printf '%s\n' "${bench_root}" > "${latest_root_file}"
        printf '%s\n' "${launcher}" > "${latest_launcher_file}"
        nohup "${script_dir}/lab1_c2.sh" _pipeline "${bench_id}" "${bench_root}" \
            > "${launcher}" 2>&1 </dev/null &
        echo "LAB1_C2_PIPELINE=STARTED"
        echo "pid=$!"
        echo "root=${bench_root}"
        echo "launcher=${launcher}"
        ;;
    _pipeline)
        [[ $# -eq 3 ]] || {
            echo "FAIL: invalid internal pipeline invocation" >&2
            exit 2
        }
        run_background_pipeline "$2" "$3"
        ;;
    benchmark-status)
        bench_root="$(latest_path "${latest_root_file}")"
        echo "root=${bench_root}"
        if [[ -f "${bench_root}/state.txt" ]]; then
            echo "state=$(cat "${bench_root}/state.txt")"
        else
            echo "state=STARTING"
        fi
        [[ ! -f "${bench_root}/summary.tsv" ]] || cat "${bench_root}/summary.tsv"
        [[ ! -f "${bench_root}/recommendation.txt" ]] || cat "${bench_root}/recommendation.txt"
        ;;
    benchmark-watch)
        launcher="$(latest_path "${latest_launcher_file}")"
        echo "Following ${launcher}; Ctrl-C stops viewing only."
        tail -f "${launcher}"
        ;;
    access|gate|smoke|validate|formal|finalize|status|watch)
        exec "${controller}" C2 "${command}"
        ;;
    *)
        echo "unknown command: ${command}" >&2
        exit 2
        ;;
esac
