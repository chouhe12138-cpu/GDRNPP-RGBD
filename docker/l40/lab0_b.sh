#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 recover|benchmark-workers|benchmark-status|benchmark-watch|access|gate|validate|formal|finalize|status|watch" >&2
    exit 2
fi

# Never inherit stale overrides from an older SSH/VS Code terminal.
unset CHX_ROOT ASSET_ROOT BASELINE_ROOT GPU_ID CONTAINER_NAME

command="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
controller="${script_dir}/stage3bc2.sh"
benchmark="${script_dir}/benchmark_stage3c_workers.sh"
container="lab0_chx"
root="/data/labs/lab0/docker_data/chx"
log_root="${root}/logs/stage3c/B_patch_pnp"
latest_root_file="${log_root}/latest_worker_benchmark.path"
latest_launcher_file="${log_root}/latest_worker_benchmark_launcher.path"

container_exists() {
    /usr/bin/docker container inspect "${container}" >/dev/null 2>&1
}

formal_pids() {
    /usr/bin/docker exec "${container}" \
        pgrep -f '[c]onvnext_stage3c0_pnp_only_lmo(\.|$)' 2>/dev/null || true
}

recover() {
    local require_idle="${1:-1}"
    local pids active attempt
    container_exists || {
        echo "FAIL: missing laboratory container ${container}" >&2
        exit 1
    }

    pids="$(formal_pids)"
    if [[ -n "${pids}" ]]; then
        echo "RECOVERY: terminating invalid early-formal process(es): ${pids}"
        /usr/bin/docker exec "${container}" kill -TERM ${pids} 2>/dev/null || true
        for attempt in {1..10}; do
            [[ -z "$(formal_pids)" ]] && break
            sleep 1
        done
        pids="$(formal_pids)"
        if [[ -n "${pids}" ]]; then
            echo "RECOVERY: force-terminating remaining process(es): ${pids}"
            /usr/bin/docker exec "${container}" kill -KILL ${pids} 2>/dev/null || true
        fi
    fi

    pids="$(formal_pids)"
    [[ -z "${pids}" ]] || {
        echo "FAIL: invalid formal process still exists: ${pids}" >&2
        exit 1
    }

    "${controller}" B validate

    active="$(nvidia-smi -i 0 \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "${active}" ]]; then
        if [[ "${require_idle}" == "1" ]]; then
            echo "FAIL: GPU0 is not idle after recovery:" >&2
            echo "${active}" >&2
            exit 1
        fi
        echo "LAB0_B_RECOVERY=PASS_GPU_WAIT_REQUIRED"
        echo "${active}"
        return
    fi
    echo "LAB0_B_RECOVERY=PASS"
}

latest_path() {
    local pointer="$1"
    [[ -f "${pointer}" ]] || {
        echo "FAIL: no benchmark record found: ${pointer}" >&2
        exit 1
    }
    cat "${pointer}"
}

case "${command}" in
    recover)
        recover 1
        ;;
    benchmark-workers)
        recover 0
        mkdir -p "${log_root}"
        bench_id="$(date +%Y%m%d_%H%M%S)"
        bench_root="${log_root}/worker_benchmark/${bench_id}"
        launcher="${log_root}/worker_benchmark_launcher_${bench_id}.log"
        printf '%s\n' "${bench_root}" > "${latest_root_file}"
        printf '%s\n' "${launcher}" > "${latest_launcher_file}"
        nohup env \
            BENCH_ID="${bench_id}" \
            WORKER_CANDIDATES="${WORKER_CANDIDATES:-4 8 12 16 24}" \
            "${benchmark}" > "${launcher}" 2>&1 </dev/null &
        echo "WORKER_BENCHMARK=STARTED"
        echo "pid=$!"
        echo "root=${bench_root}"
        echo "launcher=${launcher}"
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
    access|gate|validate|formal|finalize|status|watch)
        exec "${controller}" B "${command}"
        ;;
    *)
        echo "unknown command: ${command}" >&2
        exit 2
        ;;
esac
