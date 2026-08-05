#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 B|C2 access|create|start|gate|smoke|validate|formal|finalize|status|watch" >&2
    exit 2
fi

role="$1"
command="$2"
case "${role}" in
    B)
        gpu_id="${GPU_ID:-0}"
        root="${CHX_ROOT:-/data/labs/lab0/docker_data/chx}"
        container="${CONTAINER_NAME:-lab0_chx}"
        baseline_root="${BASELINE_ROOT:-${root}/baselines/official_gt}"
        run_name="B_patch_pnp"
        ;;
    C2)
        gpu_id="${GPU_ID:-1}"
        root="${CHX_ROOT:-/data/labs/lab1/docker_data/chx}"
        container="${CONTAINER_NAME:-lab1_chx}"
        baseline_root="${BASELINE_ROOT:-${root}/outputs/EXP-20260731-006/official_gt}"
        run_name="C2_joint"
        ;;
    *)
        echo "role must be B or C2" >&2
        exit 2
        ;;
esac

docker_bin="${DOCKER_BIN:-/usr/bin/docker}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${root}/logs/stage3c/${run_name}"
smoke_output="${root}/outputs/stage3c_smoke/${run_name}"
formal_output="${root}/outputs/stage3c/${run_name}"

container_exists() {
    "${docker_bin}" container inspect "${container}" >/dev/null 2>&1
}

container_running() {
    [[ "$("${docker_bin}" inspect "${container}" --format '{{.State.Running}}')" == "true" ]]
}

start_container() {
    container_exists || {
        echo "FAIL: missing container ${container}; run '$0 ${role} create'" >&2
        exit 1
    }
    if ! container_running; then
        "${docker_bin}" start "${container}" >/dev/null
    fi
}

latest_file() {
    local suffix="$1"
    local files=()
    shopt -s nullglob
    files=("${log_dir}"/*.${suffix})
    shopt -u nullglob
    ((${#files[@]} > 0)) || return 1
    ls -1t "${files[@]}" | head -n 1
}

latest_smoke_status() {
    local files=()
    shopt -s nullglob
    files=("${log_dir}"/smoke_*.status.json)
    shopt -u nullglob
    ((${#files[@]} > 0)) || return 1
    ls -1t "${files[@]}" | head -n 1
}

check_gpu_idle() {
    local active
    active="$(nvidia-smi -i "${gpu_id}" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null || true)"
    [[ -z "${active}" ]] || {
        echo "FAIL: physical GPU ${gpu_id} has active compute processes:" >&2
        echo "${active}" >&2
        exit 1
    }
}

launch() {
    local mode="$1"
    local output="$2"
    local timestamp log status
    start_container
    check_gpu_idle
    [[ ! -e "${output}" ]] || {
        echo "FAIL: output already exists and will not be overwritten: ${output}" >&2
        exit 1
    }
    mkdir -p "${log_dir}"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    log="${log_dir}/${mode}_${timestamp}.log"
    status="${log_dir}/${mode}_${timestamp}.status.json"
    "${docker_bin}" exec -d \
        -e CUDA_DEVICE=0 \
        -e DEEP_PREFLIGHT=1 \
        -e RUN_LOG="/workspace/logs/stage3c/${run_name}/$(basename "${log}")" \
        -e RUN_STATUS="/workspace/logs/stage3c/${run_name}/$(basename "${status}")" \
        "${container}" bash -lc "
            printf '{\"state\":\"RUNNING\",\"role\":\"${role}\",\"mode\":\"${mode}\",\"started\":\"%s\"}\n' \
                \"\$(date --iso-8601=seconds)\" > \"\${RUN_STATUS}\"
            cd /workspace/gdrnpp
            set +e
            research/stage3c_runtime/run.sh ${role} ${mode} > \"\${RUN_LOG}\" 2>&1
            rc=\$?
            set -e
            printf '{\"state\":\"FINISHED\",\"role\":\"${role}\",\"mode\":\"${mode}\",\"exit\":%s,\"finished\":\"%s\"}\n' \
                \"\${rc}\" \"\$(date --iso-8601=seconds)\" > \"\${RUN_STATUS}\"
            exit \"\${rc}\"
        "
    echo "${mode^^}_LAUNCH=STARTED"
    echo "log=${log}"
    echo "status=${status}"
}

validate_smoke() {
    local status checkpoint metrics
    start_container
    status="$(latest_smoke_status 2>/dev/null || true)"
    [[ -n "${status}" ]] || {
        echo "FAIL: no smoke status found for ${role}" >&2
        exit 1
    }
    grep -q '"state":"FINISHED"' "${status}" || {
        echo "FAIL: smoke has not finished: ${status}" >&2
        exit 1
    }
    grep -q '"exit":0' "${status}" || {
        echo "FAIL: smoke did not exit successfully: ${status}" >&2
        exit 1
    }
    checkpoint="${smoke_output}/checkpoints/model_epoch_001.pth"
    metrics="${smoke_output}/train/metrics.jsonl"
    [[ -f "${checkpoint}" && -f "${metrics}" ]] || {
        echo "FAIL: incomplete smoke artifacts below ${smoke_output}" >&2
        exit 1
    }
    "${docker_bin}" exec "${container}" bash -lc "
        cd /workspace/gdrnpp
        python -m research.stage3c_runtime.verify_checkpoint_isolation \
            ${role} \
            --official pretrained_models/lmo_pbr/model_final_wo_optim.pth \
            --trained output/stage3c_smoke/${run_name}/checkpoints/model_epoch_001.pth
    "
    echo "SMOKE_VALIDATION=PASS role=${role}"
}

case "${command}" in
    access)
        asset_root="${ASSET_ROOT:-${root}}"
        [[ -x "${docker_bin}" ]] || {
            echo "ACCESS=FAIL docker_not_executable=${docker_bin}"
            exit 1
        }
        "${docker_bin}" info >/dev/null 2>&1 || {
            echo "ACCESS=FAIL docker_daemon=${docker_bin}"
            exit 1
        }
        gpu_info="$(nvidia-smi -i "${gpu_id}" \
            --query-gpu=index,uuid,name,memory.total \
            --format=csv,noheader 2>/dev/null)" || {
            echo "ACCESS=FAIL physical_gpu=${gpu_id}"
            exit 1
        }
        for directory in \
            "${asset_root}/datasets/BOP_DATASETS/lm/train_pbr" \
            "${asset_root}/datasets/BOP_DATASETS/lmo/test" \
            "${asset_root}/datasets/VOC/VOC2012/JPEGImages" \
            "${baseline_root}"; do
            [[ -d "${directory}" && -r "${directory}" && -x "${directory}" ]] || {
                echo "ACCESS=FAIL directory=${directory}"
                exit 1
            }
        done
        checkpoint="${asset_root}/weights/lmo_pbr/model_final_wo_optim.pth"
        [[ -f "${checkpoint}" && -r "${checkpoint}" ]] || {
            echo "ACCESS=FAIL checkpoint=${checkpoint}"
            exit 1
        }
        echo "ACCESS=PASS asset_root=${asset_root}"
        echo "GPU=${gpu_info}"
        df -h "${root}" 2>/dev/null | tail -n 1 || true
        ;;
    create)
        "${script_dir}/run_stage3c_container.sh" "${role}"
        ;;
    start)
        start_container
        echo "CONTAINER=RUNNING name=${container}"
        ;;
    gate)
        start_container
        "${docker_bin}" exec "${container}" bash -lc \
            "cd /workspace/gdrnpp && docker/l40/verify_runtime.sh"
        ;;
    smoke)
        launch smoke "${smoke_output}"
        ;;
    formal)
        validate_smoke
        launch formal "${formal_output}"
        ;;
    finalize)
        start_container
        checkpoint="${formal_output}/checkpoints/model_epoch_040.pth"
        [[ -f "${checkpoint}" ]] || {
            echo "FAIL: missing fixed epoch-40 checkpoint: ${checkpoint}" >&2
            exit 1
        }
        "${docker_bin}" exec "${container}" bash -lc "
            cd /workspace/gdrnpp
            python -m research.stage3c_runtime.verify_checkpoint_isolation \
                ${role} \
                --official pretrained_models/lmo_pbr/model_final_wo_optim.pth \
                --trained output/stage3c/${run_name}/checkpoints/model_epoch_040.pth
            python -m research.stage3c_runtime.summarize_formal \
                ${role} \
                output/stage3c/${run_name} \
                /workspace/baselines/stage3c/official_gt
        "
        ;;
    validate)
        validate_smoke
        ;;
    status)
        status="$(latest_file status.json 2>/dev/null || true)"
        if [[ -z "${status}" ]]; then
            echo "STATE=NOT_STARTED role=${role}"
        else
            echo "status=${status}"
            cat "${status}"
            log="${status%.status.json}.log"
            grep ' iter: ' "${log}" | tail -n 1 || true
        fi
        ;;
    watch)
        log="$(latest_file log)"
        echo "Following ${log}; Ctrl-C stops viewing only."
        tail -f "${log}"
        ;;
    *)
        echo "unknown command: ${command}" >&2
        exit 2
        ;;
esac
