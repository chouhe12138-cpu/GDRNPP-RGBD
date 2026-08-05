#!/usr/bin/env bash
set -Eeuo pipefail

container="lab0_chx"
root="/data/labs/lab0/docker_data/chx"
bench_id="${BENCH_ID:-$(date +%Y%m%d_%H%M%S)}"
candidate_text="${WORKER_CANDIDATES:-4 8 12 16 24}"
bench_root="${root}/logs/stage3c/B_patch_pnp/worker_benchmark/${bench_id}"
summary="${bench_root}/summary.tsv"
ranking="${bench_root}/confirmed_ranking.tsv"
state="${bench_root}/state.txt"
lock_file="${root}/logs/stage3c/B_patch_pnp/worker_benchmark.lock"

mkdir -p "${bench_root}"
exec 9>"${lock_file}"
flock -n 9 || {
    echo "FAIL: another NUM_WORKERS benchmark is already running" >&2
    exit 1
}

printf 'RUNNING\n' > "${state}"
printf 'label\tworkers\texit\telapsed_seconds\toutput\tlog\n' > "${summary}"

finish_failed() {
    printf 'FAILED\n' > "${state}"
}
trap finish_failed ERR

active="$(nvidia-smi -i 0 \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true)"
[[ -z "${active}" ]] || {
    echo "FAIL: GPU0 has active compute processes:" >&2
    echo "${active}" >&2
    exit 1
}

echo "BENCH_ID=${bench_id}"
echo "CANDIDATES=${candidate_text}"
echo "PROTOCOL=batch48_local_pbr_one_epoch_no_eval"
/usr/bin/docker inspect "${container}" \
    --format 'CONTAINER={{.Name}} IMAGE_ID={{.Image}} USER={{.Config.User}}'

run_case() {
    local label="$1"
    local workers="$2"
    local output_rel output_host log started finished elapsed rc
    output_rel="output/stage3c_worker_benchmark/${bench_id}/${label}"
    output_host="${root}/outputs/stage3c_worker_benchmark/${bench_id}/${label}"
    log="${bench_root}/${label}.log"
    [[ ! -e "${output_host}" ]] || {
        echo "FAIL: output already exists: ${output_host}" >&2
        return 1
    }

    echo "START label=${label} workers=${workers}"
    started="$(date +%s)"
    set +e
    /usr/bin/docker exec "${container}" bash -lc "
        cd /workspace/gdrnpp
        ./core/gdrn_modeling/train_gdrn.sh \
            configs/gdrn/lmo_pbr/convnext_stage3c0_pnp_only_lmo.py \
            0 \
            --opts \
            DATALOADER.NUM_WORKERS=${workers} \
            'DATASETS.TRAIN=(\"lmo_pbr_stage3_local_train\",)' \
            'DATASETS.TEST=()' \
            SOLVER.TOTAL_EPOCHS=1 \
            SOLVER.CHECKPOINT_PERIOD=99 \
            TEST.EVAL_PERIOD=0 \
            TRAIN.PRINT_FREQ=25 \
            OUTPUT_DIR=${output_rel}
    " > "${log}" 2>&1
    rc=$?
    set -e
    finished="$(date +%s)"
    elapsed="$((finished - started))"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${label}" "${workers}" "${rc}" "${elapsed}" "${output_host}" "${log}" \
        >> "${summary}"
    echo "FINISH label=${label} workers=${workers} exit=${rc} elapsed_seconds=${elapsed}"
}

# Warm the dataset and filesystem caches. This row is never ranked.
run_case "warmup_w8" 8

for workers in ${candidate_text}; do
    run_case "trial1_w${workers}" "${workers}"
done

mapfile -t finalists < <(
    awk -F '\t' 'NR > 1 && $1 ~ /^trial1_w/ && $3 == 0 {print $4, $2}' "${summary}" \
        | sort -n \
        | head -n 2 \
        | awk '{print $2}'
)
[[ ${#finalists[@]} -eq 2 ]] || {
    echo "FAIL: fewer than two worker candidates completed successfully" >&2
    exit 1
}

for workers in "${finalists[@]}"; do
    run_case "trial2_w${workers}" "${workers}"
done

awk -F '\t' '
    NR > 1 && $1 ~ /^trial[12]_w/ && $3 == 0 {
        total[$2] += $4
        count[$2] += 1
    }
    END {
        for (workers in total) {
            if (count[workers] == 2) {
                printf "%s\t%.3f\t%s\n", workers, total[workers] / count[workers], count[workers]
            }
        }
    }
' "${summary}" | sort -t $'\t' -k2,2n > "${ranking}"

read -r best_workers best_average best_count < <(
    awk -F '\t' 'NR == 1 {print $1, $2, $3}' "${ranking}"
)
read -r second_workers second_average second_count < <(
    awk -F '\t' 'NR == 2 {print $1, $2, $3}' "${ranking}"
)
[[ -n "${best_workers:-}" && -n "${second_workers:-}" ]] || {
    echo "FAIL: confirmed ranking is incomplete" >&2
    exit 1
}

relative_gap="$(awk -v best="${best_average}" -v second="${second_average}" \
    'BEGIN {printf "%.3f", ((second - best) / best) * 100.0}')"
chosen="${best_workers}"
if awk -v gap="${relative_gap}" 'BEGIN {exit !(gap < 3.0)}'; then
    if ((second_workers < best_workers)); then
        chosen="${second_workers}"
    fi
fi

{
    echo "NUM_WORKERS=${chosen}"
    echo "BEST_MEAN_SECONDS=${best_average}"
    echo "SECOND_NUM_WORKERS=${second_workers}"
    echo "SECOND_MEAN_SECONDS=${second_average}"
    echo "RELATIVE_GAP_PERCENT=${relative_gap}"
    echo "TIE_POLICY=choose_lower_workers_when_gap_below_3_percent"
} > "${bench_root}/recommendation.txt"

printf 'COMPLETE\n' > "${state}"
trap - ERR

echo "WORKER_BENCHMARK=COMPLETE"
echo "SUMMARY=${summary}"
cat "${summary}"
echo "CONFIRMED_RANKING=${ranking}"
cat "${ranking}"
echo "RECOMMENDATION=${bench_root}/recommendation.txt"
cat "${bench_root}/recommendation.txt"
