#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 B|C2 smoke|formal" >&2
    exit 2
fi

role="$1"
mode="$2"
case "${role}:${mode}" in
    B:formal)
        config="configs/gdrn/lmo_pbr/convnext_stage3c0_pnp_only_lmo.py"
        ;;
    B:smoke)
        config="configs/gdrn/lmo_pbr/convnext_stage3c0_pnp_only_smoke_lmo.py"
        ;;
    C2:formal)
        config="configs/gdrn/lmo_pbr/convnext_stage3c2_pnp_quality_coverage_lmo.py"
        ;;
    C2:smoke)
        config="configs/gdrn/lmo_pbr/convnext_stage3c2_pnp_quality_coverage_smoke_lmo.py"
        ;;
    *)
        echo "usage: $0 B|C2 smoke|formal" >&2
        exit 2
        ;;
esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gdrnpp-matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/gdrnpp-pycache}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:On January 1, 2023:UserWarning,ignore:pkg_resources is deprecated as an API:UserWarning}"

build_model=()
if [[ "${DEEP_PREFLIGHT:-0}" == "1" ]]; then
    build_model=(--build-model)
fi
python -m research.stage3c_runtime.preflight \
    "${role}" \
    --config "${config}" \
    "${build_model[@]}"

exec ./core/gdrn_modeling/train_gdrn.sh "${config}" "${CUDA_DEVICE:-0}"
