#!/usr/bin/env bash
set -Eeuo pipefail

mode="${1:-preflight}"
project_root="/home/wsluser/GDRNPP-RGBD"
official_cfg="${GDRNPP_OFFICIAL_CFG:-configs/gdrn/lmo_pbr/research/_base_/lmo_gt_eval.py}"
official_ckpt="${GDRNPP_OFFICIAL_CKPT:-pretrained_models/lmo_pbr/model_final_wo_optim.pth}"
: "${EPROPNP_ROOT:?set EPROPNP_ROOT to EPro-PnP-v2-main/EPro-PnP-6DoF_v2}"

cd "${project_root}"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPYCACHEPREFIX="/tmp/gdrnpp-pycache"
export MPLCONFIGDIR="/tmp/gdrnpp-matplotlib"

case "${mode}" in
  preflight)
    python -m research.diagnostics.exp019_epro.preflight \
      --epropnp-root "${EPROPNP_ROOT}" \
      --gdrn-config "${official_cfg}" \
      --checkpoint "${official_ckpt}" \
      --device cpu
    ;;
  smoke)
    run_id="RUN-$(date +%Y%m%d-%H%M%S)-smoke-s20260730"
    run_dir="output/experiments/EXP-20260907-019-epro-geometry-utilization/${run_id}"
    python -m research.diagnostics.exp019_epro.runner \
      --epropnp-root "${EPROPNP_ROOT}" \
      --gdrn-config "${official_cfg}" \
      --checkpoint "${official_ckpt}" \
      --device cuda:0 \
      --limit 32 \
      --output "${run_dir}"
    echo "SMOKE_RESULT=${run_dir}/run_metadata.json"
    ;;
  full)
    if [[ "$(git branch --show-current)" != "EPro-GDRNPP研究" ]]; then
      echo "formal diagnostic must run from branch EPro-GDRNPP研究" >&2
      exit 2
    fi
    if [[ -n "$(git status --porcelain)" ]]; then
      echo "formal diagnostic requires a clean committed source tree" >&2
      exit 2
    fi
    run_id="RUN-$(date +%Y%m%d-%H%M%S)-full-s20260730"
    run_dir="output/experiments/EXP-20260907-019-epro-geometry-utilization/${run_id}"
    python -m research.diagnostics.exp019_epro.runner \
      --epropnp-root "${EPROPNP_ROOT}" \
      --gdrn-config "${official_cfg}" \
      --checkpoint "${official_ckpt}" \
      --device cuda:0 \
      --output "${run_dir}"
    python -m research.diagnostics.exp019_epro.evaluate "${run_dir}"
    echo "FULL_RESULT=${run_dir}"
    echo "SUMMARY=${run_dir}/summary.json"
    echo "GATES=${run_dir}/gate_report.json"
    ;;
  evaluate)
    run_dir="${2:?usage: bash scripts/run_exp019_epro.sh evaluate RUN_DIR}"
    python -m research.diagnostics.exp019_epro.evaluate "${run_dir}"
    echo "SUMMARY=${run_dir}/summary.json"
    echo "GATES=${run_dir}/gate_report.json"
    ;;
  *)
    echo "usage: bash scripts/run_exp019_epro.sh {preflight|smoke|full|evaluate RUN_DIR}" >&2
    exit 2
    ;;
esac
