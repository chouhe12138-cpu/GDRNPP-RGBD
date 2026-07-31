#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gdrnpp-matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/gdrnpp-pycache}"

config_path="${CONFIG_PATH:-configs/gdrn/lmo_pbr/convnext_stage3c1_quality_coverage_lmo.py}"

python -m research.quality_coverage.preflight --config "$config_path"

exec ./core/gdrn_modeling/train_gdrn.sh \
  "$config_path" \
  "${CUDA_DEVICE:-0}"
