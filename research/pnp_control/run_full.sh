#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gdrnpp-matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/gdrnpp-pycache}"

python -m research.pnp_control.preflight \
  --config configs/gdrn/lmo_pbr/convnext_stage3c0_pnp_only_lmo.py \
  ${DEEP_PREFLIGHT:+--deep}

exec ./core/gdrn_modeling/train_gdrn.sh \
  configs/gdrn/lmo_pbr/convnext_stage3c0_pnp_only_lmo.py \
  "${CUDA_DEVICE:-0}"
