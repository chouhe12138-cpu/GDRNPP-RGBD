#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gdrnpp-matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/gdrnpp-pycache}"

exec ./core/gdrn_modeling/train_gdrn.sh \
  configs/gdrn/lmo_pbr/convnext_stage3c1_official_gt_lmo.py \
  "${CUDA_DEVICE:-0}" \
  --eval-only
