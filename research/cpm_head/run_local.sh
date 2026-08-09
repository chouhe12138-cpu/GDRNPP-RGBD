#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

output_dir="output/cpm_head/local_integration"
if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty local CPM output: $output_dir" >&2
  exit 2
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gdrnpp-matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/gdrnpp-pycache}"

python -m research.cpm_head.preflight --device cuda --skip-round-trip

exec ./core/gdrn_modeling/train_gdrn.sh \
  configs/gdrn/lmo_pbr/convnext_cpm_head_local_lmo.py \
  "${CUDA_DEVICE:-0}"
