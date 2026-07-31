#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/.local/bop_toolkit:${PROJECT_ROOT}/.local/bop_renderer/build${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${PROJECT_ROOT}/core/csrc/uncertainty_pnp/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gdrnpp-matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/gdrnpp-pycache}"

exec python "${PROJECT_ROOT}/research/pose_aggregation/run_diagnostic.py" "$@"
