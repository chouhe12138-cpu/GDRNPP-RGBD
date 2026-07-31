#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/gdrnpp
/usr/local/bin/verify-gdrn-environment
/usr/local/bin/verify-gdrn-native
PYTHONPATH=/workspace/gdrnpp PYTHONPYCACHEPREFIX=/tmp/gdrnpp-pycache python -m pytest -q -o cache_dir=/tmp/gdrnpp-pytest-cache research/quality_coverage/tests research/pnp_control/tests research/pose_head_utilization/tests research/pbr_validation/tests research/oracle_diagnostic/tests research/pose_aggregation/tests
python -m research.quality_coverage.preflight
