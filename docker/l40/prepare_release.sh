#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "usage: $0 lab0|lab1 ENVIRONMENT_IMAGE_REF" >&2
    exit 2
}

[[ $# -eq 2 ]] || usage
machine="$1"
environment_image="$2"

case "${machine}" in
    lab0|lab1) ;;
    *) usage ;;
esac

[[ "$(id -un)" == "${machine}" ]] || {
    echo "FAIL: release must be prepared from the ${machine} account" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

python3 -m research.experiment_system.environment prepare \
    --repo-root "${repo_root}" \
    --image "${environment_image}" \
    --docker-bin /usr/bin/docker
