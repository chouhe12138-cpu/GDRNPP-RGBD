#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
chx_root="$(cd -- "${repo_root}/../.." && pwd)"
image="${IMAGE:-gdrnpp-stage3bc2:torch220-cu121-sm89-v2}"
commit="$(git -C "${repo_root}" rev-parse HEAD)"
stamp="$(date +%Y%m%d_%H%M%S)"
audit_dir="${chx_root}/audit/image_${stamp}"
build_log="${audit_dir}/docker-build.log"

git -C "${repo_root}" diff --quiet
git -C "${repo_root}" diff --cached --quiet
mkdir -p "${audit_dir}"

set +e
/usr/bin/docker build --progress=plain --file "${script_dir}/Dockerfile" --build-arg GDRN_UID="$(id -u)" --build-arg GDRN_GID="$(id -g)" --build-arg GIT_COMMIT="${commit}" --tag "${image}" "${repo_root}" 2>&1 | tee "${build_log}"
build_code=${PIPESTATUS[0]}
set -e

echo "docker_build_exit_code=${build_code}"
echo "build_log=${build_log}"
if (( build_code != 0 )); then
    exit "${build_code}"
fi

/usr/bin/docker image inspect "${image}" > "${audit_dir}/image-inspect.json"
/usr/bin/docker image inspect "${image}" --format '{{.Id}}' | tee "${audit_dir}/image-id.txt"
echo "image=${image}"
echo "commit=${commit}"
echo "audit_dir=${audit_dir}"
