#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
chx_root="$(cd -- "${repo_root}/../.." && pwd)"
commit="$(git -C "${repo_root}" rev-parse HEAD)"
short_commit="$(git -C "${repo_root}" rev-parse --short=12 HEAD)"
image="${IMAGE:-gdrnpp-research:torch220-cu121-sm89-${short_commit}}"
remote_raw="$(git -C "${repo_root}" remote get-url origin)"
remote="$(printf '%s' "${remote_raw}" | sed -E 's#(https?://)[^/@]+@#\1#')"
stamp="$(date +%Y%m%d_%H%M%S)"
audit_dir="${chx_root}/audit/image_${stamp}"
build_log="${audit_dir}/docker-build.log"

if [[ -n "$(git -C "${repo_root}" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "FAIL: Docker release build requires a completely clean Git worktree" >&2
    git -C "${repo_root}" status --short --branch >&2
    exit 1
fi
mkdir -p "${audit_dir}"

# The lab network routes IPv4 egress nowhere (the archive host and the mirrors both
# time out on 101.6.15.130) while IPv6 works, and the default bridge gives a build
# container no IPv6 at all, so apt and pip can only reach the mirrors through the
# host's network stack.
set +e
/usr/bin/docker build --network=host --progress=plain --file "${script_dir}/Dockerfile" --build-arg GDRN_UID="$(id -u)" --build-arg GDRN_GID="$(id -g)" --build-arg GIT_COMMIT="${commit}" --build-arg GIT_REMOTE="${remote}" --tag "${image}" "${repo_root}" 2>&1 | tee "${build_log}"
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
