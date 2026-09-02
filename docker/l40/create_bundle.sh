#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
output_dir="${1:-${repo_root}/.local/release}"

branch="$(git -C "${repo_root}" symbolic-ref --quiet --short HEAD)" || {
    echo "FAIL: bundle source must be an attached branch" >&2
    exit 1
}
[[ "${branch}" == "main" ]] || {
    echo "FAIL: bundle source branch must be main, got ${branch}" >&2
    exit 1
}

status="$(git -C "${repo_root}" status --porcelain --untracked-files=all)"
[[ -z "${status}" ]] || {
    echo "FAIL: Git working tree must be clean before bundle creation" >&2
    printf '%s\n' "${status}" >&2
    exit 1
}

full_sha="$(git -C "${repo_root}" rev-parse HEAD)"
short_sha="$(git -C "${repo_root}" rev-parse --short=7 HEAD)"
bundle="${output_dir}/GDRNPP-RGBD-${short_sha}.bundle"
[[ ! -e "${bundle}" ]] || {
    echo "FAIL: bundle already exists: ${bundle}" >&2
    exit 1
}

mkdir -p "${output_dir}"
git -C "${repo_root}" bundle create "${bundle}" main

verify_repo="$(mktemp -d "/tmp/gdrnpp-bundle-verify-${short_sha}.XXXXXX")"
git -c init.defaultBranch=main -C "${verify_repo}" init --bare --quiet
git -C "${verify_repo}" bundle verify "${bundle}"

printf 'BUNDLE_CREATED path=%s source_commit=%s branch=%s\n' \
    "${bundle}" "${full_sha}" "${branch}"
