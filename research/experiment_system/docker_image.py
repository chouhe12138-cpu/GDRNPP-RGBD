"""Resolve immutable Docker image identity for formal run manifests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def inspect_docker_image(image: str, docker_bin: Path = Path("/usr/bin/docker")) -> dict[str, Any]:
    if not image:
        raise ValueError("Docker image reference must not be empty")
    output = subprocess.check_output(
        [
            str(docker_bin),
            "image",
            "inspect",
            image,
            "--format",
            '{{.Id}}\t{{index .Config.Labels "org.opencontainers.image.revision"}}',
        ],
        text=True,
    ).strip()
    parts = output.split("\t")
    if len(parts) != 2 or not parts[0].startswith("sha256:"):
        raise RuntimeError(f"unexpected Docker image identity: {output!r}")
    revision = parts[1].strip()
    if not revision or revision == "<no value>":
        raise RuntimeError("Docker image has no org.opencontainers.image.revision label")
    return {
        "image": image,
        "image_id": parts[0],
        "revision": revision,
    }
