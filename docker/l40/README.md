# GDRNPP managed L40 container

## EXP013 managed aliases

`managed_experiment.sh` supports the fixed mapping `EXP013A→lab0`,
`EXP013B→lab1`, and (only after metadata authorization) `EXP013C→lab0`.
All three use `PNP_REPLACEMENT` checkpoint isolation and the shared
`research.exp013.preflight`. A/B/C use the standard
`gate→smoke→audit48→launch→finalize` sequence; no EXP012 smoke/audit exception
is inherited. A `PLANNED` C is rejected before any mutating or run command.

This image reproduces the verified GDRNPP stack on an NVIDIA L40:

- Ubuntu 22.04, CUDA 12.1, cuDNN 8;
- Python 3.10;
- PyTorch 2.2.0+cu121 and torchvision 0.17.0+cu121;
- Detectron2 at commit `02b5c4e295e990042a714712c21dc79b731e8833`;
- native extensions rebuilt for SM 8.9;
- Ceres 1.14.0 and the pinned BOP renderer;
- experiment-system、CPM、Stage 1–3C 研究测试在镜像构建时执行。

The image is a stable runtime/environment image. Its revision label identifies
the source used to build the environment and native extensions; it is not the
source commit of every experiment. Datasets and official weights are mounted
read-only. Outputs, caches, and logs are separate writable host directories.

Host workflow:

```bash
git clone https://gitee.com/Aa1156433279/gdrnpp-rgbd.git \
  /data/labs/<lab>/docker_data/chx/releases/GDRNPP-RGBD-<short-commit>
cd /data/labs/<lab>/docker_data/chx/releases/GDRNPP-RGBD-<short-commit>
git checkout --detach <full-commit>
docker/l40/prepare_release.sh lab0 <existing-environment-image>
```

When GitHub/Gitee is unreachable from the server, copy a complete Git bundle to
the account-owned `transfer/` directory and clone the release from that bundle.
The release must still be checked out at the registered 40-character commit in
detached mode before `prepare_release.sh` is called. Neither release preparation
nor the managed launcher uses `sudo`.

The managed container names are intentionally fixed as `lab0_chx` and
`lab1_chx`. If `create` reports that the name already exists, preserve the idle
legacy container before retrying:

```bash
docker/l40/managed_experiment.sh lab0 EXP013A preserve
docker/l40/managed_experiment.sh lab0 EXP013A create
```

`preserve` never stops or deletes the old container. It refuses to rename while
a managed formal supervisor or a non-idle container process is active, otherwise
it renames the container to `lab0_chx_legacy_<UTC timestamp>` (or the equivalent
lab1 name). Do not bypass a refusal with `sudo`, `docker rm`, or `docker stop`.

`prepare_release.sh` does not build an image. It verifies that Docker,
requirements/vendor and native inputs remain compatible, then copies only the
image-built native artifacts into ignored paths in the release checkout. The
release is mounted read-only at `/workspace/gdrnpp`, so all Python/config code
comes from the selected Git commit.

Release preparation also writes a binding-v2 SHA-256 snapshot of every tracked
source file. Git cleanliness and detached HEAD are checked on the host. The
runtime container does not need a `git` executable: it re-hashes the tracked
snapshot, image ID and native artifacts. A missing Git client inside the stable
environment image is therefore expected and does not require rebuilding it.

The managed container mounts an account-local writable home and cache. The same
external dataset cache is also mounted at `/workspace/gdrnpp/.cache` as a
compatibility point for unchanged upstream dataset/evaluator code. This nested
mount contains only ignored runtime cache files; the release source and all
tracked Python/config files remain read-only and authoritative. The launcher
creates the empty, Git-ignored release-side `.cache` mountpoint before Docker
applies the read-only source mount; Docker never needs to modify that mount.

Run `build_image.sh` only after an intentional Dockerfile, dependency, vendor,
native extension or ABI change. Ordinary Python/config experiments reuse the
existing image ID.

不要在仍有实验或未提交修改的旧 server repo 中 pull。

EXP005/EXP009 由宿主机统一入口运行：

```bash
docker/l40/managed_experiment.sh lab0 EXP005 status
docker/l40/managed_experiment.sh lab1 EXP009 status
```

新的受管 run 固定 seed `42`。正式 40 epoch 之前必须依次通过 gate、smoke 和
batch-48 audit。
