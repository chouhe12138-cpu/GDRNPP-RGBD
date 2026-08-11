# GDRNPP managed L40 container

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
