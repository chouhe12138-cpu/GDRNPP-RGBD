# GDRNPP Stage 3C-1 Server Migration Handoff

Last updated: 2026-08-05

## Scope and safety

This file records the Docker/Gitee/L40 migration and the verified runtime used
by the Stage 3C-1 formal run.  The data, runtime, and one-epoch smoke gates
passed before formal training started.

Do not:

- reset or clean either Git worktree;
- delete `/data/labs/lab1/docker_data/chx_old_20260801`;
- run Docker prune commands;
- modify or delete other users' containers, images, volumes, or directories;
- use physical GPU 1 from `lab1` or physical GPU 0 from the newly assigned
  `lab0`; do not use GPUs 2 or 3;
- use long pasted command blocks when an equivalent repository script can be
  used.

## Git and Gitee

- Private repository:
  `https://gitee.com/Aa1156433279/gdrnpp-rgbd.git`
- Branch: `main`
- Original migration/build commit:
  `57856598ffa262c5f9711cb2b4c963713b0377c6`
- Server worktree commit observed by the formal-run host gate:
  `1e335b22e936411f4819854b5fe8f448cc305e6d`
- Tags:
  - `stage3c1-local-pass-20260801`
  - `l40-docker-v1-20260801`
  - `l40-docker-v1.1-20260801`
- Server clone:
  `/data/labs/lab1/docker_data/chx/code/GDRNPP-RGBD`
- Server uses HTTPS with an interactively entered temporary PAT and
  `git -c credential.helper=`. Do not store or embed the PAT.
- Personal WSL is the source of truth. The server only pulls, builds, and
  runs; it does not commit or push.

## Old environment cleanup

The following user-owned resources were removed successfully:

- containers `lab1_chx` and `lab1_chx_test`;
- volume `lab1_chx_home`;
- images `gdrnpp-clean:torch220-cu121-sm89-v1` through `v4`.

The old personal directory was preserved, not deleted:

```text
/data/labs/lab1/docker_data/chx_old_20260801
```

Its container inspection record is:

```text
/data/labs/lab1/docker_data/chx_old_20260801/audit/old_containers_20260801.json
```

The old containers selected physical GPU 1.  The user explicitly confirmed
that `lab1` remains assigned physical GPU 1.  Physical GPU 0 is now assigned
through `lab0`, not through the `lab1` container or account.

## New server workspace

Root:

```text
/data/labs/lab1/docker_data/chx
```

Created subdirectories:

```text
code
outputs
weights
cache
logs
transfer
datasets
audit
```

The verified old VOC2012 copy was copied into the new workspace:

```text
/data/labs/lab1/docker_data/chx/datasets/VOC/VOC2012
```

Verified JPEG count:

```text
17125
```

## Docker image

The clean Stage 3C-1 L40 image was built successfully from commit
`57856598ffa262c5f9711cb2b4c963713b0377c6`.

```text
image tag: gdrnpp-stage3c1:torch220-cu121-sm89-v1
image ID:  sha256:8e2ee36cae8c9916c6f98b2e29d7c0c9d8cde4d06daca31532f2f7ca47891a99
```

Build result:

```text
docker_build_exit_code=0
43 tests passed
```

Audit directory:

```text
/data/labs/lab1/docker_data/chx/audit/image_20260801_015600
```

Build log:

```text
/data/labs/lab1/docker_data/chx/audit/image_20260801_015600/docker-build.log
```

The image contains:

- Ubuntu 22.04;
- CUDA 12.1 and cuDNN 8;
- Python 3.10;
- PyTorch 2.2.0+cu121 and torchvision 0.17.0+cu121;
- Detectron2, Ceres 1.14, BOP renderer, and GDRNPP native extensions;
- native CUDA code built for SM 8.9;
- the exact Git revision in the OCI image label.

Third-party build inputs passed SHA-256 verification. The Tsinghua PyPI mirror
and the PyTorch CUDA 12.1 wheel index both returned HTTP 200 from the server.

## Verified data and checkpoint

The required assets were transferred and verified in the new workspace:

```text
/data/labs/lab1/docker_data/chx/datasets/BOP_DATASETS/lm
/data/labs/lab1/docker_data/chx/datasets/BOP_DATASETS/lmo
/data/labs/lab1/docker_data/chx/weights/lmo_pbr/model_final_wo_optim.pth
```

Observed dataset facts:

```text
LM:                 approximately 27 GB
LM-O:               approximately 816 MB
VOC:                approximately 2.0 GB
LM train_pbr scenes: 50
LM train_pbr RGBs:   50,000
VOC2012 JPEGs:       17,125
```

Official checkpoint:

```text
size:   410708489 bytes
sha256: bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c
```

Stage 3C-1 fixed Epoch 40 checkpoint:

```text
server path: /data/labs/lab1/docker_data/chx/outputs/EXP-20260731-006/quality_coverage_full/model_0255919.pth
local copy:  E:\6D姿态估计\26-08-02\model_0255919.pth
size:        411145010 bytes
sha256:      d3ab7167f2fc5f6aab8d7e8444c5b816036bd64e38f647a26e994c8e91939aa6
verification: server and local SHA-256 match
```

Server SSH address observed on 2026-08-01:

```text
219.216.65.163
```

The user performs all server actions.  Agents must not initiate SSH, generate
SSH keys, or persist credentials.  `/usr/bin/rsync` is available if a future
transfer is explicitly requested.

## Host and GPU facts

Observed on the shared server:

```text
CPU:           2 x Intel Xeon Platinum 8358P
cores/threads: 64 physical / 128 logical
NUMA nodes:    2
memory:        approximately 503 GiB total, 483 GiB available when sampled
GPU:           NVIDIA L40, 46,068 MiB each
lab1 GPU:      physical GPU 1
lab0 GPU:      physical GPU 0 (assigned; container/data gate pending)
GPU UUID:      GPU-90265a5c-6348-51b7-c829-c60dd351c289
```

Physical GPU 1 appears as logical CUDA device 0 inside the container.

## Container and runtime state

```text
container: lab1_chx_stage3c1
image:     gdrnpp-stage3c1:torch220-cu121-sm89-v1
```

Verified mounts:

```text
BOP datasets  -> /workspace/gdrnpp/datasets/BOP_DATASETS  read-only
VOC            -> /workspace/gdrnpp/datasets/VOCdevkit    read-only
weights        -> /workspace/gdrnpp/pretrained_models     read-only
outputs        -> /workspace/gdrnpp/output                read-write
cache          -> /home/gdrn/.cache                       read-write
logs           -> /workspace/logs                         read-write
```

The container runtime gate passed:

- CUDA and the L40 were visible;
- native extensions loaded;
- 43 tests passed;
- all 392 compatible official checkpoint tensors loaded;
- identity initialization passed;
- the formal config resolved 50 PBR scenes, 40 epochs, batch 48, and
  five-epoch evaluation.

The one-epoch smoke completed all 2,048 micro-batches and passed:

- finite loss;
- quality and coverage branch updates;
- zero changes to the 392 official tensors;
- strict checkpoint reload;
- peak GPU memory of approximately 4,820 MiB.

Use the repository controller rather than pasted command blocks:

```text
docker/l40/stage3c1.sh start
docker/l40/stage3c1.sh gate
docker/l40/stage3c1.sh status
docker/l40/stage3c1.sh watch
docker/l40/stage3c1.sh validate
```

Use `/usr/bin/docker`, not the shell alias `sudo docker`; the user does not
have an administrator password.

## Completed C1 state

The Stage 3C-1 formal 40-epoch run completed.  Fixed epoch 40 scored 68.9742%
BOP AR and 50.57% ADD(-S), with 4/8 objects nonnegative.  The final result is
`C1_SCREEN_FAIL`.  Preserve the existing container and output directory as
immutable evidence.

## Expected experiment discipline

- The formal C1 run trains only `quality_coverage_net`.
- Backbone, geometry head, and direct Patch-PnP remain frozen.
- Training uses all 50 PBR scenes.
- LM-O GT-box evaluation runs after every five completed epochs.
- The deployment pose remains direct GDRNPP `R,t`; RANSAC is diagnostic only.
- Keep best one plus latest two checkpoints.

## B/C2 dual-GPU preparation

The next matched formal controls are:

```text
physical GPU 0 / lab0 / lab0_chx_stage3b  -> B Patch-PnP-only
physical GPU 1 / lab1 / lab1_chx_stage3c2 -> C2 joint adaptation
```

Use image `gdrnpp-stage3bc2:torch220-cu121-sm89-v2` for both roles.  Build it
once after the source is clean and committed by the user; both containers
must record the same image ID.  When both accounts use the same host Docker
daemon, the image is already shared.  The B/C2 launcher maps the invoking
account's numeric UID/GID into the container, so a lab1-built image does not
need rebuilding solely for lab0 file ownership.

`lab0` has not yet passed its data-access or Docker gate.  First run the
read-only `access` command in `docker/l40/stage3bc2.sh`.  If lab0 can read the
existing lab1 dataset and weight directories, mount them read-only and create
only lab0-owned outputs, logs, cache, and audit directories.  If it cannot,
prepare an independent lab0 copy through an approved shared path or
non-deleting rsync and verify all counts and hashes.  Do not change ownership
or permissions of lab1 resources.

```bash
ASSET_ROOT=/data/labs/lab1/docker_data/chx \
BASELINE_ROOT=/data/labs/lab1/docker_data/chx/outputs/EXP-20260731-006/official_gt \
docker/l40/stage3bc2.sh B access
```

The new formal runs use structured outputs under:

```text
output/stage3c/B_patch_pnp
output/stage3c/C2_joint
```

They disable TensorBoard, capture only compact human-readable logs, store
full resolved configuration/environment metadata separately, and avoid the
duplicate final evaluation when epoch 40 has already been evaluated.
