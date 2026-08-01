# GDRNPP Stage 3C-1 Server Migration Handoff

Last updated: 2026-08-01

## Scope and safety

This file records the Docker/Gitee/L40 migration performed in the side
conversation. It does not authorize formal 40-epoch training until every gate
listed below passes.

Do not:

- reset or clean either Git worktree;
- delete `/data/labs/lab1/docker_data/chx_old_20260801`;
- run Docker prune commands;
- modify or delete other users' containers, images, volumes, or directories;
- use a GPU other than the one assigned to `lab1`;
- start the formal L40 run before the runtime and one-epoch gates pass.

## Git and Gitee

- Private repository:
  `https://gitee.com/Aa1156433279/gdrnpp-rgbd.git`
- Branch: `main`
- Current local/Gitee/server commit:
  `57856598ffa262c5f9711cb2b4c963713b0377c6`
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

The old containers selected physical GPU 1. Confirm that the current lab
assignment for `lab1` is still GPU 1 before creating the new container.

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
image ID:  sha256:8e2ee36cae8c9916c6f98b2e29d7c0c9d8cdde4d06daca31532f2f7ca47891a99
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

## Current stopping point

The image and VOC are ready. The following assets have not yet been confirmed
as transferred into the new workspace:

```text
/data/labs/lab1/docker_data/chx/datasets/BOP_DATASETS/lm
/data/labs/lab1/docker_data/chx/datasets/BOP_DATASETS/lmo
/data/labs/lab1/docker_data/chx/weights/lmo_pbr/model_final_wo_optim.pth
```

Local source assets:

```text
/home/wsluser/Datasets/BOP_DATASETS/lm   approximately 27 GB
/home/wsluser/Datasets/BOP_DATASETS/lmo  approximately 815 MB
/home/wsluser/GDRNPP-RGBD/pretrained_models/lmo_pbr/model_final_wo_optim.pth
```

Official checkpoint:

```text
size:   410708489 bytes
sha256: bafa869d4e6c00410517ecb1add59f234ed1642e47fabcf3aa6e0e8a1b498a8c
```

Server SSH address observed on 2026-08-01:

```text
219.216.65.163
```

`/usr/bin/rsync` is available. Use `rsync -a --partial --info=progress2` so an
interrupted transfer can be resumed. Use `rsync -aL` for the checkpoint
because the path inside this repository is a local symbolic link.

## Next actions

1. Inspect `git status` locally before doing anything.
2. Transfer full `lm`, `lmo`, and the official checkpoint from WSL to the
   server with resumable rsync.
3. Verify:
   - 50 directories under `lm/train_pbr`;
   - approximately 27 GB `lm`, 815 MB `lmo`, and 2 GB VOC;
   - official checkpoint SHA-256 exactly matches the value above.
4. Confirm with the lab administrator that `lab1` is still assigned GPU 1.
5. Create the container with:
   `docker/l40/run_container.sh 1`.
   Physical GPU 1 will appear as logical CUDA device 0 inside the container.
6. Inside the container run:
   `docker/l40/verify_runtime.sh`.
   This checks CUDA/L40, native extensions, 43 tests, official checkpoint
   compatibility, parameter isolation, and identity initialization.
7. Run the one-epoch architecture gate:
   `research/quality_coverage/run_local.sh`.
8. Inspect finite loss, memory, checkpoint reload, and changed-tensor
   isolation.
9. Only after all gates pass may
   `research/quality_coverage/run_l40.sh` start the formal 40-epoch run.

## Expected experiment discipline

- The formal C1 run trains only `quality_coverage_net`.
- Backbone, geometry head, and direct Patch-PnP remain frozen.
- Training uses all 50 PBR scenes.
- LM-O GT-box evaluation runs after every five completed epochs.
- The deployment pose remains direct GDRNPP `R,t`; RANSAC is diagnostic only.
- Keep best one plus latest two checkpoints.
