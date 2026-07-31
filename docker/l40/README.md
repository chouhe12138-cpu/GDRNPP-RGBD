# Stage 3C-1 L40 container

This image reproduces the verified GDRNPP stack on an NVIDIA L40:

- Ubuntu 22.04, CUDA 12.1, cuDNN 8;
- Python 3.10;
- PyTorch 2.2.0+cu121 and torchvision 0.17.0+cu121;
- Detectron2 at commit `02b5c4e295e990042a714712c21dc79b731e8833`;
- native extensions rebuilt for SM 8.9;
- Ceres 1.14.0 and the pinned BOP renderer;
- the 43 Stage 1–3C tests executed during the image build.

The image contains the exact Git commit passed by `build_image.sh`. Datasets
and official weights are mounted read-only. Outputs, caches, and logs are
separate writable host directories.

Host workflow:

```bash
cd /data/labs/lab1/docker_data/chx/code/GDRNPP-RGBD
git pull --ff-only
docker/l40/build_image.sh
docker/l40/run_container.sh 1
```

Inside the container:

```bash
docker/l40/verify_runtime.sh
research/quality_coverage/run_local.sh
```

Do not run the formal 40-epoch command until runtime verification and the
one-epoch architecture gate both pass.
