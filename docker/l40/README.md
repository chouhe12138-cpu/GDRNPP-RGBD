# GDRNPP managed L40 container

This image reproduces the verified GDRNPP stack on an NVIDIA L40:

- Ubuntu 22.04, CUDA 12.1, cuDNN 8;
- Python 3.10;
- PyTorch 2.2.0+cu121 and torchvision 0.17.0+cu121;
- Detectron2 at commit `02b5c4e295e990042a714712c21dc79b731e8833`;
- native extensions rebuilt for SM 8.9;
- Ceres 1.14.0 and the pinned BOP renderer;
- experiment-system、CPM、Stage 1–3C 研究测试在镜像构建时执行。

The image contains the exact Git commit passed by `build_image.sh`. Datasets
and official weights are mounted read-only. Outputs, caches, and logs are
separate writable host directories.

Host workflow:

```bash
git clone https://gitee.com/Aa1156433279/gdrnpp-rgbd.git \
  /data/labs/<lab>/docker_data/chx/releases/GDRNPP-RGBD-<short-commit>
cd /data/labs/<lab>/docker_data/chx/releases/GDRNPP-RGBD-<short-commit>
git checkout --detach <full-commit>
docker/l40/build_image.sh
```

不要在仍有实验或未提交修改的旧 server repo 中 pull。

EXP005/EXP009 由宿主机统一入口运行：

```bash
docker/l40/managed_experiment.sh lab0 EXP005 status
docker/l40/managed_experiment.sh lab1 EXP009 status
```

新的受管 run 固定 seed `42`。正式 40 epoch 之前必须依次通过 gate、smoke 和
batch-48 audit。正在运行的 C2 继续使用原冻结镜像和脚本，不由该入口接管。
