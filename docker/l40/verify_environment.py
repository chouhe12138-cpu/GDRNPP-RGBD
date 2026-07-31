#!/usr/bin/env python3
import argparse
import importlib
import os
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    assert sys.version_info[:2] == (3, 10), sys.version
    assert os.environ.get("TORCH_CUDA_ARCH_LIST") == "8.9"

    modules = [
        "bop_toolkit_lib",
        "cv2",
        "detectron2",
        "fvcore",
        "imageio",
        "imgaug",
        "iopath",
        "matplotlib",
        "mmcv",
        "numpy",
        "numba",
        "omegaconf",
        "PIL",
        "plyfile",
        "pytorch_lightning",
        "pytest",
        "scipy",
        "skimage",
        "tensorboardX",
        "timm",
        "torch",
        "torchvision",
        "transforms3d",
        "webdataset",
        "yacs",
    ]
    for name in modules:
        importlib.import_module(name)

    from core.gdrn_modeling.models import GDRN_double_mask  # noqa: F401
    import core.gdrn_modeling.main_gdrn  # noqa: F401
    import torch
    import torchvision

    assert torch.__version__ == "2.2.0+cu121", torch.__version__
    assert torchvision.__version__ == "0.17.0+cu121", torchvision.__version__
    assert torch.version.cuda == "12.1", torch.version.cuda

    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__} torchvision={torchvision.__version__}")
    print(f"torch_cuda={torch.version.cuda} imports={len(modules)} PASS")

    if not args.build:
        assert torch.cuda.is_available(), "torch CUDA is unavailable"
        assert torch.cuda.device_count() == 1, torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        assert (props.major, props.minor) == (8, 9)
        x = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        assert torch.isfinite(x @ x).all().item()
        print(f"gpu_count=1 gpu={props.name} capability={props.major}.{props.minor} PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
