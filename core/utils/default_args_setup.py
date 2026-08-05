import argparse
import os
import os.path as osp
import sys
import mmcv
from mmcv import DictAction
import torch
import PIL
import json
import platform
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.collect_env import collect_env_info
from loguru import logger

from lib.utils.setup_logger import setup_my_logger
from lib.utils.time_utils import get_time_str
from lib.utils.setup_logger_loguru import setup_logger
from core.utils import my_comm as comm
from core.gdrn_modeling.engine.artifact_layout import (
    artifact_dir,
    compact_log_enabled,
)


def my_default_argument_parser(epilog=None):
    """Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:
Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth
Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    # distributed training launcher:
    # none: non-distributed or use the detectron2 default launcher
    # pytorch: use pytorch launcher
    # hvd: use horovod launcher
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "hvd"],
        default="none",
        help="job launcher",
    )
    # pytorch dist options ======
    parser.add_argument("--local_rank", type=int, default=0)
    # hvd options ======
    parser.add_argument(
        "--fp16_allreduce",
        action="store_true",
        default=False,
        help="use fp16 compression during allreduce for hvd",
    )
    parser.add_argument(
        "--use-adasum",
        action="store_true",
        default=False,
        help="use adasum algorithm to do reduction",
    )
    # -------------
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--opts",
        nargs="+",
        action=DictAction,
        help="arguments in dict, modify config using command-line args",
    )
    return parser


def my_default_setup(cfg, args):
    """Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.OUTPUT_DIR
    compact_log = compact_log_enabled(cfg)
    meta_dir = artifact_dir(cfg, "meta")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)
        PathManager.mkdirs(meta_dir)

    rank = comm.get_rank()
    # setup_my_logger(output_dir, distributed_rank=rank, name="fvcore")
    # setup_my_logger(output_dir, distributed_rank=rank, name="detectron2")
    # setup_my_logger(output_dir, distributed_rank=rank, name="timm")
    # setup_my_logger(output_dir, distributed_rank=rank, name="core")
    # setup_my_logger(output_dir, distributed_rank=rank, name="lib")
    # logger = setup_my_logger(output_dir, distributed_rank=rank)
    log_output = None if compact_log else osp.join(output_dir, f"log_{get_time_str()}.txt")
    setup_logger(log_output, distributed_rank=rank)

    if compact_log:
        logger.info(
            "RUN_SETUP rank=%d world_size=%d seed=%d config=%s output=%s",
            rank,
            comm.get_world_size(),
            int(cfg.SEED),
            getattr(args, "config_file", ""),
            output_dir,
        )
    else:
        logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
        logger.info("Environment info:\n" + collect_env_info())
        logger.info("Command line arguments: " + str(args))
        if hasattr(args, "config_file") and args.config_file != "":
            logger.info(
                "Contents of args.config_file={}:\n{}".format(
                    args.config_file,
                    PathManager.open(args.config_file, "r").read(),
                )
            )
        logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        # path = os.path.join(output_dir, "config.yaml")
        # with PathManager.open(path, "w") as f:
        #     f.write(cfg.dump())
        path = osp.join(meta_dir, osp.basename(args.config_file))
        cfg.dump(path)
        logger.info("Full config saved to {}".format(path))
        if compact_log:
            environment_path = osp.join(meta_dir, "environment.json")
            environment = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "visible_cuda_devices": torch.cuda.device_count(),
                "command_line": str(args),
            }
            with PathManager.open(environment_path, "w") as handle:
                json.dump(environment, handle, indent=2, sort_keys=True)
            manifest_path = osp.join(meta_dir, "manifest.json")
            manifest = {
                "role": os.environ.get("GDRN_STAGE3C_ROLE"),
                "seed": int(cfg.SEED),
                "output_dir": cfg.OUTPUT_DIR,
                "model_weights": cfg.MODEL.WEIGHTS,
                "git_commit": os.environ.get("GDRN_GIT_COMMIT"),
                "docker_image_id": os.environ.get("GDRN_IMAGE_ID"),
                "physical_gpu": os.environ.get("GDRN_PHYSICAL_GPU"),
                "gpu_uuid": os.environ.get("GDRN_GPU_UUID"),
                "logical_cuda_device": os.environ.get("CUDA_DEVICE", "0"),
                "structured_layout": True,
                "tensorboard": False,
            }
            if not osp.exists(manifest_path):
                with PathManager.open(manifest_path, "w") as handle:
                    json.dump(manifest, handle, indent=2, sort_keys=True)

    assert (
        args.num_gpus <= torch.cuda.device_count() and args.num_gpus >= 1
    ), f"args.num_gpus: {args.num_gpus}, available num gpus: {torch.cuda.device_count()}"

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

    # set mmcv backend
    mmcv_backend = cfg.get("IM_BACKEND", "cv2")
    if mmcv_backend == "pillow" and "post" not in PIL.__version__:
        logger.warning("Consider installing pillow-simd!")
    logger.info(f"Used mmcv backend: {mmcv_backend}")
    mmcv.use_backend(mmcv_backend)
