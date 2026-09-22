"""The DeepIM OpenGL renders of LINEMOD, 1000 images per object.

The renders live outside ``BOP_DATASETS`` (``datasets/lm_imgn``) and carry their
own conventions: images are ``{im_id}-color.png`` with ``{im_id} = "{obj}/{n}"``,
poses are 3x4 matrices in **meters** read from ``{im_id}-pose.txt``, and masks
are derived from the depth image because no mask files are shipped.  Camera
intrinsics are the fixed LINEMOD ones rather than per-image ones.
"""

import hashlib
import logging
import os
import os.path as osp
import sys
import time
from collections import OrderedDict

import mmcv
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import ref
from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask, mask2bbox_xywh
from lib.utils.utils import dprint, iprint, lazy_property

from .lm_pbr import DATASET_CACHE_ROOT, LM_13_OBJECTS, get_lm_metadata


logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))

# the renderer names the benchvise sequence after its colour variant
IMGN_OBJ_ALIASES = {"benchviseblue": "benchvise"}


class LM_SYN_IMGN_Dataset:
    """DeepIM LINEMOD renders sampled uniformly to ``n_per_obj`` per object."""

    def __init__(self, data_cfg):
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects
        self.dataset_root = data_cfg.get("dataset_root", osp.join(DATASETS_ROOT, "lm_imgn"))
        self.ann_root = data_cfg.get("ann_root", osp.join(self.dataset_root, "image_set"))
        self.img_root = data_cfg.get("img_root", osp.join(self.dataset_root, "imgn"))
        self.xyz_root = data_cfg.get("xyz_root", osp.join(self.dataset_root, "xyz_crop_imgn"))
        self.require_xyz = data_cfg.get("require_xyz", True)
        self.n_per_obj = int(data_cfg.get("n_per_obj", 1000))

        assert osp.exists(self.dataset_root), self.dataset_root
        assert osp.exists(self.ann_root), self.ann_root
        assert osp.exists(self.img_root), self.img_root

        self.models_root = data_cfg["models_root"]  # BOP_DATASETS/lm/models
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001

        self.with_masks = data_cfg["with_masks"]
        self.with_depth = data_cfg["with_depth"]

        self.height = data_cfg["height"]  # 480
        self.width = data_cfg["width"]  # 640

        self.cache_dir = data_cfg.get("cache_dir", DATASET_CACHE_ROOT)
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg.get("filter_invalid", True)

        ##################################################

        # NOTE: careful! Only the selected objects
        self.data_ref = ref.__dict__[data_cfg["ref_key"]]
        self.cat_ids = [int(self.data_ref.obj2id[name]) for name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))
        ##########################################################

        self.cam = np.array(self.data_ref.camera_matrix, dtype=np.float32)

    def __call__(self):
        """Load light-weight instance annotations of all images into a list of
        dicts in Detectron2 format.

        Do not load heavy data into memory in this file, since we will
        load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}".format(
                    self.name,
                    self.dataset_root,
                    self.with_masks,
                    self.with_depth,
                    __name__,
                )
                + "_n_per_obj_{}_require_xyz_{}_num_to_load_{}".format(
                    self.n_per_obj, self.require_xyz, self.num_to_load
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(self.cache_dir, "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name))

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()

        logger.info("loading dataset dicts: {}".format(self.name))
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0
        dataset_dicts = []  # ######################################################
        for obj_name in tqdm(self.objs):
            ann_file = osp.join(self.ann_root, f"train_{obj_name}.txt")
            assert osp.exists(ann_file), ann_file
            with open(ann_file, "r") as f_ann:
                indices = [line.strip("\r\n").split()[-1] for line in f_ann.readlines() if line.strip()]

            if self.n_per_obj > 0:
                sample_num = min(self.n_per_obj, len(indices))
                sel_indices_idx = np.linspace(0, len(indices) - 1, sample_num, dtype=np.int32)
                sel_indices = [indices[int(_i)] for _i in sel_indices_idx]
            else:
                sel_indices = indices

            for im_id in tqdm(sel_indices, postfix=obj_name):
                rgb_path = osp.join(self.img_root, "{}-color.png").format(im_id)
                assert osp.exists(rgb_path), rgb_path

                depth_path = osp.join(self.img_root, "{}-depth.png".format(im_id))
                pose_path = osp.join(self.img_root, "{}-pose.txt".format(im_id))

                record_obj_name = im_id.split("/")[0]
                record_obj_name = IMGN_OBJ_ALIASES.get(record_obj_name, record_obj_name)
                if record_obj_name not in self.objs:
                    continue
                obj_id = int(self.data_ref.obj2id[record_obj_name])
                cur_label = self.cat2label[obj_id]  # 0-based label

                # the renders store poses in meters, unlike the BOP annotation files
                pose = np.loadtxt(pose_path, skiprows=1).astype("float32")
                if pose.shape != (3, 4):
                    raise ValueError(f"lm_imgn pose must be 3x4, got {pose.shape} for {im_id}")
                R = pose[:3, :3]
                t = pose[:3, 3]
                quat = mat2quat(R).astype("float32")

                depth = mmcv.imread(depth_path, "unchanged") / 1000.0  # to m
                mask = (depth > 0).astype(np.uint8)

                bbox_obj = mask2bbox_xywh(mask)
                x1, y1, w, h = bbox_obj
                if self.filter_invalid:
                    if h <= 1 or w <= 1:
                        self.num_instances_without_valid_box += 1
                        continue
                area = mask.sum()
                if area < 32:  # filter out too small or nearly invisible instances
                    self.num_instances_without_valid_segmentation += 1
                    continue
                mask_rle = binary_mask_to_rle(mask.astype("bool"), compressed=True)

                proj = (self.cam @ t)
                proj = proj[:2] / proj[2]

                record = {
                    "dataset_name": self.name,
                    "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                    "depth_file": osp.relpath(depth_path, PROJ_ROOT),
                    "height": self.height,
                    "width": self.width,
                    "image_id": im_id.split("/")[-1],
                    "scene_im_id": im_id,
                    "cam": self.cam,
                    "depth_factor": 1000.0,  # the renders store depth in mm
                    "img_type": "syn",  # NOTE: OpenGL renders, always replace bg
                }

                xyz_path = osp.join(self.xyz_root, f"{im_id}-xyz.pkl")
                if self.require_xyz:
                    assert osp.exists(xyz_path), xyz_path

                inst = {
                    "category_id": cur_label,  # 0-based label
                    "bbox": bbox_obj,  # the renders ship no visibility annotation
                    "bbox_obj": bbox_obj,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "pose": pose,
                    "quat": quat,
                    "trans": t,
                    "centroid_2d": proj,  # absolute (cx, cy)
                    "segmentation": mask_rle,
                    "mask_full": mask_rle,  # the renders have no separate amodal mask
                    "visib_fract": 1.0,
                    "xyz_path": xyz_path,
                }

                model_info = self.models_info[str(obj_id)]
                inst["model_info"] = model_info
                for key in ["bbox3d_and_center"]:
                    inst[key] = self.models[cur_label][key]

                record["annotations"] = [inst]
                dataset_dicts.append(record)
                # a bounded load returns the same first N records, just sooner
                if 0 < self.num_to_load <= len(dataset_dicts):
                    break
            if 0 < self.num_to_load <= len(dataset_dicts):
                break

        if self.num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".format(
                    self.num_instances_without_valid_segmentation
                )
            )
        if self.num_instances_without_valid_box > 0:
            logger.warning(
                "Filtered out {} instances without valid box. "
                "There might be issues in your dataset generation process.".format(self.num_instances_without_valid_box)
            )
        ##########################################################################
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[: self.num_to_load]
        logger.info("loaded {} dataset dicts, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start))

        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return dataset_dicts

    @lazy_property
    def models_info(self):
        models_info_path = osp.join(self.models_root, "models_info.json")
        assert osp.exists(models_info_path), models_info_path
        models_info = mmcv.load(models_info_path)  # key is str(obj_id)
        return models_info

    @lazy_property
    def models(self):
        """Load models into a list."""
        identity = f"{osp.realpath(self.models_root)}|{self.scale_to_meter}|{'_'.join(self.objs)}"
        digest = hashlib.md5(identity.encode('utf-8')).hexdigest()
        cache_path = osp.join(self.cache_dir, f"models_{digest}.pkl")
        if osp.exists(cache_path) and self.use_cache:
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(
                    self.models_root,
                    f"obj_{self.data_ref.obj2id[obj_name]:06d}.ply",
                ),
                vertex_scale=self.scale_to_meter,
            )
            # NOTE: the bbox3d_and_center is not obtained from centered vertices
            # for BOP models, not a big problem since they had been centered
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])

            models.append(model)
        if self.use_cache:
            mmcv.mkdir_or_exist(self.cache_dir)
            logger.info("cache models to {}".format(cache_path))
            mmcv.dump(models, cache_path, protocol=4)
        return models

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


########### register datasets ############################################################


lm_model_root = "BOP_DATASETS/lm/models/"
################################################################################


SPLITS_LM_IMGN = dict(
    lm_imgn_13_train_1k_per_obj=dict(
        name="lm_imgn_13_train_1k_per_obj",
        objs=LM_13_OBJECTS,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "lm_imgn"),
        ann_root=osp.join(DATASETS_ROOT, "lm_imgn/image_set"),
        img_root=osp.join(DATASETS_ROOT, "lm_imgn/imgn"),
        xyz_root=osp.join(DATASETS_ROOT, "lm_imgn/xyz_crop_imgn"),
        n_per_obj=1000,
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/models"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        cache_dir=DATASET_CACHE_ROOT,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        ref_key="lm_full",
    ),
)
# Online geometry rendering supplies the XYZ target, so the shipped
# `xyz_crop_imgn` files are not required for training.
SPLITS_LM_IMGN["lm_imgn_13_train_1k_per_obj_online"] = dict(
    SPLITS_LM_IMGN["lm_imgn_13_train_1k_per_obj"], require_xyz=False
)
SPLITS_LM_IMGN["lm_imgn_13_train_1k_per_obj_smoke"] = dict(
    SPLITS_LM_IMGN["lm_imgn_13_train_1k_per_obj_online"],
    name="lm_imgn_13_train_1k_per_obj_smoke",
    n_per_obj=4, num_to_load=8, use_cache=False,
)


def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))
    if name in SPLITS_LM_IMGN:
        used_cfg = SPLITS_LM_IMGN[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, LM_SYN_IMGN_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        id="lm",  # NOTE: for pvnet to determine module
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_lm_metadata(obj_names=used_cfg["objs"], ref_key=used_cfg["ref_key"]),
    )


def get_available_datasets():
    return list(SPLITS_LM_IMGN.keys())
