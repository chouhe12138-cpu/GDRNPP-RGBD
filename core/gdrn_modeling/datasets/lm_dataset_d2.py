"""LINEMOD real images selected by the official BOP ``image_set`` splits.

Each LINEMOD test scene holds exactly one object and is named by that object's
BOP id, so ``lm/test/{obj_id:06d}`` is both the scene directory and the object.
The official ``lm/image_set/{obj}_{train,test}.txt`` files pick the images of
that scene, which keeps the split independent of how a machine happens to have
laid out ``lm/train``.
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
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from lib.utils.utils import dprint, iprint, lazy_property

from .lm_pbr import DATASET_CACHE_ROOT, LM_13_OBJECTS, get_lm_metadata


logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class LM_D2_Dataset:
    """LM real images; the split is defined by the official image_set files."""

    def __init__(self, data_cfg):
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects
        self.split = data_cfg.get("split", "train")
        if self.split not in ("train", "test"):
            raise ValueError(f"LM split must be 'train' or 'test', got {self.split}")

        self.dataset_root = data_cfg.get(
            "dataset_root", osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/test")
        )
        self.ann_root = data_cfg.get(
            "ann_root", osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/image_set")
        )
        self.xyz_root = data_cfg.get("xyz_root", osp.join(self.dataset_root, "xyz_crop"))
        self.require_xyz = data_cfg.get("require_xyz", True)

        assert osp.exists(self.dataset_root), self.dataset_root
        assert osp.exists(self.ann_root), self.ann_root

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

        # linemod each scene is an object, so the scene id is the BOP object id
        self.scenes = [(obj, int(obj_id)) for obj, obj_id in zip(self.objs, self.cat_ids)]

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
                + "_split_{}_require_xyz_{}_num_to_load_{}".format(
                    self.split, self.require_xyz, self.num_to_load
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
        # it is slow because of loading and converting masks to rle
        for obj_name, scene_id in tqdm(self.scenes):
            scene = f"{scene_id:06d}"
            scene_root = osp.join(self.dataset_root, scene)

            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            ann_file = osp.join(self.ann_root, f"{obj_name}_{self.split}.txt")
            assert osp.exists(ann_file), ann_file
            with open(ann_file, "r") as f_ann:
                selected = {line.strip().split()[-1] for line in f_ann.readlines() if line.strip()}

            for str_im_id in tqdm(sorted(gt_dict, key=int), postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)
                if f"{int_im_id:06d}" not in selected:
                    continue

                rgb_path = osp.join(scene_root, "rgb/{:06d}.png").format(int_im_id)
                assert osp.exists(rgb_path), rgb_path

                depth_path = osp.join(scene_root, "depth/{:06d}.png".format(int_im_id))

                K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
                depth_factor = 1000.0 / cam_dict[str_im_id]["depth_scale"]

                scene_im_id = f"{scene_id}/{int_im_id}"

                record = {
                    "dataset_name": self.name,
                    "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                    "depth_file": osp.relpath(depth_path, PROJ_ROOT),
                    "height": self.height,
                    "width": self.width,
                    "image_id": int_im_id,
                    "scene_im_id": scene_im_id,  # for evaluation
                    "cam": K,
                    "depth_factor": depth_factor,
                    "img_type": "real",  # NOTE: real image
                }
                insts = []
                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id not in self.cat_ids:
                        continue
                    gt_info = gt_info_dict[str_im_id][anno_i]
                    visib_fract = gt_info.get("visib_fract", 1.0)

                    cur_label = self.cat2label[obj_id]  # 0-based label
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") * self.scale_to_meter

                    pose = np.hstack([R, t.reshape(3, 1)])

                    quat = mat2quat(pose[:3, :3]).astype("float32")
                    trans = pose[:3, 3]

                    proj = (record["cam"] @ t.T).T
                    proj = proj[:2] / proj[2]

                    bbox_visib = gt_info["bbox_visib"]
                    bbox_obj = gt_info["bbox_obj"]
                    x1, y1, w, h = bbox_visib
                    if self.filter_invalid:
                        if h <= 1 or w <= 1:
                            self.num_instances_without_valid_box += 1
                            continue

                    mask_file = osp.join(
                        scene_root,
                        "mask/{:06d}_{:06d}.png".format(int_im_id, anno_i),
                    )
                    mask_visib_file = osp.join(
                        scene_root,
                        "mask_visib/{:06d}_{:06d}.png".format(int_im_id, anno_i),
                    )
                    assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file
                    # load mask visib
                    mask_single = mmcv.imread(mask_visib_file, "unchanged")
                    mask_single = mask_single.astype("bool")
                    area = mask_single.sum()
                    if area < 32:  # filter out too small or nearly invisible instances
                        self.num_instances_without_valid_segmentation += 1
                        continue
                    mask_rle = binary_mask_to_rle(mask_single, compressed=True)

                    # load mask full
                    mask_full = mmcv.imread(mask_file, "unchanged")
                    mask_full = mask_full.astype("bool")
                    mask_full_rle = binary_mask_to_rle(mask_full, compressed=True)

                    xyz_path = osp.join(
                        self.xyz_root,
                        f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl",
                    )
                    if self.require_xyz:
                        assert osp.exists(xyz_path), xyz_path

                    inst = {
                        "category_id": cur_label,  # 0-based label
                        "bbox": bbox_visib,
                        "bbox_obj": bbox_obj,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "pose": pose,
                        "quat": quat,
                        "trans": trans,
                        "centroid_2d": proj,  # absolute (cx, cy)
                        "segmentation": mask_rle,
                        "mask_full": mask_full_rle,
                        "visib_fract": visib_fract,
                        "xyz_path": xyz_path,
                    }

                    model_info = self.models_info[str(obj_id)]
                    inst["model_info"] = model_info
                    for key in ["bbox3d_and_center"]:
                        inst[key] = self.models[cur_label][key]
                    insts.append(inst)
                if len(insts) == 0:  # filter im without anno
                    continue
                record["annotations"] = insts
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


SPLITS_LM = dict(
    lm_13_train=dict(
        name="lm_13_train",
        objs=LM_13_OBJECTS,  # selected objects
        split="train",
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/test"),
        ann_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/image_set"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/test/xyz_crop"),
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
    lm_13_test=dict(
        name="lm_13_test",
        objs=LM_13_OBJECTS,
        split="test",
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/test"),
        ann_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/image_set"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/test/xyz_crop"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/models"),
        scale_to_meter=0.001,
        with_masks=True,
        with_depth=True,
        height=480,
        width=640,
        cache_dir=DATASET_CACHE_ROOT,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=False,
        ref_key="lm_full",
    ),
)
# Online geometry rendering supplies the XYZ target, so the pre-generated
# `xyz_crop` files are not required for training splits.
SPLITS_LM["lm_13_train_online"] = dict(SPLITS_LM["lm_13_train"], require_xyz=False)
SPLITS_LM["lm_13_test_online"] = dict(SPLITS_LM["lm_13_test"], require_xyz=False)
SPLITS_LM["lm_13_train_smoke"] = dict(
    SPLITS_LM["lm_13_train_online"], name="lm_13_train_smoke",
    num_to_load=8, use_cache=False,
)
SPLITS_LM["lm_13_test_smoke"] = dict(
    SPLITS_LM["lm_13_test_online"], name="lm_13_test_smoke",
    num_to_load=8, use_cache=False,
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
    if name in SPLITS_LM:
        used_cfg = SPLITS_LM[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, LM_D2_Dataset(used_cfg))
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
    return list(SPLITS_LM.keys())
