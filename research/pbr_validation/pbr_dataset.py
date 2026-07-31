"""Lightweight LM-PBR validation registration without generated xyz_crop files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

import ref
from core.gdrn_modeling.datasets.lm_pbr import get_lm_metadata


LMO_OBJECTS = (
    "ape",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
)
LMO_OBJECT_IDS = tuple(ref.lmo_full.obj2id[name] for name in LMO_OBJECTS)


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["validation_scenes"] != [12, 13, 14]:
        raise ValueError("unexpected validation scenes")
    if manifest["diagnostic_images_total"] != 1500:
        raise ValueError("unexpected diagnostic image count")
    return manifest


def build_validation_records(
    pbr_root: Path,
    manifest_path: Path,
    models_root: Path,
    minimum_visible_pixels: int = 32,
) -> Tuple[List[dict], dict]:
    """Build Detectron2 test records using only JSON metadata and file paths."""

    pbr_root = pbr_root.resolve()
    models_root = models_root.resolve()
    manifest = load_manifest(manifest_path)
    with (models_root / "models_info.json").open(encoding="utf-8") as handle:
        models_info = json.load(handle)
    cat2label = {obj_id: index for index, obj_id in enumerate(LMO_OBJECT_IDS)}
    records: List[dict] = []
    raw_instances = 0
    filtered_small = 0
    filtered_box = 0
    object_counts = {name: 0 for name in LMO_OBJECTS}
    visibility_counts = {"lt_0.1": 0, "0.1_to_0.3": 0, "0.3_to_0.5": 0, "ge_0.5": 0}

    for scene_text, image_ids in sorted(manifest["diagnostic_images"].items()):
        scene_id = int(scene_text)
        scene_root = pbr_root / f"{scene_id:06d}"
        with (scene_root / "scene_gt.json").open(encoding="utf-8") as handle:
            scene_gt = json.load(handle)
        with (scene_root / "scene_gt_info.json").open(encoding="utf-8") as handle:
            scene_info = json.load(handle)
        with (scene_root / "scene_camera.json").open(encoding="utf-8") as handle:
            scene_camera = json.load(handle)

        for image_id in image_ids:
            image_key = str(int(image_id))
            camera_info = scene_camera[image_key]
            camera = np.asarray(camera_info["cam_K"], dtype=np.float32).reshape(3, 3)
            record = {
                "dataset_name": "lmo_pbr_stage3_calibration",
                "file_name": str(scene_root / "rgb" / f"{image_id:06d}.jpg"),
                "depth_file": str(scene_root / "depth" / f"{image_id:06d}.png"),
                "height": 480,
                "width": 640,
                "image_id": int(image_id),
                "scene_im_id": f"{scene_id}/{int(image_id)}",
                "cam": camera,
                "depth_factor": 1000.0 / float(camera_info["depth_scale"]),
                "img_type": "syn_pbr",
            }
            annotations = []
            for source_index, (pose_item, info_item) in enumerate(
                zip(scene_gt[image_key], scene_info[image_key])
            ):
                obj_id = int(pose_item["obj_id"])
                if obj_id not in cat2label:
                    continue
                raw_instances += 1
                bbox_visib = info_item["bbox_visib"]
                bbox_obj = info_item["bbox_obj"]
                if bbox_visib[2] <= 1 or bbox_visib[3] <= 1:
                    filtered_box += 1
                    continue
                if int(info_item.get("px_count_visib", 0)) < minimum_visible_pixels:
                    filtered_small += 1
                    continue
                rotation = np.asarray(pose_item["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                translation = np.asarray(pose_item["cam_t_m2c"], dtype=np.float32) / 1000.0
                pose = np.hstack((rotation, translation.reshape(3, 1)))
                projection = camera @ translation
                centroid = projection[:2] / projection[2]
                visibility = float(info_item.get("visib_fract", 1.0))
                obj_name = ref.lmo_full.id2obj[obj_id]
                object_counts[obj_name] += 1
                if visibility < 0.1:
                    visibility_counts["lt_0.1"] += 1
                elif visibility < 0.3:
                    visibility_counts["0.1_to_0.3"] += 1
                elif visibility < 0.5:
                    visibility_counts["0.3_to_0.5"] += 1
                else:
                    visibility_counts["ge_0.5"] += 1
                annotations.append(
                    {
                        "category_id": cat2label[obj_id],
                        "bbox": bbox_visib,
                        "bbox_obj": bbox_obj,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "pose": pose,
                        "trans": translation,
                        "centroid_2d": centroid,
                        "visib_fract": visibility,
                        "model_info": models_info[str(obj_id)],
                        "source_anno_id": source_index,
                        "mask_visib_file": str(
                            scene_root / "mask_visib" / f"{image_id:06d}_{source_index:06d}.png"
                        ),
                        "mask_full_file": str(
                            scene_root / "mask" / f"{image_id:06d}_{source_index:06d}.png"
                        ),
                    }
                )
            if annotations:
                record["annotations"] = annotations
                records.append(record)

    summary = {
        "images": len(records),
        "raw_lmo_instances": raw_instances,
        "kept_instances": sum(len(record["annotations"]) for record in records),
        "filtered_small_mask": filtered_small,
        "filtered_invalid_box": filtered_box,
        "object_counts": object_counts,
        "visibility_counts": visibility_counts,
        "minimum_visible_pixels": minimum_visible_pixels,
    }
    return records, summary


def register_validation_dataset(
    name: str,
    pbr_root: Path,
    manifest_path: Path,
    models_root: Path,
) -> dict:
    records, summary = build_validation_records(pbr_root, manifest_path, models_root)
    register_validation_records(name, records)
    return summary


def register_validation_records(name: str, records: List[dict]) -> None:
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    DatasetCatalog.register(name, lambda records=records: records)
    MetadataCatalog.get(name).set(
        id="lmo_pbr_stage3_calibration",
        ref_key="lmo_full",
        objs=list(LMO_OBJECTS),
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_lm_metadata(obj_names=list(LMO_OBJECTS), ref_key="lmo_full"),
    )
