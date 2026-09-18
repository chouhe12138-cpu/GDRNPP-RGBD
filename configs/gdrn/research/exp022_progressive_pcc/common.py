"""Dataset-independent Progressive PCC model on the generic GDRN runtime."""

from configs.gdrn.research.exp022_progressive_pcc.method import PCC_INIT_CFG

_base_ = ["../_base_/research_runtime.py"]

INPUT = dict(DZI_PAD_SCALE=1.5, COLOR_AUG_PROB=0.8,
             COLOR_AUG_TYPE="ROI10D")
MODEL = dict(
    WEIGHTS="", LOAD_DETS_TEST=False, BBOX_TYPE="AMODAL_CLIP",
    POSE_NET=dict(
        NAME="GDRN_PCC", XYZ_ONLINE=True, XYZ_RENDERER="egl", XYZ_BP=True,
        BACKBONE=dict(PRETRAINED="timm", INIT_CFG=dict(
            _delete_=True, type="timm/convnext_base", pretrained=False,
            in_chans=3, features_only=True, out_indices=(3,),
        )),
        GEO_HEAD=dict(FREEZE=True, TRAIN_SUPERVISION=False,
                      MASK_THR_TEST=0.5),
        PCC_HEAD=dict(ENABLED=True, INIT_CFG=PCC_INIT_CFG),
        LOSS_CFG=dict(MASK_LOSS_TYPE="BCE"),
    ),
)
DATASETS = dict(DET_FILES_TEST=())
del PCC_INIT_CFG
