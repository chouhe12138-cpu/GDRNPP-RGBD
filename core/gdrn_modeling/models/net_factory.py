from .backbone_factory import BACKBONES

from .necks.fpn import FPN
from .heads.fpn_mask_xyz_region_head import FPNMaskXyzRegionHead
from .heads.top_down_mask_xyz_region_head import TopDownMaskXyzRegionHead
from .heads.top_down_mask_xyz_head import TopDownMaskXyzHead
from .heads.top_down_doublemask_xyz_region_head import (
    TopDownDoubleMaskXyzRegionHead,
)
from .heads.conv_mask_xyz_region_head import ConvMaskXyzRegionHead
from .heads.conv_pnp_net import ConvPnPNet
from .heads.cpm_pnp_net import CorrespondenceAwareMomentPnPNet
from .heads.hierarchical_corr_pnp_net import HierarchicalCorrespondencePnPNet
from .heads.exp013_geometry_pnp_net import (
    GeometryAttentionResidualPnPNet,
    RTDecoupledGeometryPnPNet,
    XYZResidualBypassPnPNet,
)
from .heads.glm_pose_net import GLMPoseLNet
from .heads.exp017_rotation_residual_pnp_net import (
    DetachedSupportAwareRotationResidualPnPNet,
    SupportAwareRotationResidualPnPNet,
)
from .heads.official_head_random_init import OfficialConvPnPNetRandomInit
from .heads.conv_pnp_net_no_region import ConvPnPNetNoRegion
from .heads.conv_pnp_net_cls import ConvPnPNetCls
from .heads.point_pnp_net import SimplePointPnPNet
from .heads.gcr_pose_corrector import GeometryConsistencyCorrector

from .fusenets.conv_fuse_net import ConvFuseNet

# -------------------------------------------------------------------------------
NECKS = {"FPN": FPN}

# -------------------------------------------------------------------------------
HEADS = {
    # mask-xyz-region
    "TopDownDoubleMaskXyzRegionHead": TopDownDoubleMaskXyzRegionHead,
    "TopDownMaskXyzRegionHead": TopDownMaskXyzRegionHead,
    "ConvMaskXyzRegionHead": ConvMaskXyzRegionHead,
    "FPNMaskXyzRegionHead": FPNMaskXyzRegionHead,
    "TopDownMaskXyzHead": TopDownMaskXyzHead,
    # pnp net
    "ConvPnPNet": ConvPnPNet,
    "CorrespondenceAwareMomentPnPNet": CorrespondenceAwareMomentPnPNet,
    "HierarchicalCorrespondencePnPNet": HierarchicalCorrespondencePnPNet,
    "XYZResidualBypassPnPNet": XYZResidualBypassPnPNet,
    "GeometryAttentionResidualPnPNet": GeometryAttentionResidualPnPNet,
    "RTDecoupledGeometryPnPNet": RTDecoupledGeometryPnPNet,
    "GLMPoseLNet": GLMPoseLNet,
    "SupportAwareRotationResidualPnPNet": SupportAwareRotationResidualPnPNet,
    "DetachedSupportAwareRotationResidualPnPNet": (
        DetachedSupportAwareRotationResidualPnPNet
    ),
    "OfficialConvPnPNetRandomInit": OfficialConvPnPNetRandomInit,
    "ConvPnPNetNoRegion": ConvPnPNetNoRegion,
    "ConvPnPNetCls": ConvPnPNetCls,
    "SimplePointPnPNet": SimplePointPnPNet,
}

FUSENETS = {
    "ConvFuseNet": ConvFuseNet,
}

# Post-decode modules are not raw PnP heads; keep their registry separate.
POSE_CORRECTORS = {"GeometryConsistencyCorrector": GeometryConsistencyCorrector}
