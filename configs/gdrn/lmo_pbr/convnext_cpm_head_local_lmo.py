_base_ = ["./convnext_stage3c0_pnp_only_local_lmo.py"]

OUTPUT_DIR = "output/cpm_head/local_integration"

MODEL = dict(
    POSE_NET=dict(
        BACKBONE=dict(
            FREEZE=True,
            INIT_CFG=dict(pretrained=False),
        ),
        GEO_HEAD=dict(FREEZE=True),
        PNP_NET=dict(
            FREEZE=False,
            INIT_CFG=dict(
                _delete_=True,
                type="CorrespondenceAwareMomentPnPNet",
                hidden_dim=512,
                latent_dim=256,
                denormalize_by_extent=True,
                eps=1e-6,
                # Frozen before training from the 8192-sample PBR moment audit
                # (2026-08-09), in (mu_X, mu_U, C_XX, C_UU, C_XU) order.
                # Each group is divided by its raw P95 absolute value.
                moment_scales=(
                    0.053791501000523545,
                    0.8383049368858337,
                    0.00045910440967418244,
                    0.03372693955898269,
                    0.0008616717066615817,
                ),
                use_cross_covariance=True,
            ),
            WITH_2D_COORD=True,
            COORD_2D_TYPE="abs",
            REGION_ATTENTION=True,
            MASK_ATTENTION="mul",
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
        ),
        QUALITY_COVERAGE=dict(ENABLED=False),
    ),
)
