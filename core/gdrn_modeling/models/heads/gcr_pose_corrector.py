"""EXP018: one camera-frame, Region-free geometry-consistency correction.

Coordinates follow the repository: XYZ=(prediction-.5)*extent (metres),
ROI2D=(u/W,v/H), NOT endpoint-inclusive coordinates or crop-local pixels.
No ground-truth correspondence or reprojection loss is used.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def so3_exp_map(vector):
    """Exact matrix exponential of a skew matrix, finite derivative at zero.

    The legacy lie_vec_to_rot evaluates sqrt(0) in its inactive branch and
    has a first-order non-SO(3) fallback; do not use it for zero-init updates.
    Only B small 3x3 matrices are exponentiated (not the correspondence grid).
    """
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    return torch.matrix_exp(skew.reshape(-1, 3, 3))


def corrected_centroid_z(raw_t, delta_t, final_t, cams, centers, whs, ratios):
    """Encode corrected camera translation in A's centroid + relative-z units.

    Algebraic incremental inverse of pose_from_pred_centroid_z (REL). This
    preserves raw_t exactly for zero correction, including z0=0. At the camera
    plane the centroid is undefined; use a signed 1um denominator safeguard.
    """
    focal = torch.stack((cams[:, 0, 0], cams[:, 1, 1]), -1)
    principal = cams[:, :2, 2]
    ray_xy = (raw_t[:, :2] * whs + centers - principal) / focal
    z = final_t[:, 2:3]
    safe_z = torch.where(z < 0, -z.abs().clamp_min(1e-6), z.abs().clamp_min(1e-6))
    shift = focal * (delta_t[:, :2] - ray_xy * delta_t[:, 2:3]) / safe_z / whs
    return torch.cat(
        (raw_t[:, :2] + shift, raw_t[:, 2:3] + delta_t[:, 2:3] / ratios.reshape(-1, 1)),
        -1,
    )


class GeometryConsistencyCorrector(nn.Module):
    """9->64->64 tokens, masked scores, 9->32 pose context, 96->64->6.

    Empty support is an identity update even after learning nonzero biases.
    Nonfinite/unsupported points are sanitized BEFORE any projection or MLP.
    The support-invariance contract holds with initial pose fixed: Region can
    still indirectly influence correction through EXP013A's initial pose.
    """

    def __init__(
        self,
        num_steps=1,
        support_threshold=0.5,
        residual_clip=2.0,
        max_rotation_deg=15.0,
        translation_extent_scale=0.15,
    ):
        super().__init__()
        if num_steps != 1:
            raise ValueError("EXP018 requires exactly one correction")
        if not 0 <= support_threshold < 1:
            raise ValueError("support_threshold must be in [0, 1)")
        if min(residual_clip, max_rotation_deg, translation_extent_scale) <= 0:
            raise ValueError("EXP018 correction scales must be positive")
        self.support_threshold = support_threshold
        self.residual_clip = residual_clip
        # Per-axis bound; vector norm can reach sqrt(3)*15 degrees.
        self.rotation_scale = math.radians(max_rotation_deg)
        self.translation_scale = translation_extent_scale
        self.token_encoder = nn.Sequential(
            nn.Linear(9, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
        )
        self.score_mlp = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.pose_encoder = nn.Sequential(nn.Linear(9, 32), nn.GELU())
        self.correction_mlp = nn.Sequential(
            nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 6)
        )
        nn.init.zeros_(self.correction_mlp[-1].weight)
        nn.init.zeros_(self.correction_mlp[-1].bias)

    def forward(
        self, xyz_norm, roi2d, visibility, init_R, init_t, cams, image_hw, extents
    ):
        # Explicit float32 island under AMP; preserve float64 for numerical tests.
        with torch.autocast(device_type=xyz_norm.device.type, enabled=False):
            dtype = torch.float64 if xyz_norm.dtype == torch.float64 else torch.float32
            values = [
                xyz_norm,
                roi2d,
                visibility,
                init_R,
                init_t,
                cams,
                image_hw,
                extents,
            ]
            if any(value is None for value in values):
                raise ValueError(
                    "EXP018 needs XYZ, ROI2D, visibility, pose, K, actual image_hw and extent"
                )
            values = [value.to(device=xyz_norm.device, dtype=dtype) for value in values]
            return self._forward(*values)

    def _forward(
        self, xyz_norm, roi2d, visibility, init_R, init_t, cams, image_hw, extents
    ):
        b, c, h, w = xyz_norm.shape
        if c != 3 or roi2d.shape != (b, 2, h, w) or visibility.shape != (b, 1, h, w):
            raise ValueError("EXP018 expects aligned XYZ3 / ROI2D2 / visibility1 BCHW")
        for value, shape in (
            (init_R, (b, 3, 3)),
            (init_t, (b, 3)),
            (cams, (b, 3, 3)),
            (image_hw, (b, 2)),
            (extents, (b, 3)),
        ):
            if value.shape != shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid EXP018 pose/camera/size/extent metadata")
        if not bool((extents > 0).all() and (image_hw > 0).all()):
            raise ValueError("Image dimensions and metric extents must be positive")
        xyz = xyz_norm.flatten(2).transpose(1, 2)
        uv = roi2d.flatten(2).transpose(1, 2)
        conf = torch.nan_to_num(
            visibility.flatten(1), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0, 1)
        candidate = (
            (conf > self.support_threshold)
            & torch.isfinite(xyz).all(-1)
            & torch.isfinite(uv).all(-1)
        )
        # Mask normalized coordinates before denormalization; avoid 0*NaN/Inf.
        xyz = torch.where(candidate[..., None], xyz, torch.full_like(xyz, 0.5))
        uv = torch.where(candidate[..., None], uv, torch.zeros_like(uv))
        xyz_m = (xyz - 0.5) * extents[:, None]
        camera_xyz = torch.bmm(xyz_m, init_R.transpose(1, 2)) + init_t[:, None]
        valid = (
            candidate & torch.isfinite(camera_xyz).all(-1) & (camera_xyz[..., 2] > 1e-6)
        )
        # Sanitize again before divisions, including unsupported behind-camera points.
        fallback = torch.zeros_like(camera_xyz)
        fallback[..., 2] = 1
        camera_xyz = torch.where(valid[..., None], camera_xyz, fallback)
        projected = torch.bmm(camera_xyz, cams.transpose(1, 2))
        uv_hat = projected[..., :2] / camera_xyz[..., 2:3]
        scale = image_hw.flip(-1)[:, None]  # (W,H), endpoint=False
        residual = uv - uv_hat / scale
        extent_mean = extents.mean(-1, keepdim=True)
        raw_token = torch.cat(
            (
                xyz - 0.5,
                uv,
                residual.clamp(-self.residual_clip, self.residual_clip),
                camera_xyz[..., 2:3] / extent_mean[:, None],
                conf[..., None],
            ),
            -1,
        )
        raw_token = torch.where(
            valid[..., None], raw_token, torch.zeros_like(raw_token)
        )
        token = self.token_encoder(raw_token)
        logits = self.score_mlp(token).squeeze(-1)
        weights = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(-1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        descriptor = (token * weights[..., None]).sum(1)
        rot6 = init_R[:, :, :2].transpose(1, 2).reshape(b, 6)
        pose_context = self.pose_encoder(torch.cat((rot6, init_t / extent_mean), -1))
        raw_delta = self.correction_mlp(torch.cat((descriptor, pose_context), -1))
        has_support = valid.any(-1, keepdim=True)
        raw_delta = torch.where(has_support, raw_delta, torch.zeros_like(raw_delta))
        delta_rotvec = raw_delta[:, :3].tanh() * self.rotation_scale
        delta_t = raw_delta[:, 3:].tanh() * (self.translation_scale * extent_mean)
        delta_R = so3_exp_map(delta_rotvec)
        final_R, final_t = delta_R @ init_R, init_t + delta_t
        # Returned only when requested by the caller; do not retain autograd graphs on self.
        debug = dict(
            init_R=init_R,
            init_t=init_t,
            projected_uv=torch.where(
                valid[..., None], uv_hat, torch.zeros_like(uv_hat)
            ),
            observed_uv=uv * scale,
            reprojection_residual=torch.where(
                valid[..., None], residual * scale, torch.zeros_like(residual)
            ),
            residual_norm=raw_token[..., 5:7],
            support=valid,
            token_weights=weights,
            descriptor=descriptor,
            raw_delta=raw_delta,
            delta_rotvec=delta_rotvec,
            delta_R=delta_R,
            delta_t=delta_t,
            final_R=final_R,
            final_t=final_t,
            empty_support=~has_support[:, 0],
        )
        return final_R, final_t, debug
