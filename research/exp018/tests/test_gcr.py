from __future__ import annotations

import inspect
import io

import numpy as np
import pytest
import torch
from torch import nn
from mmcv import Config
from detectron2.utils.events import EventStorage

from core.gdrn_modeling.models.heads.gcr_pose_corrector import (
    GeometryConsistencyCorrector,
    corrected_centroid_z,
    so3_exp_map,
)
from core.gdrn_modeling.models.GDRN_double_mask import GDRN_DoubleMask
from core.gdrn_modeling.models.model_utils import get_pnp_net
from core.gdrn_modeling.models.pose_from_pred_centroid_z import (
    pose_from_pred_centroid_z,
)
from core.gdrn_modeling.losses.pm_loss import PyPMLoss
from core.gdrn_modeling.engine.engine_utils import batch_data, pose_corrector_kwargs
from core.utils.data_utils import get_2d_coord_np
from research.exp018.preflight import (
    CONFIG,
    PARAMETERS,
    validate_config,
    synthetic_inputs,
    add_targets,
)
from research.run_contract import validate_research_run_config


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(18)
    torch.set_num_threads(2)


def inputs(batch=2):
    d = synthetic_inputs(batch)
    xyz = torch.rand(batch, 3, 8, 8)
    uv = torch.rand(batch, 2, 8, 8)
    visibility = torch.ones(batch, 1, 8, 8)
    visibility[:, :, :3] = 0
    return [
        xyz,
        uv,
        visibility,
        so3_exp_map(torch.randn(batch, 3) * 0.2),
        torch.tensor([[0.02, -0.04, 0.6]]).repeat(batch, 1),
        d["roi_cams"],
        d["roi_image_hw"],
        d["roi_extents"],
    ]


def trained_corrector():
    model = GeometryConsistencyCorrector()
    nn.init.normal_(model.correction_mlp[-1].weight, std=0.03)
    nn.init.normal_(model.correction_mlp[-1].bias, std=0.02)
    return model


@pytest.mark.parametrize("batch", [1, 3])
def test_zero_init_and_gradients(batch):
    m = GeometryConsistencyCorrector()
    x = inputs(batch)
    x[0].requires_grad_()
    r, t, info = m(*x)
    assert torch.equal(r, x[3]) and torch.equal(t, x[4])
    assert info["raw_delta"].count_nonzero() == 0
    loss = r[:, 0, 1].sum() + t.sum()
    loss.backward()
    grad = m.correction_mlp[-1].weight.grad
    assert grad[:3].abs().sum() > 0 and grad[3:].abs().sum() > 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )
    assert (
        x[0].grad.count_nonzero() == 0
    )  # zero output weights block first-step internals
    torch.optim.SGD(m.parameters(), lr=0.01).step()
    m.zero_grad(set_to_none=True)
    r, t, _ = m(*x)
    (r[:, 0, 1].sum() + t.sum()).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    assert x[0].grad.abs().sum() > 0


@pytest.mark.parametrize("poison", [1e30, float("nan"), float("inf"), -float("inf")])
def test_unsupported_xyz_invariance_after_learning(poison):
    m, x = trained_corrector(), inputs()
    expected = m(*x)
    x[0] = x[0].clone()
    x[0][:, :, :3] = poison
    x[0].requires_grad_()
    actual = m(*x)
    for key in ("descriptor", "token_weights", "raw_delta", "final_R", "final_t"):
        assert torch.equal(expected[2][key], actual[2][key]), key
    (actual[0].sum() + actual[1].sum()).backward()
    assert torch.isfinite(x[0].grad).all() and x[0].grad[:, :, :3].count_nonzero() == 0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_empty_and_mixed_support_after_learning():
    m, x = trained_corrector(), inputs(3)
    x[2][0] = 0
    x[4][1, 2] = -1  # all behind camera
    x[0][2, :, 4, 4] = float("nan")  # invalid point within visible support
    r, t, info = m(*x)
    assert torch.equal(r[:2], x[3][:2]) and torch.equal(t[:2], x[4][:2])
    assert info["empty_support"].tolist() == [True, True, False]
    assert info["token_weights"][:2].count_nonzero() == 0
    assert not info["support"][2, 36]
    torch.testing.assert_close(
        info["token_weights"].sum(-1), torch.tensor([0.0, 0.0, 1.0])
    )
    (r.sum() + t.sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_region_free_api_and_parameter_budget():
    m = GeometryConsistencyCorrector()
    assert "region" not in inspect.signature(m.forward).parameters
    assert not any("region" in name for name, _ in m.named_modules())
    assert sum(p.numel() for p in m.parameters()) == PARAMETERS < 100000
    with pytest.raises(ValueError, match="exactly one"):
        GeometryConsistencyCorrector(num_steps=2)


def test_residual_changes_with_correspondence_and_current_pose():
    m, x = trained_corrector(), inputs(1)
    before = m(*x)[2]
    x[1] = x[1] + 0.05
    changed_uv = m(*x)[2]
    assert not torch.equal(before["residual_norm"], changed_uv["residual_norm"])
    assert not torch.equal(before["raw_delta"], changed_uv["raw_delta"])
    x[4] = x[4] + torch.tensor([[0.03, 0.0, 0.0]])
    changed_pose = m(*x)[2]
    assert not torch.equal(changed_uv["projected_uv"], changed_pose["projected_uv"])
    assert not torch.equal(changed_uv["residual_norm"], changed_pose["residual_norm"])


def test_analytic_projection_metric_and_endpoint_false():
    x = inputs(1)
    coords = torch.from_numpy(get_2d_coord_np(8, 8))[None]
    x[1] = coords
    x[2].fill_(1)
    x[3] = torch.eye(3)[None]
    x[4] = torch.tensor([[0.0, 0.0, 0.8]])
    # Build metric XYZ along these image rays at constant object z=0.
    uv = coords * torch.tensor([640.0, 480.0])[None, :, None, None]
    metric = torch.cat(
        (
            (uv - torch.tensor([320.0, 240.0])[None, :, None, None]) / 572 * 0.8,
            torch.zeros(1, 1, 8, 8),
        ),
        1,
    )
    x[0] = metric / x[-1][:, :, None, None] + 0.5
    _, _, info = GeometryConsistencyCorrector()(*x)
    torch.testing.assert_close(
        info["projected_uv"], uv.flatten(2).transpose(1, 2), atol=7e-5, rtol=0
    )
    assert info["reprojection_residual"].abs().max() < 7e-5
    assert info["observed_uv"][0, -1].tolist() == [560.0, 420.0]


def test_so3_zero_derivative_and_group_properties():
    v = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(so3_exp_map, (v,))
    r = so3_exp_map(torch.randn(8, 3, dtype=torch.float64))
    torch.testing.assert_close(
        r @ r.transpose(1, 2), torch.eye(3, dtype=r.dtype).expand(8, 3, 3)
    )
    torch.testing.assert_close(torch.linalg.det(r), torch.ones(8, dtype=r.dtype))


def test_centroid_z_inverse_and_exact_zero():
    x = synthetic_inputs(3)
    raw = torch.tensor(
        [[0.1, -0.2, 0.8], [-0.3, 0.4, 0.5], [0.2, 0.1, -0.7]], requires_grad=True
    )
    cams, centers, whs, ratios = [
        x[k] for k in ("roi_cams", "roi_centers", "roi_whs", "resize_ratios")
    ]

    def decode(t):
        return pose_from_pred_centroid_z(
            torch.eye(3).repeat(3, 1, 1),
            t[:, :2],
            t[:, 2:3],
            cams,
            centers,
            ratios,
            whs,
            is_allo=False,
            is_train=True,
        )[1]

    t0 = decode(raw)
    delta = torch.tensor([[0.01, -0.01, 0.02]]).repeat(3, 1).requires_grad_()
    raw_final = corrected_centroid_z(raw, delta, t0 + delta, cams, centers, whs, ratios)
    torch.testing.assert_close(decode(raw_final), t0 + delta)
    same = corrected_centroid_z(raw, delta * 0, t0, cams, centers, whs, ratios)
    assert torch.equal(same, raw)
    raw_final.sum().backward()
    assert delta.grad.abs().sum() > 0 and torch.isfinite(raw.grad).all()
    for z in [0.0, 1e-10, -1e-10]:
        edge = raw.detach().clone()
        edge[:, 2] = z
        assert torch.equal(
            corrected_centroid_z(
                edge, delta * 0, decode(edge), cams, centers, whs, ratios
            ),
            edge,
        )


def test_checkpoint_roundtrip_and_batch_independence():
    m, x = trained_corrector(), inputs(3)
    expected = m(*x)
    payload = io.BytesIO()
    torch.save(m.state_dict(), payload)
    payload.seek(0)
    restored = GeometryConsistencyCorrector()
    restored.load_state_dict(torch.load(payload), strict=True)
    actual = restored(*x)
    assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])
    for index in range(3):
        single = restored(*(value[index : index + 1] for value in x))
        torch.testing.assert_close(single[0], actual[0][index : index + 1])
        torch.testing.assert_close(single[1], actual[1][index : index + 1])


class FixedGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("xyz", torch.rand(1, 3, 64, 64))
        self.register_buffer("mask", torch.rand(1, 1, 64, 64))
        self.register_buffer("region", torch.randn(1, 65, 64, 64))

    def forward(self, x):
        b = len(x)
        coords = self.xyz.expand(b, -1, -1, -1)
        mask = self.mask.expand(b, -1, -1, -1)
        return (
            mask,
            mask,
            coords[:, :1],
            coords[:, 1:2],
            coords[:, 2:],
            self.region.expand(b, -1, -1, -1),
        )


def integration_model():
    cfg = Config.fromfile(str(CONFIG))
    cfg.MODEL.POSE_NET.GEO_HEAD.XYZ_CLASS_AWARE = False
    cfg.MODEL.POSE_NET.GEO_HEAD.MASK_CLASS_AWARE = False
    cfg.MODEL.POSE_NET.GEO_HEAD.REGION_CLASS_AWARE = False
    cfg.SOLVER.BASE_LR = 0.0008
    head, _ = get_pnp_net(cfg)
    # Only the synthetic fixture fixes a sensible positive initial depth to
    # exercise correction gradients; production A initialization is unchanged.
    with torch.no_grad():
        head.pose_translation.bias[2] = 0.7
    return GDRN_DoubleMask(
        cfg,
        nn.Identity(),
        FixedGeometry(),
        pnp_net=head,
        pose_corrector=GeometryConsistencyCorrector(),
    )


@pytest.mark.parametrize("batch", [1, 3])
def test_real_model_forward_loss_and_checkpoint(batch):
    model = integration_model().eval()
    x = synthetic_inputs(batch)
    targets = add_targets(x)
    with EventStorage():
        actual, losses = model(**targets, return_pose_debug=True)
        corrector = model.pose_corrector
        model.pose_corrector = None
        try:
            _, base_losses = model(**targets)
            expected = model(**x)
        finally:
            model.pose_corrector = corrector
        actual_eval = model(**x, return_pose_debug=True)
    assert torch.equal(actual_eval["rot"], expected["rot"])
    assert torch.equal(actual_eval["trans"], expected["trans"])
    assert (
        losses.keys() == base_losses.keys() == {"loss_PM_R", "loss_centroid", "loss_z"}
    )
    for key in losses:
        assert torch.equal(losses[key], base_losses[key]), key
    grad_r = torch.autograd.grad(
        losses["loss_PM_R"], corrector.correction_mlp[-1].weight, retain_graph=True
    )[0]
    grad_t = torch.autograd.grad(
        losses["loss_centroid"] + losses["loss_z"],
        corrector.correction_mlp[-1].weight,
        retain_graph=True,
    )[0]
    assert grad_r[:3].abs().sum() > 0
    assert grad_t[3:].abs().sum() > 0 and grad_t[:3].count_nonzero() == 0
    sum(losses.values()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    torch.optim.SGD(model.parameters(), lr=0.001).step()
    with torch.no_grad():
        expected = model(**x)
        payload = io.BytesIO()
        torch.save(model.state_dict(), payload)
        payload.seek(0)
        restored = integration_model().eval()
        restored.load_state_dict(torch.load(payload), strict=True)
        output = restored(**x)
    assert torch.equal(expected["rot"], output["rot"])
    assert torch.equal(expected["trans"], output["trans"])


def test_symmetry_uses_existing_final_pose_pm():
    r = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))[None]
    points = torch.randn(1, 100, 3)
    loss = PyPMLoss(symmetric=True, r_only=True)
    sym = [np.stack([np.eye(3), r[0].numpy()])]
    result = loss(r, torch.eye(3)[None], points, sym_infos=sym)
    assert result["loss_PM_R"] == 0
    result = PyPMLoss(symmetric=False, r_only=True)(r, torch.eye(3)[None], points)
    assert result["loss_PM_R"] > 0


def test_region_shuffle_with_initial_pose_held_fixed():
    model = integration_model().eval()
    model.pose_corrector = trained_corrector()
    x = synthetic_inputs(1)
    raw = []
    handle = model.pnp_net.register_forward_hook(
        lambda m, i, o: raw.append(tuple(v.detach().clone() for v in o))
    )
    with torch.no_grad():
        before = model(**x, return_pose_debug=True)
    handle.remove()
    # Causal boundary: freeze R0/t0 while changing Region upstream. It is NOT
    # valid to demand Region-invariance if Region also changes the initial pose.
    handle = model.pnp_net.register_forward_hook(lambda m, i, o: raw[0])
    try:
        model.geo_head_net.region.copy_(model.geo_head_net.region.flip(1))
        with torch.no_grad():
            after = model(**x, return_pose_debug=True)
    finally:
        handle.remove()
    for key in ("descriptor", "token_weights", "raw_delta", "final_R", "final_t"):
        assert torch.equal(before["pose_debug"][key], after["pose_debug"][key])


def test_config_and_batch_metadata_contract():
    cfg = Config.fromfile(str(CONFIG))
    validate_config(cfg)
    for mode, filename in (("smoke", "smoke.py"), ("eval", "eval.py")):
        validate_research_run_config(
            Config.fromfile(str(CONFIG.with_name(filename))), mode=mode
        )
    b = 2
    data = [
        dict(
            roi_img=torch.zeros(3, 4, 4),
            roi_cls=i,
            cam=torch.eye(3),
            bbox_center=torch.zeros(2),
            roi_wh=torch.ones(2),
            resize_ratio=1.0,
            roi_extent=torch.ones(3),
            trans_ratio=torch.ones(3),
            roi_image_hw=torch.tensor([480.0, 640.0]) + i,
        )
        for i in range(b)
    ]
    batch = batch_data(cfg, data, device="cpu")
    assert torch.equal(
        batch["roi_image_hw"], torch.stack([d["roi_image_hw"] for d in data])
    )
    assert pose_corrector_kwargs(cfg, batch).keys() == {"roi_image_hw"}
    test_data = [
        dict(
            roi_img=torch.zeros(b, 3, 4, 4),
            cam=torch.eye(3).repeat(b, 1, 1),
            bbox_center=torch.zeros(b, 2),
            im_H=torch.tensor([480.0, 600.0]),
            im_W=torch.tensor([640.0, 800.0]),
        )
    ]
    test_batch = batch_data(cfg, test_data, device="cpu", phase="test")
    assert test_batch["roi_image_hw"].tolist() == [[480.0, 640.0], [600.0, 800.0]]
    cfg.MODEL.POSE_NET.POSE_CORRECTOR.ENABLED = False
    assert pose_corrector_kwargs(cfg, {}) == {}


def test_cpu_autocast_float32_island():
    m, x = trained_corrector(), inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        r, t, info = m(*x)
        assert r.dtype == t.dtype == torch.float32
        (r.sum() + t.sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_smoke_orchestration_with_synthetic_loaders_only(monkeypatch, tmp_path):
    # No dataset registration/read and no real producer: exercise the user-run
    # script's control flow separately from the pending real-data smoke.
    from types import SimpleNamespace
    from research.exp018 import real_smoke

    x = synthetic_inputs(2)
    targets = add_targets(x)
    data = []
    for i in range(2):
        data.append(
            dict(
                roi_img=x["x"][i],
                roi_cls=x["roi_classes"][i],
                cam=x["roi_cams"][i],
                bbox_center=x["roi_centers"][i],
                roi_wh=x["roi_whs"][i],
                resize_ratio=x["resize_ratios"][i],
                roi_extent=x["roi_extents"][i],
                roi_coord_2d=x["roi_coord_2d"][i],
                roi_image_hw=x["roi_image_hw"][i],
                trans_ratio=targets["gt_trans_ratio"][i],
                trans=targets["gt_trans"][i],
                ego_rot=targets["gt_ego_rot"][i],
                roi_points=targets["gt_points"][i],
                sym_info=None,
            )
        )
    test_data = {
        key: torch.stack([d[key] for d in data])
        for key in (
            "roi_img",
            "roi_cls",
            "cam",
            "bbox_center",
            "roi_wh",
            "resize_ratio",
            "roi_extent",
            "roi_coord_2d",
        )
    }
    test_data.update(
        im_H=torch.tensor([480.0, 480.0]), im_W=torch.tensor([640.0, 640.0])
    )
    monkeypatch.setattr(real_smoke, "register_datasets_in_cfg", lambda cfg: None)
    monkeypatch.setattr(real_smoke, "build_gdrn_train_loader", lambda *a, **k: [data])
    monkeypatch.setattr(
        real_smoke, "build_gdrn_test_loader", lambda *a, **k: [[test_data]]
    )
    monkeypatch.setattr(real_smoke, "load_official_state", lambda *a: None)

    def build(cfg):
        m = integration_model()
        return m, torch.optim.Adam(m.parameters(), lr=0.0008)

    monkeypatch.setattr(real_smoke, "build_model_optimizer", build)
    result = real_smoke.run(
        SimpleNamespace(device="cpu", batch_size=2, weights=None), tmp_path
    )
    assert result["status"] == "PASS" and result["optimizer_steps"] == 2
    assert (tmp_path / "pose_debug.pt").is_file()


def test_actual_online_mapper_resized_image_metadata(monkeypatch):
    from types import SimpleNamespace
    import pycocotools.mask as mask_util
    from detectron2.data import transforms as T
    from detectron2.structures import BoxMode
    from core.gdrn_modeling.datasets import data_loader_online as online

    cfg = Config.fromfile(str(CONFIG))
    cfg.INPUT.CHANGE_BG_PROB = 0
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    monkeypatch.setattr(online, "read_image_mmcv", lambda *a, **k: image.copy())
    mapper = SimpleNamespace(
        cfg=cfg,
        split="train",
        img_format="BGR",
        with_depth=False,
        head_depth=False,
        color_aug_prob=0,
        color_augmentor=None,
        augmentation=[T.Resize((480, 640))],
        flatten=True,
        _get_extents=lambda name: [np.array([0.1, 0.12, 0.14])],
        _get_fps_points=lambda name: [np.zeros((64, 3))],
        _get_model_points=lambda name: [np.zeros((100, 3))],
        _get_sym_infos=lambda name: [None],
        aug_bbox_DZI=lambda *a: (np.array([320.0, 240.0]), 160.0),
        normalize_image=lambda cfg, img: img,
    )
    mask = mask_util.encode(np.asfortranarray(np.ones((240, 320), dtype=np.uint8)))
    k = np.array([[286.0, 0, 160.0], [0, 286.0, 120.0], [0, 0, 1.0]])
    annotation = dict(
        category_id=0,
        bbox=[40, 30, 240, 180],
        bbox_obj=[40, 30, 240, 180],
        bbox_mode=BoxMode.XYWH_ABS,
        segmentation=mask,
        centroid_2d=[160.0, 120.0],
        pose=np.concatenate((np.eye(3), np.array([[0.0], [0.0], [0.7]])), 1),
        trans=np.array([0.0, 0.0, 0.7]),
    )
    sample = dict(
        file_name="synthetic-only.png",
        height=240,
        width=320,
        dataset_name="synthetic",
        img_type="real",
        cam=k,
        inst_infos=annotation,
    )
    result = online.GDRN_Online_DatasetFromList.read_data_train(mapper, sample)
    assert result["roi_image_hw"].tolist() == [480.0, 640.0]
    assert result["cam"][0, 0] == 572 and result["cam"][0, 2] == 320
    # Central ROI grid entry corresponds to centre pixel of the RESIZED image.
    torch.testing.assert_close(
        result["roi_coord_2d"][:, 32, 32], torch.tensor([0.5, 0.5])
    )
