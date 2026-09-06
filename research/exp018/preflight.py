"""CPU, synthetic-input EXP018 preflight; never consumes training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from detectron2.utils.events import EventStorage
from mmcv import Config

from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from research.exp013.preflight import checkpoint_model_state, synthetic_full_inputs
from research.run_contract import validate_research_run_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/gdrn/lmo_pbr/research/exp018/gcr_pose/train.py"
A_CONFIG = ROOT / "configs/gdrn/lmo_pbr/research/exp013/a_xyz_residual/train.py"
EXPERIMENT_ID = "EXP-20260906-018-geometry-consistency-residual"
PARAMETERS = 13831


def validate_config(cfg):
    result = validate_research_run_config(
        cfg, mode="formal", expected_experiment_id=EXPERIMENT_ID
    )
    a = Config.fromfile(str(A_CONFIG)).to_dict()
    current = cfg.to_dict()
    correction = current["MODEL"]["POSE_NET"].pop("POSE_CORRECTOR")
    assert correction == dict(
        ENABLED=True,
        INIT_CFG=dict(
            type="GeometryConsistencyCorrector",
            num_steps=1,
            support_threshold=0.5,
            residual_clip=2.0,
            max_rotation_deg=15.0,
            translation_extent_scale=0.15,
        ),
    )
    for key in ("EXPERIMENT_ID", "OUTPUT_DIR"):
        current.pop(key)
        a.pop(key)
    assert (
        current == a
    ), "EXP018 changed configuration outside its single correction branch"
    return result


def load_official_state(model, path):
    state = checkpoint_model_state(Path(path))
    result = model.load_state_dict(dict(state), strict=False)
    expected = {
        f"{prefix}.{key}"
        for prefix in ("pnp_net", "pose_corrector")
        for key in getattr(model, prefix).state_dict()
    }
    assert set(result.missing_keys) == expected and not result.unexpected_keys, result
    # Check all inherited producer tensors, not just missing-key counts.
    for name, value in model.state_dict().items():
        if name.startswith(("backbone.", "geo_head_net.")):
            assert torch.equal(value.cpu(), state[name].cpu()), name


def synthetic_inputs(batch=1, device="cpu"):
    inputs = synthetic_full_inputs(torch.device(device), batch)
    inputs["roi_image_hw"] = torch.tensor([[480.0, 640.0]], device=device).repeat(
        batch, 1
    )
    # Non-centred rays, variable crop scaling and extent: don't test only K-centre identities.
    inputs["roi_centers"] += torch.tensor([[17.0, -11.0]], device=device)
    inputs["resize_ratios"] = torch.linspace(0.6, 1.2, batch, device=device)
    return inputs


def add_targets(inputs):
    b = inputs["x"].shape[0]
    device = inputs["x"].device
    trans = torch.tensor([[0.03, -0.02, 0.7]], device=device).repeat(b, 1)
    cams = inputs["roi_cams"]
    uv = torch.stack(
        (
            cams[:, 0, 0] * trans[:, 0] / trans[:, 2] + cams[:, 0, 2],
            cams[:, 1, 1] * trans[:, 1] / trans[:, 2] + cams[:, 1, 2],
        ),
        -1,
    )
    ratio = torch.cat(
        (
            (uv - inputs["roi_centers"]) / inputs["roi_whs"],
            trans[:, 2:3] / inputs["resize_ratios"][:, None],
        ),
        -1,
    )
    return dict(
        inputs,
        gt_trans=trans,
        gt_trans_ratio=ratio,
        gt_ego_rot=torch.eye(3, device=device).repeat(b, 1, 1),
        gt_points=torch.randn(b, 100, 3, device=device) * 0.04,
        sym_infos=[None] * b,
        do_loss=True,
    )


def check_optimizer(model, optimizer):
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    registered = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert set(registered) == trainable and len(registered) == len(trainable)
    assert all(
        name.startswith(("pnp_net.", "pose_corrector."))
        for name, p in model.named_parameters()
        if p.requires_grad
    )
    assert (
        sum(p.numel() for p in model.pose_corrector.parameters()) == PARAMETERS < 100000
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    )
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = Config.fromfile(str(CONFIG))
    contract = validate_config(cfg)
    cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    model, optimizer = build_model_optimizer(cfg)
    load_official_state(model, args.weights)
    check_optimizer(model, optimizer)
    model.eval()
    inputs = synthetic_inputs()
    with torch.no_grad():
        output = model(**inputs, return_pose_debug=True)
        corrector = model.pose_corrector
        model.pose_corrector = None
        try:
            base = model(**inputs)
        finally:
            model.pose_corrector = corrector
        assert torch.equal(output["rot"], base["rot"])
        assert torch.equal(output["trans"], base["trans"])
    with EventStorage():
        out, losses = model(**add_targets(inputs), return_pose_debug=True)
    loss = sum(losses.values())
    assert set(losses) == {"loss_PM_R", "loss_centroid", "loss_z"}
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, name
    grad = corrector.correction_mlp[-1].weight.grad
    assert grad[:3].abs().sum() > 0 and grad[3:].abs().sum() > 0
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    print(
        json.dumps(
            dict(
                status="PASS",
                device="cpu",
                real_data=False,
                formal_training=False,
                correction_parameters=PARAMETERS,
                base_init_value_exact=True,
                official_producer_load_exact=True,
                optimizer_membership_exact=True,
                actual_pose_loss_backward_step=True,
                valid_tokens=out["pose_debug"]["support"].sum(-1).tolist(),
                losses={k: float(v.detach()) for k, v in losses.items()},
                contract=contract,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
