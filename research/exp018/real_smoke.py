"""User-run bounded real-data smoke: one PBR batch, two optimizer steps.

Not the server one-epoch smoke, and never formal training. A unique output
directory is mandatory; existing directories are refused, including failed runs.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import torch
from detectron2.utils.events import EventStorage
from mmcv import Config

from core.gdrn_modeling.datasets.data_loader import (
    build_gdrn_train_loader,
    build_gdrn_test_loader,
)
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine.engine_utils import batch_data, pose_corrector_kwargs
from core.gdrn_modeling.models.GDRN_double_mask import build_model_optimizer
from research.diagnostics.pose_structure.model_access import make_model_kwargs
from research.diagnostics.pose_structure.runtime import set_seed
from research.exp018.preflight import (
    ROOT,
    CONFIG,
    EXPERIMENT_ID,
    PARAMETERS,
    validate_config,
    load_official_state,
    check_optimizer,
)


def run(args, output_dir):
    set_seed(42)
    torch.set_num_threads(4)
    validate_config(Config.fromfile(str(CONFIG)))
    cfg = Config.fromfile(str(CONFIG.with_name("smoke.py")))
    cfg.MODEL.DEVICE = args.device
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.REFERENCE_BS = args.batch_size
    cfg.SOLVER.BASE_LR = float(cfg.SOLVER.OPTIMIZER_CFG.lr)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.PERSISTENT_WORKERS = False
    register_datasets_in_cfg(cfg)
    model, optimizer = build_model_optimizer(cfg)
    load_official_state(model, args.weights)
    check_optimizer(model, optimizer)
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    versions = {n: p._version for n, p in frozen}
    before = {
        n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    loader = build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN)
    batch = batch_data(cfg, next(iter(loader)), renderer=None, device=args.device)
    kwargs = pose_corrector_kwargs(cfg, batch)

    # Same upstream realization, same A state, ordinary production inference decode.
    model.eval()
    with torch.no_grad():
        initial = model(
            batch["roi_img"],
            **make_model_kwargs(batch, False),
            **kwargs,
            return_pose_debug=True,
        )
        corrector = model.pose_corrector
        model.pose_corrector = None
        try:
            base = model(batch["roi_img"], **make_model_kwargs(batch, False))
        finally:
            model.pose_corrector = corrector
        assert torch.equal(initial["rot"].cpu(), base["rot"].cpu())
        assert torch.equal(initial["trans"], base["trans"])
    del initial, base

    model.train()
    history = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with EventStorage(step):
            out, losses = model(
                batch["roi_img"],
                **make_model_kwargs(batch, True),
                **kwargs,
                return_pose_debug=True,
            )
        loss = sum(losses.values())
        assert torch.isfinite(loss), losses
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
        grad = corrector.correction_mlp[-1].weight.grad
        assert (
            grad[:3].abs().sum() > 0 and grad[3:].abs().sum() > 0
        ), "correction R/t has no real gradient"
        if step == 1:
            for name, p in corrector.named_parameters():
                assert (
                    p.grad.abs().sum() > 0
                ), f"second-step internal gradient absent: {name}"
        info = out["pose_debug"]
        history.append(
            dict(
                step=step + 1,
                losses={k: float(v.detach()) for k, v in losses.items()},
                support_count=info["support"].sum(-1).tolist(),
                empty_support=info["empty_support"].tolist(),
                delta_rotvec=info["delta_rotvec"].detach().cpu().tolist(),
                delta_t=info["delta_t"].detach().cpu().tolist(),
                initial_z=info["init_t"][:, 2].detach().cpu().tolist(),
                final_z=info["final_t"][:, 2].detach().cpu().tolist(),
            )
        )
        optimizer.step()
    changed = [
        n
        for n, p in model.named_parameters()
        if n in before and not torch.equal(before[n], p)
    ]
    assert any(n.startswith("pnp_net.") for n in changed)
    assert any(n.startswith("pose_corrector.") for n in changed)
    assert all(p.grad is None and p._version == versions[n] for n, p in frozen)
    assert all(torch.isfinite(p).all() for p in model.parameters())

    # Serialize both trainable branches and Ranger state, then strictly restore
    # them on the same frozen producer (avoid a duplicate ConvNeXt on laptop GPU).
    payload = io.BytesIO()
    torch.save(
        dict(
            pnp_net=model.pnp_net.state_dict(),
            pose_corrector=corrector.state_dict(),
            optimizer=optimizer.state_dict(),
        ),
        payload,
    )
    model.eval()
    with torch.no_grad():
        expected = model(batch["roi_img"], **make_model_kwargs(batch, False), **kwargs)
        payload.seek(0)
        saved = torch.load(payload, map_location=args.device)
        model.pnp_net.load_state_dict(saved["pnp_net"], strict=True)
        corrector.load_state_dict(saved["pose_corrector"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restored = model(
            batch["roi_img"],
            **make_model_kwargs(batch, False),
            **kwargs,
            return_pose_debug=True,
        )
        assert torch.equal(expected["rot"], restored["rot"])
        assert torch.equal(expected["trans"], restored["trans"])
        torch.save(
            {k: v.detach().cpu() for k, v in restored["pose_debug"].items()},
            output_dir / "pose_debug.pt",
        )

    # One LM-O image exercises the actual test mapper/batcher and GPU output
    # path. No BOP metric, renderer, PnP, or checkpoint selection is run.
    eval_cfg = Config.fromfile(str(CONFIG.with_name("eval.py")))
    eval_cfg.DATALOADER.NUM_WORKERS = 0
    register_datasets_in_cfg(eval_cfg)
    test_loader = build_gdrn_test_loader(eval_cfg, "lmo_bop_test")
    test_batch = batch_data(
        eval_cfg, next(iter(test_loader)), device=args.device, phase="test"
    )
    with torch.no_grad():
        test_out = model(
            test_batch["roi_img"],
            **make_model_kwargs(test_batch, False),
            **pose_corrector_kwargs(eval_cfg, test_batch),
        )
    assert (
        torch.isfinite(test_out["rot"]).all()
        and torch.isfinite(test_out["trans"]).all()
    )
    return dict(
        status="PASS",
        experiment_id=EXPERIMENT_ID,
        run_id=output_dir.name,
        formal_training=False,
        server_epoch_smoke=False,
        seed=42,
        device=args.device,
        batch_size=args.batch_size,
        optimizer_steps=2,
        correction_parameters=PARAMETERS,
        base_init_value_exact=True,
        checkpoint_roundtrip_value_exact=True,
        backbone_geometry_parameters_unchanged=True,
        actual_test_mapper_forward=True,
        test_roi_count=len(test_out["trans"]),
        history=history,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_tree_clean=not bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "pretrained_models/lmo_pbr/model_final_wo_optim.pth",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args, output_dir)
    except Exception as exc:
        (output_dir / "result.json").write_text(
            json.dumps(dict(status="FAIL", error=repr(exc)), indent=2)
        )
        raise
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
