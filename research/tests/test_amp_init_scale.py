"""`SOLVER.AMP.INIT_SCALE`: pinning the GradScaler's start without touching AMP itself.

`LightningLite(precision=16)` builds its own GradScaler at torch's default 65536, which is
not always what a model can consume on its first updates, and a config had no handle on it.
The configuration is optional in both directions: unset, every existing experiment keeps
Lite's own scaler; set, the production entry hands Lite a native AMP plugin owning a scaler
created at that value.  Only the initial value changes -- dynamic growth/backoff, the
scaler's checkpoint state and the engine's skipped-step scheduler gate are untouched.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from detectron2.config import CfgNode

from core.utils import solver_utils
from core.utils.my_checkpoint import MyCheckpointer

REPO = Path(__file__).resolve().parents[2]


def _cfg(scale=None, enabled=True):
    cfg = CfgNode()
    cfg.SOLVER = CfgNode()
    cfg.SOLVER.AMP = CfgNode({"ENABLED": enabled})
    if scale is not None:
        cfg.SOLVER.AMP.INIT_SCALE = scale
    return cfg


def test_unset_scale_leaves_the_production_plugin_to_lite():
    """No INIT_SCALE means no plugins argument at all: the historical behaviour."""
    assert solver_utils.amp_precision_plugins(_cfg()) is None
    assert solver_utils.amp_precision_plugins(_cfg(scale=None)) is None
    # The explicit key is what turns the capability on; a config that never mentions it
    # must not gain a key here either.
    assert "INIT_SCALE" not in _cfg().SOLVER.AMP


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GradScaler disables itself without CUDA")
def test_configured_scale_is_used_by_the_plugin():
    plugins = solver_utils.amp_precision_plugins(_cfg(2048.))
    assert plugins is not None and len(plugins) == 1
    plugin = plugins[0]
    assert plugin.precision == 16 and plugin.device == "cuda"
    assert plugin.scaler.get_scale() == pytest.approx(2048.)
    # Growth/backoff are the torch defaults, not something this helper re-implements.
    assert plugin.scaler.get_growth_factor() == 2.0
    assert plugin.scaler.get_backoff_factor() == 0.5


def test_invalid_scale_configuration_fails_closed():
    with pytest.raises(ValueError, match="requires SOLVER.AMP.ENABLED"):
        solver_utils.amp_precision_plugins(_cfg(2048., enabled=False))
    for bad in (0., -1.):
        with pytest.raises(ValueError, match="must be >= 1"):
            solver_utils.amp_precision_plugins(_cfg(bad))


def test_main_entry_hands_the_plugin_to_lite():
    """The wiring the capability depends on, checked where it is written."""
    source = (REPO / "core/gdrn_modeling/main_gdrn.py").read_text(encoding="utf-8")
    call = source.index("Lite(\n")
    body = source[call:source.index(").run(args, cfg)", call)]
    assert "plugins=solver_utils.amp_precision_plugins(cfg)" in body


def test_diagnostics_build_lite_with_the_same_plugin_as_production():
    """The local AMP diagnostics must not silently run a different scaler than production."""
    for name in ("accumulation_smoke.py",):
        source = (REPO / "research/exp025" / name).read_text(encoding="utf-8")
        window = source[source.index("_Lite(accelerator='gpu'"):][:400]
        assert "plugins=solver_utils.amp_precision_plugins(cfg)" in window, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native AMP needs a CUDA device")
def test_production_lite_really_starts_at_the_configured_scale():
    from pytorch_lightning.lite import LightningLite

    class _Lite(LightningLite):
        def run(self):
            pass

    def build(plugins):
        model = torch.nn.Linear(4, 4).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        lite = _Lite(accelerator="gpu", devices=1, precision=16, plugins=plugins)
        model, wrapper = lite.setup(model, optimizer)
        return model, wrapper, lite

    _, _, default_lite = build(None)
    assert default_lite._precision_plugin.scaler.get_scale() == 65536., \
        "unset INIT_SCALE must keep Lite's own default"

    plugins = solver_utils.amp_precision_plugins(_cfg(4096.))
    model, wrapper, lite = build(plugins)
    assert lite._precision_plugin.scaler is plugins[0].scaler, \
        "Lite must train with the scaler the configuration asked for"
    assert lite._precision_plugin.scaler.get_scale() == 4096.

    before = [parameter.detach().clone() for parameter in model.parameters()]
    inputs = torch.randn(3, 4, device="cuda")
    lite.backward(model(inputs).sum())
    wrapper.step()
    assert any(not torch.equal(old, new.detach())
               for old, new in zip(before, model.parameters())), "the pinned scaler must still train"
    assert lite._precision_plugin.scaler.get_scale() == 4096., "a clean step must not back off"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GradScaler disables itself without CUDA")
def test_scaler_state_survives_a_checkpoint_roundtrip(tmp_path):
    """What the engine registers as `gradscaler` must still restore, scale included."""
    scaler = solver_utils.amp_precision_plugins(_cfg(2048.))[0].scaler
    # A used scaler's state carries the scale it backed off to; load it through the public
    # API instead of hand-writing private attributes.
    used = scaler.state_dict()
    used["scale"], used["_growth_tracker"] = torch.tensor(512.), torch.tensor(3)
    scaler.load_state_dict(used)
    assert scaler.get_scale() == 512.
    MyCheckpointer(torch.nn.Linear(2, 2), str(tmp_path), gradscaler=scaler).save(
        "split", iteration=3, epoch=1)

    resumed = solver_utils.amp_precision_plugins(_cfg(2048.))[0].scaler
    assert resumed.get_scale() == 2048.
    MyCheckpointer(torch.nn.Linear(2, 2), str(tmp_path), gradscaler=resumed).resume_or_load(
        "", resume=True)
    assert resumed.get_scale() == 512., "the checkpointed scale must win over INIT_SCALE"
    assert resumed.state_dict() == scaler.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GradScaler disables itself without CUDA")
def test_the_skip_gate_still_reads_a_real_scaler():
    """The engine's scheduler gate only compares scales; a pinned start must not alter it."""
    scaler = solver_utils.amp_precision_plugins(_cfg(4096.))[0].scaler
    plugin = type("P", (), {"scaler": scaler})()
    assert solver_utils.amp_scale(plugin) == 4096.
    assert not solver_utils.gradscaler_skipped_step(4096., 8192.)  # growth is not a skip
    assert solver_utils.gradscaler_skipped_step(4096., 2048.)  # backoff is


def test_engine_gate_and_scheduler_wiring_is_unchanged():
    source = (REPO / "core/gdrn_modeling/engine/engine.py").read_text(encoding="utf-8")
    read_before = source.index("amp_scale_before = solver_utils.amp_scale(self._precision_plugin)")
    step = source.index("optimizer.step()", read_before)
    gate = source.index("if not solver_utils.gradscaler_skipped_step(")
    assert read_before < step < gate < source.index("scheduler.step()", gate)
    # The scaler still reaches the checkpoint through the precision plugin it belongs to.
    assert 'extra_ckpt_dict["gradscaler"] = self._precision_plugin.scaler' in source
