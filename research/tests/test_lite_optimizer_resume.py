"""LightningLite optimizer boundary: a resume must restore the optimizer that trains.

A Lite wrapper passes `isinstance(..., torch.optim.Optimizer)` but steps an inner
optimizer, and it inherits `torch.optim.Optimizer.load_state_dict`, which restores
onto the wrapper only.  Registering the wrapper used to save correct state and then
silently drop it: momentum restarted and the step counter reset while the scheduler
and GradScaler still advanced.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from detectron2.config import CfgNode
from pytorch_lightning.lite import LightningLite

from core.utils import solver_utils
from core.utils.my_checkpoint import (
    MyCheckpointer,
    resync_wrapped_optimizer,
    unwrap_optimizer_for_checkpoint,
)

TOTAL_UPDATES = 26
SPLIT_UPDATES = 13


class _Lite(LightningLite):
    def run(self):
        pass


def _solver_cfg():
    cfg = CfgNode()
    cfg.SOLVER = CfgNode()
    cfg.SOLVER.LR_SCHEDULER_NAME = "flat_and_anneal"
    cfg.SOLVER.WARMUP_RATIO = 0.25
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.ANNEAL_METHOD = "cosine"
    cfg.SOLVER.TARGET_LR_FACTOR = 0.01
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.REL_STEPS = [2.0 / 3.0, 8.0 / 9.0]
    return cfg


def _batch():
    generator = torch.Generator().manual_seed(7)
    return (torch.randn(32, 8, generator=generator),
            torch.randn(32, 4, generator=generator))


def _raw_model():
    torch.manual_seed(20260920)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4))


def _weights(model):
    # `_LiteModule` registers the wrapped module, so unwrapped keys carry no prefix.
    return {key.replace("_module.", ""): value for key, value in model.state_dict().items()}


def _build(initial_state=None):
    model = _raw_model()
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    lite = _Lite(accelerator="cpu", devices=1)
    model, wrapper = lite.setup(model, optimizer)
    state_optimizer = unwrap_optimizer_for_checkpoint(wrapper)
    scheduler = solver_utils.build_lr_scheduler(_solver_cfg(), state_optimizer, total_iters=TOTAL_UPDATES)
    return model, wrapper, state_optimizer, scheduler


def _update(model, wrapper, scheduler, batch):
    inputs, targets = batch
    wrapper.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    wrapper.step()
    scheduler.step()


def _steps(optimizer):
    return sorted({int(state["step"]) for state in optimizer.state.values() if "step" in state})


def _compare(left, right, path="state"):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), (path, left.keys(), right.keys())
        for key in left:
            _compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _compare(a, b, f"{path}[{index}]")
    elif torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=1e-7, msg=path)
    else:
        assert left == right, (path, left, right)


def test_unwrap_rejects_unknown_and_ambiguous_objects():
    with pytest.raises(TypeError, match="Cannot checkpoint"):
        unwrap_optimizer_for_checkpoint(object())

    class Ambiguous:
        def __init__(self):
            self._optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=.1)
            self.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=.1)

    with pytest.raises(TypeError, match="Ambiguous"):
        unwrap_optimizer_for_checkpoint(Ambiguous())


def test_unwrap_keeps_plain_optimizer_and_detects_group_drift():
    plain = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=3e-4)
    assert unwrap_optimizer_for_checkpoint(plain) is plain
    resync_wrapped_optimizer(plain)  # no-op, must not raise

    class Drifted:
        def __init__(self, inner, param):
            self._optimizer = inner
            self.param_groups = [dict(params=[param])]

    inner = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=3e-4)
    with pytest.raises(RuntimeError, match="param_groups disagree"):
        unwrap_optimizer_for_checkpoint(Drifted(inner, torch.nn.Parameter(torch.zeros(1))))


def test_wrapper_load_state_dict_never_reaches_the_training_optimizer():
    """The failing behaviour the helper exists to avoid."""
    model, wrapper, state_optimizer, scheduler = _build()
    for _ in range(3):
        _update(model, wrapper, scheduler, _batch())
    assert _steps(state_optimizer) == [3]

    _, fresh_wrapper, _, _ = _build()
    wrapper.load_state_dict(fresh_wrapper.state_dict())  # inherited torch implementation
    assert _steps(state_optimizer) == [3], "the optimizer that steps must own the restored state"
    assert _steps(wrapper) == [], "only the wrapper's own view was replaced"


def test_resume_matches_continuous_training(tmp_path):
    batch = _batch()
    initial_state = {key: value.clone() for key, value in _raw_model().state_dict().items()}

    # A: one uninterrupted run.
    model_a, wrapper_a, state_a, scheduler_a = _build(initial_state)
    for _ in range(TOTAL_UPDATES):
        _update(model_a, wrapper_a, scheduler_a, batch)

    # B: half the run, checkpoint through MyCheckpointer, resume, finish.
    model_b, wrapper_b, state_b, scheduler_b = _build(initial_state)
    for _ in range(SPLIT_UPDATES):
        _update(model_b, wrapper_b, scheduler_b, batch)
    checkpointer = MyCheckpointer(
        model_b,
        str(tmp_path),
        optimizer=state_b,
        scheduler=scheduler_b,
    )
    checkpointer.save("split", iteration=SPLIT_UPDATES - 1, epoch=1)
    assert (tmp_path / "last_checkpoint").is_file()

    model_c, wrapper_c, state_c, scheduler_c = _build()
    resumed = MyCheckpointer(
        model_c,
        str(tmp_path),
        optimizer=state_c,
        scheduler=scheduler_c,
    ).resume_or_load("", resume=True)
    resync_wrapped_optimizer(wrapper_c)
    assert resumed["iteration"] == SPLIT_UPDATES - 1
    # The wrapper steps the inner optimizer, so the inner one owns the resumed state.
    assert _steps(state_c) == [SPLIT_UPDATES]
    assert wrapper_c.param_groups is state_c.param_groups
    assert wrapper_c.state is state_c.state
    for _ in range(SPLIT_UPDATES, TOTAL_UPDATES):
        _update(model_c, wrapper_c, scheduler_c, batch)

    _compare(_weights(model_a), _weights(model_c), "model")
    _compare(state_a.state_dict(), state_c.state_dict(), "optimizer")
    _compare(scheduler_a.state_dict(), scheduler_c.state_dict(), "scheduler")
    assert _steps(state_a) == _steps(state_c) == [TOTAL_UPDATES]
    assert not torch.equal(initial_state["0.weight"], _weights(model_a)["0.weight"])
    assert not torch.equal(_weights(model_b)["0.weight"], _weights(model_c)["0.weight"])


def test_engine_binds_scheduler_and_checkpointer_to_state_optimizer():
    engine = Path(__file__).resolve().parents[2] / "core/gdrn_modeling/engine/engine.py"
    source = engine.read_text(encoding="utf-8")
    unwrap = source.index("state_optimizer = my_checkpoint.unwrap_optimizer_for_checkpoint(optimizer)")
    assert unwrap < source.index("solver_utils.build_lr_scheduler(cfg, state_optimizer")
    assert unwrap < source.index("optimizer=state_optimizer,")
    assert source.index("checkpointer.resume_or_load(") < source.index("my_checkpoint.resync_wrapped_optimizer(optimizer)")
