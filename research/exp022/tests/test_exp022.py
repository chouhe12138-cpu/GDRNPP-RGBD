"""CPU contracts for the progressive PCC implementation."""

import math

import numpy as np
import pytest
import torch
from mmcv import Config

from core.gdrn_modeling.models.heads.progressive_pcc_head import (
    ProgressivePCCHead,
    load_pcc_hierarchy,
)
from core.gdrn_modeling.models.heads.pcc_blocks import (
    ImageAttentionBlock, HierarchicalCADMatcher, StageTransition,
)
from core.utils.solver_utils import build_lr_scheduler
from research.exp022.build_hierarchy import SOURCE_DEFAULT, _reused_object
from research.run_contract import validate_research_run_config


HIERARCHY = ".local/dataset_cache/exp022/reused_v1.npz"


@pytest.fixture(scope="module")
def head():
    torch.manual_seed(42)
    return ProgressivePCCHead(HIERARCHY, token_dim=32)


def test_reused_tree_preserves_source_leaves():
    with np.load(SOURCE_DEFAULT) as data:
        for index in range(8):
            levels, indices = _reused_object(data, index)
            np.testing.assert_array_equal(np.sort(indices), np.arange(4096))
            np.testing.assert_array_equal(
                levels[-1][0][np.argsort(indices)], data["fine_anchors"][index].reshape(4096, 3)
            )


def test_hierarchy_schema():
    arrays = load_pcc_hierarchy(HIERARCHY)
    assert tuple(arrays["level4_anchors"].shape) == (8, 4096, 3)
    assert tuple(arrays["symmetry_counts"].shape) == (8,)
    assert arrays["symmetry_counts"].tolist() == [1, 1, 1, 1, 1, 2, 2, 1]


def test_hierarchy_rejects_more_than_two_symmetries(tmp_path):
    with np.load(HIERARCHY, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    arrays["symmetry_counts"][0] = 3
    path = tmp_path / "too_many_symmetries.npz"
    np.savez(path, **arrays)
    with pytest.raises(RuntimeError, match="supports at most 2 equivalents, got 3"):
        load_pcc_hierarchy(path)


def test_sparse_resize_probability_and_path_identity():
    ids = torch.tensor([[[[3, 7], [3, 7]], [[3, 7], [3, 7]]]])
    scores = torch.tensor([[[[-0.2, -1.7], [-0.2, -1.7]], [[-0.2, -1.7], [-0.2, -1.7]]]])
    ids_new, log_new = ProgressivePCCHead._resize_sparse_paths(ids, scores)
    assert tuple(ids_new.shape) == (1, 4, 4, 2)
    assert torch.all(ids_new[..., 0] == 3)
    assert torch.allclose(log_new.exp().sum(-1), torch.ones(1, 4, 4), atol=1e-6)


@pytest.mark.parametrize("resolution,window,shift", [(8, None, 0), (16, None, 0),
                                                      (32, 8, 0), (32, 8, 4),
                                                      (64, 8, 0), (64, 8, 4)])
def test_image_attention_shape_finite_and_gradient(resolution, window, shift):
    torch.manual_seed(31)
    block = ImageAttentionBlock(32, 8, resolution, window, shift)
    image = torch.randn(1, resolution * resolution, 32, requires_grad=True)
    output = block(image)
    assert output.shape == image.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert torch.isfinite(block.attn.in_proj_weight.grad).all()
    if window is not None:
        grid = image.detach().reshape(1, resolution, resolution, 32)
        restored = block._reverse(block._partition(grid, window), 1, resolution, window)
        torch.testing.assert_close(restored, grid)


@pytest.mark.parametrize("resolution", [32, 64])
def test_shift_mask_blocks_cyclic_wraparound(resolution):
    block = ImageAttentionBlock(8, 2, resolution, 8, 4)
    block.norm = torch.nn.Identity()
    with torch.no_grad():
        block.attn.in_proj_weight.zero_()
        block.attn.in_proj_weight[16:] = torch.eye(8)
        block.attn.in_proj_bias.zero_()
        block.attn.out_proj.weight.copy_(torch.eye(8))
        block.attn.out_proj.bias.zero_()
    image = torch.zeros(1, resolution, resolution, 8)
    image[0, 0, 0, 0] = 1
    input_tokens = image.reshape(1, resolution * resolution, 8)
    blocked = block(input_tokens).reshape(1, resolution, resolution, 8)
    assert blocked[0, -1, -1, 0].abs() < 1e-7
    mask = block.shift_mask
    block.shift_mask = None
    leaked = block(input_tokens).reshape(1, resolution, resolution, 8)
    block.shift_mask = mask
    assert leaked[0, -1, -1, 0] > 0


def test_shifted_sdpa_matches_mha_output_and_gradients():
    torch.manual_seed(73)
    block = ImageAttentionBlock(32, 8, 32, 8, 4).double()
    image = torch.randn(2, 32 * 32, 32, dtype=torch.float64, requires_grad=True)
    normalized = block.norm(image)
    grid = normalized.reshape(2, 32, 32, 32).roll(shifts=(-4, -4), dims=(1, 2))
    windows = block._partition(grid, 8)
    mask = block.shift_mask.repeat(2, 1, 1).repeat_interleave(8, dim=0)
    reference, _ = block.attn(windows, windows, windows, attn_mask=mask, need_weights=False)
    actual = block._shifted_attention(windows, 2)
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
    targets = (image, block.attn.in_proj_weight, block.attn.in_proj_bias,
               block.attn.out_proj.weight, block.attn.out_proj.bias)
    reference_grads = torch.autograd.grad(reference.square().mean(), targets, retain_graph=True)
    actual_grads = torch.autograd.grad(actual.square().mean(), targets)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-8, atol=1e-10)


def test_local_cad_qkv_logits_context_and_gradient():
    matcher = HierarchicalCADMatcher(32)
    query = torch.randn(3, 1, 32, requires_grad=True)
    bank = torch.randn(3, 8, 32, requires_grad=True)
    keys, values = matcher.project_bank(bank)
    logits, context = matcher.match_root(query, torch.arange(3), (keys, values))
    assert logits.shape == (3, 1, 8)
    assert context.shape == (3, 1, 32)
    torch.testing.assert_close(logits.float().softmax(-1).sum(-1), torch.ones(3, 1))
    (logits.square().mean() + context.square().mean()).backward()
    for parameter in (matcher.q_proj.weight, matcher.k_proj.weight,
                      matcher.v_proj.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert torch.isfinite(query.grad).all() and torch.isfinite(bank.grad).all()


def test_stage_transition_and_removed_matcher_projection():
    transition = StageTransition(512, 256)
    assert sum(parameter.numel() for parameter in transition.parameters()) == 131328
    assert transition(torch.randn(2, 512, 8, 8)).shape == (2, 256, 16, 16)
    matcher = HierarchicalCADMatcher(256)
    assert sum(parameter.numel() for parameter in matcher.parameters()) == 3 * 256 * 256
    # The old two consecutive linear maps can be folded exactly at inference.
    old_projection = torch.nn.Linear(256, 256, bias=False)
    context_projection = torch.nn.Linear(256, 512)
    context = torch.randn(4, 256)
    folded = torch.nn.functional.linear(
        context, context_projection.weight @ old_projection.weight,
        context_projection.bias,
    )
    torch.testing.assert_close(folded, context_projection(old_projection(context)))


def test_beam_top2_matches_exhaustive_retained_candidates():
    parent = torch.tensor([[1, 3]])
    prior = torch.tensor([[-0.8, -1.0]])
    conditional = torch.full((1, 2, 8), -5.0)
    conditional[0, 0, 2] = -0.1
    conditional[0, 1, 4] = -0.05
    scores, ids = ProgressivePCCHead._advance_beam(parent, prior, conditional)
    exhaustive = sorted((float(prior[0, p] + conditional[0, p, child]),
                         int(parent[0, p]) * 8 + child)
                        for p in range(2) for child in range(8))[-2:][::-1]
    torch.testing.assert_close(scores[0], torch.tensor([item[0] for item in exhaustive]))
    torch.testing.assert_close(ids[0], torch.tensor([item[1] for item in exhaustive]))
    assert ids[0, 0] // 8 != ids[0, 1] // 8


def test_head_train_infer_and_gradients(head):
    feature = torch.randn(2, 1024, 8, 8)
    classes = torch.tensor([0, 1])
    xyz = torch.full((2, 3, 64, 64), 0.5)
    mask = torch.ones(2, 1, 64, 64)
    losses, stats = head(feature, classes, xyz, mask, collect_diagnostics=True)
    assert set(losses) == {"loss_pcc_route", "loss_pcc_residual", "loss_pcc_mask"}
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    parameters = (head.stages[0].image_proj.weight,
                  head.stages[0].image_attention[0].attn.in_proj_weight,
                  head.stages[0].matcher.q_proj.weight,
                  head.stages[0].matcher.k_proj.weight,
                  head.stages[0].matcher.v_proj.weight,
                  head.cad_token_encoder[0].weight,
                  head.residual_head[0].weight)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in parameters)
    assert torch.isfinite(stats["selected_symmetry_branch_mean"])
    assert all(torch.isfinite(stats[name]).all() for name in
               ("fusion_update_ratio", "route_entropy", "top1_route_prob",
                "top2_route_prob_mass"))
    assert (stats["top2_route_prob_mass"] < 1).all()
    with torch.no_grad():
        output = head(feature[:1], classes[:1])
    assert tuple(output["xyz_norm"].shape) == (1, 3, 64, 64)
    assert tuple(output["mask_logit"].shape) == (1, 1, 64, 64)
    assert output["leaf_ids"].min() >= 0
    assert output["leaf_ids"].max() < 4096
    assert output["residual_norm"].max() <= 1.000001
    assert torch.allclose(output["beam_scores"].exp().sum(-1),
                          torch.ones(1, 64, 64), atol=1e-5)


def test_symmetric_subset_branch_is_finite(head):
    feature = torch.zeros(2, 1024, 8, 8)
    classes = torch.tensor([0, 5])
    xyz = torch.full((2, 3, 64, 64), 0.5)
    visible = torch.ones(2, 1, 64, 64)
    losses, stats = head(feature, classes, xyz, visible)
    assert all(torch.isfinite(value) for value in losses.values())
    assert 0 <= stats["selected_symmetry_branch_mean"] <= 0.5


def test_formal_training_skips_optional_diagnostics(head, monkeypatch):
    def unexpected_diagnostics(*_args):
        raise AssertionError("Formal training must not compute route diagnostics")

    monkeypatch.setattr(head, "_route_diagnostics", unexpected_diagnostics)
    feature = torch.randn(2, 1024, 8, 8)
    classes = torch.tensor([0, 5])
    xyz = torch.full((2, 3, 64, 64), 0.5)
    mask = torch.ones(2, 1, 64, 64)
    losses, stats = head(feature, classes, xyz, mask)
    assert all(torch.isfinite(value) for value in losses.values())
    assert set(stats) == {"selected_symmetry_branch_mean", "fusion_gates"}


def test_symmetric_branch_shares_stage1_and_handles_empty_visibility(head):
    calls = []
    handle = head.stages[0].image_attention[0].attn.register_forward_hook(
        lambda *_: calls.append(1)
    )
    try:
        feature = torch.randn(2, 1024, 8, 8)
        classes = torch.tensor([0, 5])
        xyz = torch.full((2, 3, 64, 64), 0.5)
        mask = torch.zeros(2, 1, 64, 64)
        losses, stats = head(feature, classes, xyz, mask)
        assert len(calls) == 1
        assert all(torch.isfinite(value) for value in losses.values())
        assert losses["loss_pcc_route"] == 0
        assert losses["loss_pcc_residual"] == 0
        assert torch.isfinite(stats["selected_symmetry_branch_mean"])
        sum(losses.values()).backward()
        assert torch.isfinite(head.stages[0].image_proj.weight.grad).all()
    finally:
        handle.remove()


def test_symmetry_selection_ignores_mask_but_trains_selected_mask(head):
    canonical = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 10.0]], requires_grad=True)
    alternate = torch.tensor([[0.9, 1.0, 100.0], [1.1, 1.0, -100.0]], requires_grad=True)
    selected, use_alternate = head._select_symmetry_losses(canonical, alternate)
    torch.testing.assert_close(use_alternate, torch.tensor([True, False]))
    torch.testing.assert_close(selected[:, 2], torch.tensor([100.0, 10.0]))
    selected.sum().backward()
    torch.testing.assert_close(canonical.grad[:, 2], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(alternate.grad[:, 2], torch.tensor([1.0, 0.0]))


@pytest.mark.parametrize("sparse_routes", [False, True])
def test_static_block_match_matches_original_packing_and_gradients(head, sparse_routes):
    torch.manual_seed(19)
    query = torch.randn(257, 32, requires_grad=True)
    raw_bank = torch.randn(2, 64, 32, requires_grad=True)
    matcher = head.stages[1].matcher
    bank = matcher.project_bank(raw_bank)
    object_inverse = torch.randint(0, 2, (257,))
    parent = torch.randint(0, 8, (257,))
    if sparse_routes:
        object_inverse = torch.cat((torch.zeros(200, dtype=torch.long),
                                    torch.ones(57, dtype=torch.long)))
        parent = torch.cat((torch.zeros(200, dtype=torch.long),
                            torch.full((57,), 7, dtype=torch.long)))
    target = torch.randn(257, 3)
    anchors = head.level2_anchors[:2]

    def simple_match(q, projected_bank):
        child_keys = projected_bank[0].reshape(2, 8, 8, 32)[object_inverse, parent]
        child_values = projected_bank[1].reshape(2, 8, 8, 32)[object_inverse, parent]
        projected_query = matcher.q_proj(q)
        logits = torch.bmm(projected_query[:, None], child_keys.transpose(1, 2))[:, 0] / math.sqrt(32)
        probability = torch.softmax(logits.float(), -1)
        context = torch.bmm(probability[:, None], child_values)[:, 0]
        child_anchors = anchors.reshape(2, 8, 8, 3)[object_inverse, parent]
        labels = (target[:, None] - child_anchors).square().sum(-1).argmin(-1)
        return logits, context, labels

    old_logits, old_context, old_labels = simple_match(query, bank)
    new_logits, new_context, new_labels = matcher.match_packed(
        query, object_inverse, parent, bank, target, anchors
    )
    torch.testing.assert_close(new_logits, old_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(new_context, old_context, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(new_labels, old_labels)
    grad_targets = (query, raw_bank, *matcher.parameters())
    old_grads = torch.autograd.grad(old_logits.square().mean() + old_context.square().mean(),
                                    grad_targets, retain_graph=True)
    new_grads = torch.autograd.grad(new_logits.square().mean() + new_context.square().mean(),
                                    grad_targets)
    for new, old in zip(new_grads, old_grads):
        torch.testing.assert_close(new, old, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name", ["train_reused.py", "smoke_reused.py", "smoke_independent.py"])
def test_config_contract(name):
    path = f"configs/gdrn/lmo_pbr/research/exp022_progressive_pcc/{name}"
    cfg = Config.fromfile(path)
    validate_research_run_config(cfg, mode="formal" if name == "train_reused.py" else "smoke",
                                 expected_experiment_id="EXP-20260916-022-progressive-pcc")
    assert cfg.SOLVER.AMP.ENABLED
    init = cfg.MODEL.POSE_NET.PCC_HEAD.INIT_CFG
    assert (init.num_heads, tuple(init.stage_attention), init.window_size,
            init.shift_size, init.attention_dropout) == (
                8, ("global", "global", "window", "window"), 8, 4, 0.0
            )


def test_warmup_transitions_directly_to_cosine():
    cfg = Config.fromfile("configs/gdrn/lmo_pbr/research/exp022_progressive_pcc/train_reused.py")
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=3e-4)
    _scheduler, factor = build_lr_scheduler(cfg, optimizer, total_iters=1000, return_function=True)
    assert factor(0) == pytest.approx(0.001)
    assert factor(40) == pytest.approx(1.0)
    assert factor(41) < 1.0
    assert factor(1000) == pytest.approx(0.01)


def test_formal_v1_parameters_are_explicit(head):
    cfg = Config.fromfile("configs/gdrn/lmo_pbr/research/exp022_progressive_pcc/train_reused.py")
    init = cfg.MODEL.POSE_NET.PCC_HEAD.INIT_CFG
    assert (init.route_weight, init.residual_weight, init.mask_weight) == (1.0, 1.0, 1.0)
    assert (init.residual_beta, init.beam_k, init.token_dim) == (0.1, 2, 256)
    assert (init.num_heads, tuple(init.stage_attention), init.window_size,
            init.shift_size, init.attention_dropout) == (
                8, ("global", "global", "window", "window"), 8, 4, 0.0
            )
    assert all(stage.gate_logit.item() == -4.0 for stage in head.stages)
    optimizer = cfg.SOLVER.OPTIMIZER_CFG
    assert (optimizer.type, optimizer.lr, optimizer.weight_decay, optimizer.betas) == (
        "AdamW", 3e-4, 0.01, (0.9, 0.999)
    )
    assert (cfg.SOLVER.WARMUP_RATIO, cfg.SOLVER.TARGET_LR_FACTOR, cfg.SEED) == (
        0.04, 0.01, 42
    )
