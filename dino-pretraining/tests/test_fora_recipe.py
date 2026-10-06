"""Tests for the recipe fixes ported from the FORA CT DINOv3 pipeline (mr_dino/recipe.py)."""
import math

import pytest
import torch

from mr_dino.data import CropSpec, MRAlignedDINO3DDataset, collate_dino3d
from mr_dino.recipe import (
    CosineLinear,
    GroupedAdamW,
    binned_sinkhorn,
    build_param_groups,
    cancel_last_layer_gradients,
    normalize_prototype_layer,
    overlap_pairs,
    position_bin_ids,
    snap_second_start,
    token_offset,
)
from test_mr_dino import make_dummy_cache

PATCH = (2, 8, 8)
GLOBAL = (8, 32, 32)


def big_dataset(cache, cross_sequence_probability=0.0, seed=5):
    return MRAlignedDINO3DDataset(
        preprocessed_dir=str(cache),
        crop_spec=CropSpec(global_shape=GLOBAL, local_shape=(4, 16, 16), local_crops=3),
        cross_sequence_probability=cross_sequence_probability,
        candidate_trials=3,
        seed=seed,
        global_overlap=0.25,
        patch_size=PATCH,
    )


def test_cosine_prototypes_are_bounded_and_keep_parameter_name():
    head = torch.nn.Module()
    head.last_layer = torch.nn.Linear(16, 10, bias=False)
    torch.nn.init.normal_(head.last_layer.weight, std=5.0)
    normalize_prototype_layer(head)
    assert isinstance(head.last_layer, CosineLinear)
    assert "last_layer.weight" in dict(head.named_parameters())
    x = torch.nn.functional.normalize(torch.randn(7, 16), dim=-1)
    logits = head.last_layer(x)
    assert logits.abs().max() <= 1.0 + 1e-5
    expected = x @ torch.nn.functional.normalize(head.last_layer.weight, dim=1).t()
    torch.testing.assert_close(logits, expected)


def test_global_crops_overlap_quarter_and_sit_on_patch_lattice(tmp_path):
    dataset = big_dataset(make_dummy_cache(tmp_path / "cache", shape=(16, 64, 64)))
    moved = 0
    for i in range(len(dataset)):
        sample = dataset[(i, 1)]
        g0, g1 = sample["global_starts"]
        token_offset(g0, g1, PATCH)                          # raises unless patch-aligned
        for a, b, n in zip(g0, g1, GLOBAL):
            assert min(a + n, b + n) - max(a, b) >= math.floor(0.25 * n) - max(PATCH)
        moved += g0 != g1
        for start in sample["local_starts"]:
            assert any(all(g <= s and s + l <= g + G for s, l, g, G in zip(start, (4, 16, 16), p, GLOBAL))
                       for p in (g0, g1))
    assert moved > 0                                         # crops are no longer near-duplicates


def test_twin_tokens_cover_identical_voxels(tmp_path):
    dataset = big_dataset(make_dummy_cache(tmp_path / "cache", shape=(16, 64, 64)))
    grid = tuple(n // p for n, p in zip(GLOBAL, PATCH))
    checked = 0
    for i in range(len(dataset)):
        sample = dataset[(i, 3)]
        g0, g1 = sample["global_starts"]
        a, b = overlap_pairs(token_offset(g0, g1, PATCH), grid)
        crop0, crop1 = sample["teacher_global"][0, 0], sample["teacher_global"][1, 0]
        def patch(crop, flat):
            z, rem = divmod(int(flat), grid[1] * grid[2]); y, x = divmod(rem, grid[2])
            return crop[z*PATCH[0]:(z+1)*PATCH[0], y*PATCH[1]:(y+1)*PATCH[1], x*PATCH[2]:(x+1)*PATCH[2]]
        for ta, tb in list(zip(a, b))[:20]:
            torch.testing.assert_close(patch(crop0, ta), patch(crop1, tb))   # same sequence (p=0)
            checked += 1
    assert checked > 0


def test_collate_pairs_only_same_sequence_by_default(tmp_path):
    dataset = big_dataset(make_dummy_cache(tmp_path / "cache", shape=(16, 64, 64)), cross_sequence_probability=1.0)
    samples = [dataset[(i, 0)] for i in range(4)]
    assert all(s["global_sequences"][0] != s["global_sequences"][1] for s in samples)
    assert collate_dino3d(samples, patch_size=PATCH)["cross_a"].numel() == 0
    assert collate_dino3d(samples, patch_size=PATCH, cross_view_sequences="any")["cross_a"].numel() > 0


def test_binned_sinkhorn_balances_each_bin_separately():
    torch.manual_seed(0)
    grid, bins = (4, 4, 4), (2, 2, 2)
    idx = torch.arange(64)
    ids = position_bin_ids(idx, 64, grid, bins)
    assert torch.bincount(ids).tolist() == [8] * 8
    logits = torch.randn(64, 16) * 3
    from mr_dino.objective import distributed_sinkhorn
    q = binned_sinkhorn(distributed_sinkhorn, logits, 0.1, ids, 8)
    torch.testing.assert_close(q.sum(-1), torch.ones(64))
    for b in range(8):                                        # identical to balancing each bin on its own
        rows = (ids == b).nonzero().flatten()
        torch.testing.assert_close(q[rows], distributed_sinkhorn(logits[rows], 0.1))
    assert not torch.allclose(q, distributed_sinkhorn(logits, 0.1))   # and differs from one global balance


def test_param_groups_layerwise_decay_and_grouped_adamw():
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.patch_embed = torch.nn.Linear(4, 4)
            self.backbone.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
            self.backbone.norm = torch.nn.LayerNorm(4)
            self.backbone.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
            self.dino_head = torch.nn.Module()
            self.dino_head.last_layer = torch.nn.Linear(4, 8, bias=False)
    student = Student()
    groups = build_param_groups(student, layerwise_decay=0.5, patch_embed_lr_mult=0.2, n_blocks=3)
    lr_of = {name: g["lr_mult"] for g in groups for name in g["names"]}
    wd_of = {name: g["wd_mult"] for g in groups for name in g["names"]}
    assert lr_of["backbone.patch_embed.weight"] == pytest.approx(0.5 ** 4 * 0.2)
    assert lr_of["backbone.blocks.0.weight"] == pytest.approx(0.5 ** 3)
    assert lr_of["backbone.blocks.2.weight"] == pytest.approx(0.5)
    assert lr_of["backbone.norm.weight"] == 1.0 and lr_of["dino_head.last_layer.weight"] == 1.0
    assert wd_of["backbone.blocks.0.bias"] == 0.0 and wd_of["backbone.cls_token"] == 0.0 and wd_of["backbone.blocks.0.weight"] == 1.0
    for g in groups:
        g.pop("names")
    opt = GroupedAdamW(groups, lr=1.0, weight_decay=0.1)
    for g in opt.param_groups:
        g["lr"] = 1e-2
    student.backbone.patch_embed.weight.grad = torch.ones(4, 4)
    before = student.backbone.patch_embed.weight.detach().clone()
    opt.step()
    assert all(g["lr"] == 1e-2 for g in opt.param_groups)           # schedule untouched after step
    step = (before - student.backbone.patch_embed.weight.detach()).abs().max()
    assert step == pytest.approx(1e-2 * 0.5 ** 4 * 0.2, rel=0.05)   # Adam step ~ lr * lr_mult


def test_cancel_last_layer_gradients():
    head = torch.nn.Module()
    head.last_layer = torch.nn.Linear(3, 3, bias=False)
    head.mlp = torch.nn.Linear(3, 3)
    (head.last_layer(head.mlp(torch.randn(2, 3)))).sum().backward()
    assert cancel_last_layer_gradients(head) == 1
    assert head.last_layer.weight.grad is None and head.mlp.weight.grad is not None


def test_cross_view_term_trains_the_backbone(tmp_path):
    pytest.importorskip("dinov3")
    from mr_dino.model import DinoVisionTransformer3D
    from mr_dino.objective import DINO3DLearner, LossWeights

    dataset = big_dataset(make_dummy_cache(tmp_path / "cache", shape=(16, 64, 64)))
    batch = collate_dino3d([dataset[(0, 1)], dataset[(1, 1)]], patch_size=PATCH)
    assert batch["cross_a"].numel() > 0
    for key in ("teacher_global", "student_global", "student_local"):
        batch[key] = batch[key].float()
    batch["loss_weights"] = batch["sample_weights"]
    torch.manual_seed(0)
    backbone = DinoVisionTransformer3D(volume_size=GLOBAL, patch_size=PATCH, voxel_spacing_mm=(1.0, 0.5, 0.5),
                                       embed_dim=96, depth=2, num_heads=3, ffn_ratio=2, n_storage_tokens=2, drop_path_rate=0)
    learner = DINO3DLearner(backbone, prototypes=32, head_hidden_dim=64, dino_bottleneck_dim=32, ibot_bottleneck_dim=32,
                            loss_weights=LossWeights(dino=0, ibot=0, koleo=0, gram=0), position_bins=(2, 2, 2), cross_view_weight=1.0)
    loss, metrics = learner(batch, teacher_temperature=0.07, step=0)
    assert metrics["cross_pairs"] > 0 and torch.isfinite(metrics["cross"]) and metrics["cross"] > 0
    for key in ("dino_target_entropy", "ibot_target_entropy", "dino_kl", "ibot_kl", "ibot_prototype_usage"):
        assert torch.isfinite(metrics[key])
    loss.backward()   # only the cross term is weighted: it must reach the backbone blocks
    grads = [p.grad for p in learner.student.backbone.blocks.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
