import json
import random

import numpy as np
import pytest
import torch

from mr_dino.data import (
    CACHE_MANIFEST_NAME,
    CropSpec,
    InfiniteStudySampler,
    MRAtlasDINO3DDataset,
    _raw_volume,
    _contained_start,
    _intersection_box,
    collate_dino3d,
    discover_raw_atlas,
    previous_mr_data_module,
    validate_atlas_cache,
)


def make_dummy_cache(root, studies=4, shape=(8, 32, 32)):
    space = root / "atlas_space"
    space.mkdir(parents=True)
    manifest = {
        "version": 1,
        "layout": "per_subject_stack",
        "space": "atlas_space",
        "target_spacing": [1.0, 0.5, 0.5],
        "target_shape": list(shape),
        "posterior_shift_mm": 15.0,
        "normalizer": "zscore",
        "normalizer_kwargs": {},
        "dtype": "float16",
    }
    (space / CACHE_MANIFEST_NAME).write_text(json.dumps(manifest))
    for study in range(studies):
        z, y, x = np.indices(shape)
        sequences = []
        for sequence in range(3):
            center = np.array(shape) / 2 + np.array([0, sequence - 1, study % 2])
            radius = ((z - center[0]) / 3) ** 2 + ((y - center[1]) / 9) ** 2 + ((x - center[2]) / 9) ** 2
            volume = np.exp(-radius) * (0.5 + 0.2 * sequence)
            volume += np.sin((x + sequence) / 4) * 0.05
            sequences.append(volume.astype(np.float16))
        np.savez(space / f"study_{study:03d}.npz", volumes=np.stack(sequences))
    return root


def tiny_dataset(cache, seed=17):
    return MRAtlasDINO3DDataset(
        preprocessed_dir=str(cache),
        crop_spec=CropSpec(
            global_shape=(8, 32, 32),
            local_shape=(4, 16, 16),
            local_crops=2,
        ),
        cross_sequence_probability=1.0,
        candidate_trials=3,
        seed=seed,
    )


def test_cache_contract_rejects_non_atlas(tmp_path):
    cache = make_dummy_cache(tmp_path / "cache")
    assert validate_atlas_cache(str(cache))["space"] == "atlas_space"
    manifest_path = cache / "atlas_space" / CACHE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["space"] = "native_space"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="expected 'atlas_space'"):
        validate_atlas_cache(str(cache))


def test_cache_contract_rejects_coreg_space_argument(tmp_path):
    cache = make_dummy_cache(tmp_path / "cache")
    with pytest.raises(ValueError, match="requires space='atlas_space'"):
        validate_atlas_cache(str(cache), space="coreg_space")


def test_raw_discovery_selects_atlas_img_not_coreg_img(tmp_path):
    atlas_dir = tmp_path / "batch00" / "study_atlas" / "atlas_img"
    atlas_dir.mkdir(parents=True)
    (atlas_dir / "t1.nii.gz").touch()
    (atlas_dir / "flair.nii.gz").touch()
    coreg_dir = tmp_path / "batch00" / "study_coreg" / "coreg_img"
    coreg_dir.mkdir(parents=True)
    (coreg_dir / "t1.nii.gz").touch()

    samples = discover_raw_atlas(str(tmp_path), selected=None)
    assert [sample["study_uid"] for sample in samples] == ["study_atlas"]
    assert samples[0]["n_sequences"] == 2


def test_raw_transform_is_exact_previous_mil_transform(tmp_path):
    nib = pytest.importorskip("nibabel")
    image_dir = tmp_path / "batch00" / "study" / "atlas_img"
    image_dir.mkdir(parents=True)
    x, y, z = np.indices((12, 14, 6), dtype=np.float32)
    array = np.sin(x / 3) + np.cos(y / 4) + z / 7
    array[(x + y + z) % 11 == 0] = 0
    path = image_dir / "t1.nii.gz"
    nib.save(nib.Nifti1Image(array, np.diag([0.5, 0.5, 1.0, 1.0])), path)

    shape = (6, 12, 14)
    spacing = (1.0, 0.5, 0.5)
    previous = previous_mr_data_module()
    expected = previous.preprocess_nii(
        str(path), spacing, shape, 0, previous.NORMALIZERS["zscore"]()
    )
    actual = _raw_volume(str(path), shape, spacing, posterior_shift_mm=0)
    torch.testing.assert_close(actual, torch.from_numpy(expected).to(torch.bfloat16), rtol=0, atol=0)


def test_aligned_cross_sequence_views_and_determinism(tmp_path):
    dataset = tiny_dataset(make_dummy_cache(tmp_path / "cache"))
    assert len(dataset) == 12  # every sequence from all four studies is indexed
    assert [dataset.index[i][1] for i in range(3)] == [0, 1, 2]
    first = dataset[(0, 2)]
    again = dataset[(0, 2)]
    assert first["global_sequences"][0] != first["global_sequences"][1]
    assert first["global_starts"] == again["global_starts"]
    assert torch.equal(first["teacher_global"], again["teacher_global"])
    assert first["teacher_global"].shape == (2, 1, 8, 32, 32)
    assert first["student_local"].shape == (2, 1, 4, 16, 16)
    for start in first["local_starts"]:   # each local crop lies inside one of the two globals
        assert any(
            all(g <= s and s + n <= g + G for s, n, g, G in zip(start, (4, 16, 16), parent, (8, 32, 32)))
            for parent in first["global_starts"]
        )


def test_collate_masks_match_patch_grid(tmp_path):
    dataset = tiny_dataset(make_dummy_cache(tmp_path / "cache"))
    batch = collate_dino3d([dataset[0], dataset[1]], patch_size=(2, 8, 8))
    assert batch["teacher_global"].shape == (2, 2, 1, 8, 32, 32)
    assert batch["student_local"].shape == (2, 2, 1, 4, 16, 16)
    assert batch["masks"].shape == (4, 64)
    assert batch["mask_indices"].numel() > 0
    torch.testing.assert_close(batch["sample_weights"], torch.full((2,), 1 / 3))


def test_grouped_sampler_has_exact_coverage_and_resume():
    sampler = InfiniteStudySampler(6, seed=91, group_sizes=[3, 1, 2])
    iterator = iter(sampler)
    first_epoch = [next(iterator)[0] for _ in range(6)]
    assert sorted(first_epoch) == list(range(6))
    # Each study's contiguous index range remains contiguous in the stream.
    positions = {value: first_epoch.index(value) for value in first_epoch}
    assert max(positions[i] for i in (0, 1, 2)) - min(positions[i] for i in (0, 1, 2)) == 2
    resumed = InfiniteStudySampler(6, seed=91, offset=4, group_sizes=[3, 1, 2])
    resumed_iterator = iter(resumed)
    assert [next(resumed_iterator)[0] for _ in range(2)] == first_epoch[4:]


def test_synthetic_forward_backward_and_checkpoint(tmp_path):
    pytest.importorskip("dinov3")
    from mr_dino.model import DinoVisionTransformer3D
    from mr_dino.objective import DINO3DLearner, LossWeights

    dataset = tiny_dataset(make_dummy_cache(tmp_path / "cache"))
    batch = collate_dino3d([dataset[0], dataset[1]], patch_size=(2, 8, 8))
    for key in ("teacher_global", "student_global", "student_local"):
        batch[key] = batch[key].float()
    batch["loss_weights"] = batch["sample_weights"]

    backbone = DinoVisionTransformer3D(
        volume_size=(8, 32, 32),
        patch_size=(2, 8, 8),
        voxel_spacing_mm=(1.0, 0.5, 0.5),
        embed_dim=96,
        depth=2,
        num_heads=3,
        ffn_ratio=2,
        n_storage_tokens=2,
        drop_path_rate=0,
    )
    learner = DINO3DLearner(
        backbone,
        prototypes=32,
        head_hidden_dim=64,
        dino_bottleneck_dim=32,
        ibot_bottleneck_dim=32,
        loss_weights=LossWeights(gram=0),
    )
    optimizer = torch.optim.AdamW(learner.student.parameters(), lr=1e-4)
    loss, metrics = learner(batch, teacher_temperature=0.07, step=0)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in learner.student.parameters())
    optimizer.step()
    learner.update_teacher(0.99)

    checkpoint = tmp_path / "dummy_checkpoint.pt"
    torch.save({"model": learner.state_dict(), "optimizer": optimizer.state_dict(), "step": 1}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    learner.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    assert saved["step"] == 1



# --------------------------------------------------------------------------- Sinkhorn count
def _reference_sinkhorn_single_process(logits, temperature, iterations=3):
    """Plain (non-distributed) Sinkhorn over ALL tokens: the definition the distributed version must match."""
    q = torch.exp((logits.float() - logits.float().max()) / temperature).t()
    global_batch, prototypes = q.shape[1], q.shape[0]
    q /= q.sum().clamp_min(1e-12)
    for _ in range(iterations):
        q /= q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        q /= prototypes
        q /= q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q /= global_batch
    return (q * global_batch).t()


def _legacy_sinkhorn(logits, temperature, world, iterations=3):
    """The historical implementation (local rows x world size denominator) for equal-count parity."""
    import torch.distributed as dist
    max_logit = logits.detach().float().max()
    if dist.is_initialized():
        dist.all_reduce(max_logit, op=dist.ReduceOp.MAX)
    q = torch.exp((logits.float() - max_logit) / temperature).t()
    global_batch = q.shape[1] * world
    prototypes = q.shape[0]
    total = q.sum()
    if dist.is_initialized():
        dist.all_reduce(total)
    q /= total.clamp_min(1e-12)
    for _ in range(iterations):
        rows = q.sum(dim=1, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(rows)
        q /= rows.clamp_min(1e-12)
        q /= prototypes
        q /= q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q /= global_batch
    return (q * global_batch).t()


def test_sinkhorn_single_process_matches_legacy_and_rejects_empty():
    from mr_dino.objective import distributed_sinkhorn
    torch.manual_seed(0)
    logits = torch.randn(23, 7)
    torch.testing.assert_close(distributed_sinkhorn(logits, 0.07), _legacy_sinkhorn(logits, 0.07, 1), atol=0, rtol=0)
    with pytest.raises(ValueError):
        distributed_sinkhorn(torch.zeros(0, 7), 0.07)


def _sinkhorn_worker(rank, rendezvous):
    import torch.distributed as dist
    from datetime import timedelta
    from mr_dino.objective import distributed_sinkhorn
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(11)
        logits = torch.randn(30, 9) * 3
        # Equal local counts: corrected == legacy bit for bit.
        local = logits[rank * 15:(rank + 1) * 15]
        torch.testing.assert_close(distributed_sinkhorn(local, 0.07), _legacy_sinkhorn(local, 0.07, 2), atol=0, rtol=0)
        # Unequal local counts (masked tokens): corrected == single-process over all tokens; legacy differs.
        for split in (7, 0, 30):
            local = logits[:split] if rank == 0 else logits[split:]
            got = distributed_sinkhorn(local, 0.07)
            expected = _reference_sinkhorn_single_process(logits, 0.07)
            expected_local = expected[:split] if rank == 0 else expected[split:]
            assert got.shape == local.shape and got.dtype == torch.float32
            torch.testing.assert_close(got, expected_local, atol=1e-6, rtol=1e-5)
            if split == 7:
                legacy = _legacy_sinkhorn(local, 0.07, 2)
                assert not torch.allclose(legacy, expected_local, atol=1e-6, rtol=1e-5)
        # Globally empty raises on both ranks.
        with pytest.raises(ValueError):
            distributed_sinkhorn(torch.zeros(0, 9), 0.07)
    finally:
        dist.destroy_process_group()


def test_sinkhorn_uses_actual_global_token_count_across_ranks(tmp_path):
    torch.multiprocessing.spawn(_sinkhorn_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
