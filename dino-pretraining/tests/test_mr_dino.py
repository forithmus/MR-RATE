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
    _discover_raw_atlas,
    _contained_start,
    _intersection_box,
    collate_dino3d,
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

    samples = _discover_raw_atlas(str(tmp_path), selected=None)
    assert [sample["study_uid"] for sample in samples] == ["study_atlas"]
    assert samples[0]["n_sequences"] == 2


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
    lo, hi = _intersection_box(first["global_starts"], (8, 32, 32))
    for start in first["local_starts"]:
        for s, n, left, right in zip(start, (4, 16, 16), lo, hi):
            assert left <= s and s + n <= right


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
