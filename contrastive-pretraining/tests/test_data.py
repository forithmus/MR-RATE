"""Tests for data pipeline: normalizers, dataset loading, collate_fn."""
import os
import json
import tempfile
import pytest
import numpy as np
import torch
import nibabel as nib
from data import (
    ZScoreNormalizer, PercentileNormalizer, MinMaxNormalizer,
    NORMALIZERS, MRReportDataset, collate_fn,
)


# ---------------------------------------------------------------------------
# Normalizer tests
# ---------------------------------------------------------------------------
class TestZScoreNormalizer:
    @pytest.fixture
    def norm(self):
        return ZScoreNormalizer()

    def test_output_range(self, norm):
        data = np.random.randn(32, 32, 32).astype(np.float32) * 100
        out = norm.normalize(data)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_zero_input(self, norm):
        data = np.zeros((8, 8, 8), dtype=np.float32)
        out = norm.normalize(data)
        assert np.all(out == 0)

    def test_preserves_shape(self, norm):
        data = np.random.randn(16, 32, 24).astype(np.float32)
        out = norm.normalize(data)
        assert out.shape == data.shape

    def test_no_nan(self, norm):
        data = np.random.randn(16, 16, 16).astype(np.float32) * 50
        out = norm.normalize(data)
        assert not np.isnan(out).any()

    def test_constant_nonzero(self, norm):
        """Constant nonzero input -> std~0 -> should clip and not NaN."""
        data = np.full((8, 8, 8), 5.0, dtype=np.float32)
        out = norm.normalize(data)
        assert not np.isnan(out).any()


class TestPercentileNormalizer:
    @pytest.fixture
    def norm(self):
        return PercentileNormalizer()

    def test_output_range(self, norm):
        data = np.random.randn(32, 32, 32).astype(np.float32) * 100
        out = norm.normalize(data)
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_zero_input(self, norm):
        data = np.zeros((8, 8, 8), dtype=np.float32)
        out = norm.normalize(data)
        assert np.all(out == 0)

    def test_custom_percentiles(self):
        norm = PercentileNormalizer(lower_percentile=1.0, upper_percentile=99.0)
        data = np.random.randn(32, 32, 32).astype(np.float32)
        out = norm.normalize(data)
        assert not np.isnan(out).any()

    def test_custom_limits(self):
        norm = PercentileNormalizer(lower_limit=0.0, upper_limit=1.0)
        data = np.random.randn(32, 32, 32).astype(np.float32)
        out = norm.normalize(data)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6


class TestMinMaxNormalizer:
    @pytest.fixture
    def norm(self):
        return MinMaxNormalizer()

    def test_output_range(self, norm):
        data = np.random.randn(32, 32, 32).astype(np.float32) * 100
        out = norm.normalize(data)
        assert abs(out.min() - (-1.0)) < 1e-5
        assert abs(out.max() - 1.0) < 1e-5

    def test_zero_input(self, norm):
        data = np.zeros((8, 8, 8), dtype=np.float32)
        out = norm.normalize(data)
        assert np.all(out == 0)

    def test_constant_input(self, norm):
        data = np.full((8, 8, 8), 42.0, dtype=np.float32)
        out = norm.normalize(data)
        assert np.all(out == 0)  # dmax - dmin < eps -> zeros

    def test_custom_limits(self):
        norm = MinMaxNormalizer(lower_limit=0.0, upper_limit=1.0)
        data = np.random.randn(16, 16, 16).astype(np.float32)
        out = norm.normalize(data)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6


class TestNormalizerRegistry:
    def test_all_normalizers_registered(self):
        assert 'zscore' in NORMALIZERS
        assert 'percentile' in NORMALIZERS
        assert 'minmax' in NORMALIZERS

    def test_instantiate_all(self):
        for name, cls in NORMALIZERS.items():
            obj = cls()
            data = np.random.randn(8, 8, 8).astype(np.float32)
            out = obj.normalize(data)
            assert out.shape == data.shape
            assert not np.isnan(out).any()


# ---------------------------------------------------------------------------
# Collate function tests
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Fixtures for synthetic dataset
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a minimal fake MR-RATE directory structure + JSONL + splits CSV."""
    mri_dir = tmp_path / "mri"

    # Create 4 subjects across 2 batches, each with 2-3 NIfTI volumes
    subjects = {
        "batch00": {
            "SUBJ_AAA": 2,
            "SUBJ_BBB": 3,
        },
        "batch01": {
            "SUBJ_CCC": 2,
            "SUBJ_DDD": 3,
        },
    }
    for batch, subs in subjects.items():
        for uid, n_vols in subs.items():
            img_dir = mri_dir / batch / uid / "img"
            img_dir.mkdir(parents=True)
            for i in range(n_vols):
                # Small random NIfTI (8x8x8)
                data = np.random.randn(8, 8, 8).astype(np.float32)
                img = nib.Nifti1Image(data, affine=np.eye(4))
                img.to_filename(str(img_dir / f"{uid}_series{i}.nii.gz"))

    # JSONL with volume_name field
    jsonl_path = tmp_path / "findings.jsonl"
    with open(jsonl_path, "w") as f:
        for batch, subs in subjects.items():
            for uid in subs:
                entry = {
                    "volume_name": uid,
                    "original_findings": "Test findings",
                    "valid_json": True,
                    "extracted_sentences": [
                        "There is a lesion",
                        "There is no hemorrhage",
                        "There is mild atrophy",
                    ],
                    "raw_output": "",
                }
                f.write(json.dumps(entry) + "\n")

    # Splits CSV
    splits_path = tmp_path / "splits.csv"
    with open(splits_path, "w") as f:
        f.write("batch_id,patient_uid,study_uid,split\n")
        f.write("batch00,1,SUBJ_AAA,train\n")
        f.write("batch00,2,SUBJ_BBB,train\n")
        f.write("batch01,3,SUBJ_CCC,val\n")
        f.write("batch01,4,SUBJ_DDD,test\n")

    return {
        "mri_dir": str(mri_dir),
        "jsonl_path": str(jsonl_path),
        "splits_path": str(splits_path),
        "subjects": subjects,
    }


# ---------------------------------------------------------------------------
# Dataset integration tests
# ---------------------------------------------------------------------------
class TestMRReportDataset:
    def test_loads_all_subjects_without_splits(self, synthetic_dataset):
        """Without splits_csv, all 4 subjects should load."""
        ds = MRReportDataset(
            data_folder=synthetic_dataset["mri_dir"],
            jsonl_file=synthetic_dataset["jsonl_path"],
            max_sentences_per_image=5,
        )
        assert len(ds) == 4

    def test_splits_train(self, synthetic_dataset):
        """With splits_csv + split=train, only SUBJ_AAA and SUBJ_BBB."""
        ds = MRReportDataset(
            data_folder=synthetic_dataset["mri_dir"],
            jsonl_file=synthetic_dataset["jsonl_path"],
            splits_csv=synthetic_dataset["splits_path"],
            split="train",
            max_sentences_per_image=5,
        )
        assert len(ds) == 2
        ids = {s["subject_id"] for s in ds.samples}
        assert ids == {"SUBJ_AAA", "SUBJ_BBB"}

    def test_splits_val(self, synthetic_dataset):
        ds = MRReportDataset(
            data_folder=synthetic_dataset["mri_dir"],
            jsonl_file=synthetic_dataset["jsonl_path"],
            splits_csv=synthetic_dataset["splits_path"],
            split="val",
            max_sentences_per_image=5,
        )
        assert len(ds) == 1
        assert ds.samples[0]["subject_id"] == "SUBJ_CCC"

    def test_splits_test(self, synthetic_dataset):
        ds = MRReportDataset(
            data_folder=synthetic_dataset["mri_dir"],
            jsonl_file=synthetic_dataset["jsonl_path"],
            splits_csv=synthetic_dataset["splits_path"],
            split="test",
            max_sentences_per_image=5,
        )
        assert len(ds) == 1
        assert ds.samples[0]["subject_id"] == "SUBJ_DDD"

    def test_getitem_shapes(self, synthetic_dataset):
        """Check output tensor shapes from __getitem__."""
        ds = MRReportDataset(
            data_folder=synthetic_dataset["mri_dir"],
            jsonl_file=synthetic_dataset["jsonl_path"],
            max_sentences_per_image=5,
            target_shape=(8, 8, 8),
        )
        volume_stack, sentences, mask = ds[0]
        n_vols = len(ds.samples[0]["image_paths"])
        assert volume_stack.shape[0] == n_vols       # N volumes
        assert volume_stack.shape[1] == 1             # 1 channel
        assert len(sentences) == 5                     # max_sentences
        assert mask.shape == (5,)
        assert mask.dtype == torch.bool

    def test_sentence_padding(self, synthetic_dataset):
        """3 real sentences padded to max_sentences=5."""
        ds = MRReportDataset(
            data_folder=synthetic_dataset["mri_dir"],
            jsonl_file=synthetic_dataset["jsonl_path"],
            max_sentences_per_image=5,
            target_shape=(8, 8, 8),
        )
        _, sentences, mask = ds[0]
        assert mask[:3].all()       # 3 real sentences
        assert not mask[3:].any()   # 2 padding
        assert sentences[3] == ""
        assert sentences[4] == ""

    def test_volume_name_field(self, synthetic_dataset):
        """Ensure volume_name (not subject_id) is read from JSONL."""
        ds = MRReportDataset(
            data_folder=synthetic_dataset["mri_dir"],
            jsonl_file=synthetic_dataset["jsonl_path"],
            max_sentences_per_image=5,
        )
        # All 4 subjects found means volume_name was correctly read
        found_ids = {s["subject_id"] for s in ds.samples}
        assert found_ids == {"SUBJ_AAA", "SUBJ_BBB", "SUBJ_CCC", "SUBJ_DDD"}

    def test_batch_directory_traversal(self, synthetic_dataset):
        """Ensure batchXX/ directories are properly traversed."""
        ds = MRReportDataset(
            data_folder=synthetic_dataset["mri_dir"],
            jsonl_file=synthetic_dataset["jsonl_path"],
            max_sentences_per_image=5,
        )
        # SUBJ_BBB has 3 volumes, SUBJ_AAA has 2
        for sample in ds.samples:
            if sample["subject_id"] == "SUBJ_BBB":
                assert len(sample["image_paths"]) == 3
            elif sample["subject_id"] == "SUBJ_AAA":
                assert len(sample["image_paths"]) == 2

    def test_collate_with_dataset(self, synthetic_dataset):
        """End-to-end: dataset -> collate_fn -> correct batch shapes."""
        ds = MRReportDataset(
            data_folder=synthetic_dataset["mri_dir"],
            jsonl_file=synthetic_dataset["jsonl_path"],
            max_sentences_per_image=5,
            target_shape=(8, 8, 8),
        )
        item = ds[0]
        images, sentences, masks = collate_fn([item])
        assert images.dim() == 6    # [1, N, 1, D, H, W]
        assert images.shape[0] == 1
        assert masks.shape[0] == 1


class TestCollateFn:
    def test_collate_adds_batch_dim(self):
        """collate_fn should add batch dim to images and masks."""
        images = torch.randn(3, 1, 8, 16, 16)  # [N, 1, D, H, W]
        sentences = ["sentence 1", "sentence 2"]
        masks = torch.tensor([True, True])

        batch = [(images, sentences, masks)]
        out_images, out_sentences, out_masks = collate_fn(batch)

        assert out_images.shape == (1, 3, 1, 8, 16, 16)  # [1, N, 1, D, H, W]
        assert out_sentences == sentences
        assert out_masks.shape == (1, 2)
