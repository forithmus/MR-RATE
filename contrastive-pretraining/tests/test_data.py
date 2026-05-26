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
    NORMALIZERS, SPACE_TO_IMG_SUBDIR, MRReportDataset, collate_fn,
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


# ---------------------------------------------------------------------------
# Layout 2 image-subdir mapping tests (raw HuggingFace downloads)
# ---------------------------------------------------------------------------
def _make_hf_layout(tmp_path, img_subdir):
    """Create a minimal HF-download-like tree with a custom image subdir name.

    Mirrors data_folder/batchXX/<study_uid>/<img_subdir>/*.nii.gz used by the
    raw HuggingFace dataset downloads (MR-RATE, MR-RATE-coreg, MR-RATE-atlas).
    """
    mri_dir = tmp_path / "mri"
    subjects = {
        "batch00": {"SUBJ_AAA": 2, "SUBJ_BBB": 3},
        "batch01": {"SUBJ_CCC": 2},
    }
    for batch, subs in subjects.items():
        for uid, n_vols in subs.items():
            img_dir = mri_dir / batch / uid / img_subdir
            img_dir.mkdir(parents=True)
            for i in range(n_vols):
                data = np.random.randn(8, 8, 8).astype(np.float32)
                img = nib.Nifti1Image(data, affine=np.eye(4))
                img.to_filename(str(img_dir / f"{uid}_series{i}.nii.gz"))

    jsonl_path = tmp_path / "findings.jsonl"
    with open(jsonl_path, "w") as f:
        for batch, subs in subjects.items():
            for uid in subs:
                f.write(json.dumps({
                    "volume_name": uid,
                    "valid_json": True,
                    "extracted_sentences": ["s1", "s2"],
                }) + "\n")

    return {
        "mri_dir": str(mri_dir),
        "jsonl_path": str(jsonl_path),
        "subjects": subjects,
    }


class TestSpaceToImgSubdirMapping:
    def test_mapping_contents(self):
        """The mapping must cover the three published spaces."""
        assert SPACE_TO_IMG_SUBDIR == {
            'native_space': 'img',
            'coreg_space': 'coreg_img',
            'atlas_space': 'atlas_img',
        }


class TestLayout2CoregDiscovery:
    """Verify Layout 2 picks the correct image subdir for each space."""

    def test_coreg_space_finds_coreg_img(self, tmp_path):
        """MR-RATE-coreg HF layout: <uid>/coreg_img/*.nii.gz must be discovered."""
        fx = _make_hf_layout(tmp_path, "coreg_img")
        ds = MRReportDataset(
            data_folder=fx["mri_dir"],
            jsonl_file=fx["jsonl_path"],
            space="coreg_space",
            max_sentences_per_image=5,
        )
        assert len(ds) == 3
        ids = {s["subject_id"] for s in ds.samples}
        assert ids == {"SUBJ_AAA", "SUBJ_BBB", "SUBJ_CCC"}
        for s in ds.samples:
            for p in s["image_paths"]:
                assert os.sep + "coreg_img" + os.sep in p

    def test_atlas_space_finds_atlas_img(self, tmp_path):
        """MR-RATE-atlas HF layout: <uid>/atlas_img/*.nii.gz must be discovered."""
        fx = _make_hf_layout(tmp_path, "atlas_img")
        ds = MRReportDataset(
            data_folder=fx["mri_dir"],
            jsonl_file=fx["jsonl_path"],
            space="atlas_space",
            max_sentences_per_image=5,
        )
        assert len(ds) == 3
        for s in ds.samples:
            for p in s["image_paths"]:
                assert os.sep + "atlas_img" + os.sep in p

    def test_native_space_still_finds_img(self, tmp_path):
        """Regression: native_space must still resolve to plain img/."""
        fx = _make_hf_layout(tmp_path, "img")
        ds = MRReportDataset(
            data_folder=fx["mri_dir"],
            jsonl_file=fx["jsonl_path"],
            space="native_space",
            max_sentences_per_image=5,
        )
        assert len(ds) == 3

    def test_coreg_space_on_native_layout_finds_nothing(self, tmp_path):
        """Pointing coreg_space at a native (img/) tree must yield no samples."""
        fx = _make_hf_layout(tmp_path, "img")
        ds = MRReportDataset(
            data_folder=fx["mri_dir"],
            jsonl_file=fx["jsonl_path"],
            space="coreg_space",
            max_sentences_per_image=5,
        )
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# Rare-pathology rebalancing (weighted sampling) tests
# ---------------------------------------------------------------------------
def _write_labels_csv(path, label_columns, rows):
    """Write a pathology-labels CSV with study_uid + binary columns."""
    with open(path, 'w') as f:
        f.write("study_uid," + ",".join(label_columns) + "\n")
        for uid, labels in rows.items():
            f.write(uid + "," + ",".join(str(int(x)) for x in labels) + "\n")


class TestRebalanceWeights:
    """Verify per-subject sampling weights from inverse-prevalence rebalancing."""

    @pytest.fixture
    def labeled_dataset(self, synthetic_dataset, tmp_path):
        """4 subjects, 3 pathologies. AAA has a rare positive, others negative or common."""
        labels_path = tmp_path / "labels.csv"
        # Prevalence: rare=1/4=0.25, common=3/4=0.75, never=0/4=0
        _write_labels_csv(
            labels_path,
            label_columns=["rare", "common", "never"],
            rows={
                "SUBJ_AAA": [1, 1, 0],   # has rare
                "SUBJ_BBB": [0, 1, 0],   # common only
                "SUBJ_CCC": [0, 1, 0],   # common only
                "SUBJ_DDD": [0, 0, 0],   # all-negative
            },
        )
        return {**synthetic_dataset, "labels_path": str(labels_path)}

    def test_default_no_weights(self, synthetic_dataset):
        """Without rebalancing args, sample_weights is None."""
        ds = MRReportDataset(
            data_folder=synthetic_dataset["mri_dir"],
            jsonl_file=synthetic_dataset["jsonl_path"],
        )
        assert ds.sample_weights is None

    def test_inverse_freq_rare_outweighs_common(self, labeled_dataset):
        ds = MRReportDataset(
            data_folder=labeled_dataset["mri_dir"],
            jsonl_file=labeled_dataset["jsonl_path"],
            pathology_labels_csv=labeled_dataset["labels_path"],
            rebalance_strategy="inverse_freq",
            rebalance_base_weight=1.0,
        )
        assert ds.sample_weights is not None
        assert ds.sample_weights.shape == (4,)
        idx = {s["subject_id"]: i for i, s in enumerate(ds.samples)}
        # rare prevalence=0.25 -> inv=4; common prevalence=0.75 -> inv=4/3
        # AAA (rare+common): 1 + 4 + 4/3 ≈ 6.333
        # BBB,CCC (common):  1 + 4/3 ≈ 2.333
        # DDD (none):        1
        w = ds.sample_weights
        assert w[idx["SUBJ_AAA"]].item() == pytest.approx(6.333, abs=1e-2)
        assert w[idx["SUBJ_BBB"]].item() == pytest.approx(2.333, abs=1e-2)
        assert w[idx["SUBJ_DDD"]].item() == pytest.approx(1.0, abs=1e-3)
        # Rare subject must have strictly higher weight than common-only
        assert w[idx["SUBJ_AAA"]] > w[idx["SUBJ_BBB"]]

    def test_max_inverse_freq(self, labeled_dataset):
        ds = MRReportDataset(
            data_folder=labeled_dataset["mri_dir"],
            jsonl_file=labeled_dataset["jsonl_path"],
            pathology_labels_csv=labeled_dataset["labels_path"],
            rebalance_strategy="max_inverse_freq",
            rebalance_base_weight=1.0,
        )
        idx = {s["subject_id"]: i for i, s in enumerate(ds.samples)}
        # AAA: max(1, max(rare=4, common=4/3)) = 4
        # BBB,CCC: max(1, 4/3) ≈ 1.333
        # DDD: max(1, 0) = 1
        w = ds.sample_weights
        assert w[idx["SUBJ_AAA"]].item() == pytest.approx(4.0, abs=1e-3)
        assert w[idx["SUBJ_BBB"]].item() == pytest.approx(4.0 / 3.0, abs=1e-3)
        assert w[idx["SUBJ_DDD"]].item() == pytest.approx(1.0, abs=1e-3)

    def test_sqrt_inverse_freq(self, labeled_dataset):
        ds = MRReportDataset(
            data_folder=labeled_dataset["mri_dir"],
            jsonl_file=labeled_dataset["jsonl_path"],
            pathology_labels_csv=labeled_dataset["labels_path"],
            rebalance_strategy="sqrt_inverse_freq",
        )
        idx = {s["subject_id"]: i for i, s in enumerate(ds.samples)}
        # AAA: 1 + sqrt(4) + sqrt(4/3) ≈ 1 + 2 + 1.155 = 4.155
        w = ds.sample_weights
        assert w[idx["SUBJ_AAA"]].item() == pytest.approx(
            1.0 + np.sqrt(4.0) + np.sqrt(4.0 / 3.0), abs=1e-3
        )

    def test_unknown_strategy_raises(self, labeled_dataset):
        with pytest.raises(ValueError, match="rebalance_strategy"):
            MRReportDataset(
                data_folder=labeled_dataset["mri_dir"],
                jsonl_file=labeled_dataset["jsonl_path"],
                pathology_labels_csv=labeled_dataset["labels_path"],
                rebalance_strategy="bogus",
            )

    def test_unlabeled_subjects_get_base_weight(self, synthetic_dataset, tmp_path):
        """Subjects missing from the CSV must get the base weight, not 0."""
        labels_path = tmp_path / "labels.csv"
        _write_labels_csv(
            labels_path,
            label_columns=["rare"],
            rows={"SUBJ_AAA": [1]},  # only one subject labeled
        )
        ds = MRReportDataset(
            data_folder=synthetic_dataset["mri_dir"],
            jsonl_file=synthetic_dataset["jsonl_path"],
            pathology_labels_csv=str(labels_path),
            rebalance_strategy="inverse_freq",
            rebalance_base_weight=2.5,
        )
        idx = {s["subject_id"]: i for i, s in enumerate(ds.samples)}
        for uid in ("SUBJ_BBB", "SUBJ_CCC", "SUBJ_DDD"):
            assert ds.sample_weights[idx[uid]].item() == pytest.approx(2.5, abs=1e-4)
        # AAA has the only positive => prevalence=1.0 => inv=1.0 => weight=base+1
        assert ds.sample_weights[idx["SUBJ_AAA"]].item() == pytest.approx(3.5, abs=1e-4)

    def test_get_weighted_sampler_smoke(self, labeled_dataset):
        """WeightedRandomSampler should draw rare subject more often than not."""
        ds = MRReportDataset(
            data_folder=labeled_dataset["mri_dir"],
            jsonl_file=labeled_dataset["jsonl_path"],
            pathology_labels_csv=labeled_dataset["labels_path"],
            rebalance_strategy="inverse_freq",
        )
        sampler = ds.get_weighted_sampler(num_samples=10_000)
        idx = {s["subject_id"]: i for i, s in enumerate(ds.samples)}
        counts = {uid: 0 for uid in idx}
        for j in sampler:
            for uid, i in idx.items():
                if j == i:
                    counts[uid] += 1
                    break
        # AAA has highest weight => most-drawn; DDD has weight=1 => least.
        most_drawn = max(counts, key=counts.get)
        least_drawn = min(counts, key=counts.get)
        assert most_drawn == "SUBJ_AAA"
        assert least_drawn == "SUBJ_DDD"

    def test_get_weighted_sampler_without_weights_raises(self, synthetic_dataset):
        ds = MRReportDataset(
            data_folder=synthetic_dataset["mri_dir"],
            jsonl_file=synthetic_dataset["jsonl_path"],
        )
        with pytest.raises(RuntimeError, match="sample_weights"):
            ds.get_weighted_sampler()


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
