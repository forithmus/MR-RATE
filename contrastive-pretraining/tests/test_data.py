"""Tests for data pipeline: normalizers, dataset loading, collate_fn."""
import pytest
import numpy as np
import torch
from data import (
    ZScoreNormalizer, PercentileNormalizer, MinMaxNormalizer,
    NORMALIZERS, collate_fn,
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
