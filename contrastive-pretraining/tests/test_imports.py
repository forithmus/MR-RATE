"""Test that all packages and modules can be imported correctly."""
import pytest


class TestDependencies:
    """Verify that all required third-party packages are installed."""

    def test_torch(self):
        import torch
        assert torch.__version__ is not None

    def test_einops(self):
        import einops
        assert einops.__version__ is not None

    def test_transformers(self):
        import transformers
        assert transformers.__version__ is not None

    def test_peft(self):
        import peft
        assert peft.__version__ is not None

    def test_accelerate(self):
        import accelerate
        assert accelerate.__version__ is not None

    def test_numpy(self):
        import numpy
        assert numpy.__version__ is not None

    def test_nibabel(self):
        import nibabel
        assert nibabel.__version__ is not None

    def test_wandb(self):
        import wandb
        assert wandb.__version__ is not None


class TestMRRATEImports:
    """Verify that MR-RATE package modules can be imported."""

    def test_import_mr_rate_package(self):
        from mr_rate import MRRATE
        assert MRRATE is not None

    def test_import_mr_rate_submodules(self):
        from mr_rate import MRRATE, SimpleAttnPool, CrossAttnPool, GatedAttnPool
        assert all(cls is not None for cls in [MRRATE, SimpleAttnPool, CrossAttnPool, GatedAttnPool])

    def test_import_mr_rate_helpers(self):
        from mr_rate import exists, l2norm, cast_tuple, all_gather_batch
        assert exists(1) is True
        assert exists(None) is False


class TestVisionEncoderImports:
    """Verify that vision_encoder package modules can be imported."""

    def test_import_vision_encoder_package(self):
        from vision_encoder import VJEPA2Encoder, ResidualTemporalDownsample, get_optimizer
        assert VJEPA2Encoder is not None
        assert ResidualTemporalDownsample is not None
        assert get_optimizer is not None

    def test_import_vision_encoder_submodules(self):
        from vision_encoder.vjepa_encoder import VJEPA2Encoder, ResidualTemporalDownsample
        from vision_encoder.optimizer import get_optimizer
        assert all(obj is not None for obj in [VJEPA2Encoder, ResidualTemporalDownsample,
                                               get_optimizer])


class TestDataImports:
    """Verify that data module can be imported."""

    def test_import_data_module(self):
        from data import MRReportDataset, collate_fn, cycle
        assert MRReportDataset is not None
        assert collate_fn is not None
        assert cycle is not None

    def test_import_normalizers(self):
        from data import ZScoreNormalizer, PercentileNormalizer, MinMaxNormalizer, NORMALIZERS
        assert ZScoreNormalizer is not None
        assert PercentileNormalizer is not None
        assert MinMaxNormalizer is not None
        assert len(NORMALIZERS) == 3
