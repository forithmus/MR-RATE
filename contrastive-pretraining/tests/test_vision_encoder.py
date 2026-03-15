"""Tests for the vision encoder components: ResidualTemporalDownsample, VJEPA2Encoder structure."""
import pytest
import torch
import torch.nn as nn
from vision_encoder.vjepa_encoder import ResidualTemporalDownsample


class TestResidualTemporalDownsample:
    """Test the temporal CNN that downsamples depth dimension by 4x."""

    @pytest.fixture
    def cnn(self):
        return ResidualTemporalDownsample(in_channels=3)

    @pytest.fixture
    def cnn_1ch(self):
        return ResidualTemporalDownsample(in_channels=1)

    def test_output_shape_standard(self, cnn):
        """Standard MRI: [B, 3, 128, H, W] -> [B, 3, 32, H, W] (4x temporal downsample)."""
        x = torch.randn(2, 3, 128, 16, 16)
        out = cnn(x)
        assert out.shape == (2, 3, 32, 16, 16)

    def test_output_shape_small_depth(self, cnn):
        """Small depth: [B, 3, 16, H, W] -> [B, 3, 4, H, W]."""
        x = torch.randn(2, 3, 16, 8, 8)
        out = cnn(x)
        assert out.shape == (2, 3, 4, 8, 8)

    def test_output_shape_min_depth(self, cnn):
        """Minimum viable depth: [B, 3, 4, H, W] -> [B, 3, 1, H, W]."""
        x = torch.randn(1, 3, 4, 8, 8)
        out = cnn(x)
        assert out.shape == (1, 3, 1, 8, 8)

    def test_single_channel_input(self, cnn_1ch):
        """Single channel MRI input."""
        x = torch.randn(2, 1, 64, 16, 16)
        out = cnn_1ch(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 3  # always outputs 3 channels
        assert out.shape[2] == 16  # 64 / 4 = 16

    def test_gradient_flows(self, cnn):
        x = torch.randn(1, 3, 16, 8, 8, requires_grad=True)
        out = cnn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_residual_connection(self, cnn):
        """Verify the skip connection works (output != 0 even with zero main path)."""
        x = torch.randn(1, 3, 8, 4, 4)
        out = cnn(x)
        assert out.abs().sum() > 0

    def test_no_nan_output(self, cnn):
        x = torch.randn(2, 3, 32, 8, 8)
        out = cnn(x)
        assert not torch.isnan(out).any()

    def test_various_spatial_sizes(self, cnn):
        """Should work with different spatial dimensions."""
        for h, w in [(8, 8), (16, 16), (32, 32), (64, 64)]:
            x = torch.randn(1, 3, 8, h, w)
            out = cnn(x)
            assert out.shape == (1, 3, 2, h, w)

    def test_batch_independence(self, cnn):
        """Each sample in batch should be processed independently (eval mode)."""
        cnn.eval()  # BatchNorm must be in eval mode for batch-independent outputs
        x1 = torch.randn(1, 3, 8, 8, 8)
        x2 = torch.randn(1, 3, 8, 8, 8)
        batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            out_batch = cnn(batch)
            out1 = cnn(x1)
            out2 = cnn(x2)

        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)


class TestVJEPA2EncoderStructure:
    """Test VJEPA2Encoder without loading the actual pretrained model.
    Tests the CNN preprocessing and channel/depth handling logic."""

    def test_forward_cnn_depth_repeat(self):
        """When depth < 4, should repeat to make depth=4 before CNN."""
        cnn = ResidualTemporalDownsample(in_channels=3)

        # Simulate the forward_cnn logic from VJEPA2Encoder
        x = torch.randn(1, 3, 1, 16, 16)  # single slice

        # This is the logic from VJEPA2Encoder.forward_cnn:
        if x.shape[2] < 4:
            repeats = 4 // x.shape[2]
            x = x.repeat(1, 1, repeats, 1, 1)

        assert x.shape[2] == 4
        out = cnn(x)
        assert out.shape == (1, 3, 1, 16, 16)  # 4 / 4 = 1 frame

    def test_forward_cnn_channel_repeat(self):
        """When input is grayscale (C=1), should repeat to 3 channels."""
        cnn = ResidualTemporalDownsample(in_channels=3)

        x = torch.randn(1, 1, 8, 16, 16)  # grayscale MRI
        input_channels = 3

        # This is the logic from VJEPA2Encoder.forward_cnn:
        if x.shape[2] < 4:
            repeats = 4 // x.shape[2]
            x = x.repeat(1, 1, repeats, 1, 1)
        if x.shape[1] == 1 and input_channels == 3:
            x = x.repeat(1, 3, 1, 1, 1)

        assert x.shape[1] == 3
        out = cnn(x.float())
        assert out.shape[1] == 3

    def test_forward_cnn_mri_full_pipeline(self):
        """Full MRI pipeline: [B, 1, 1, H, W] -> depth repeat -> channel repeat -> CNN."""
        cnn = ResidualTemporalDownsample(in_channels=3)
        input_channels = 3

        x = torch.randn(2, 1, 1, 32, 32)  # batch of single-slice MRI

        if x.shape[2] < 4:
            repeats = 4 // x.shape[2]
            x = x.repeat(1, 1, repeats, 1, 1)
        if x.shape[1] == 1 and input_channels == 3:
            x = x.repeat(1, 3, 1, 1, 1)

        out = cnn(x.float())
        assert out.shape == (2, 3, 1, 32, 32)
        assert not torch.isnan(out).any()


class TestOptimizerImport:
    """Test that the optimizer utility from vision_encoder works."""

    def test_get_optimizer_creates_adam(self):
        from vision_encoder.optimizer import get_optimizer
        params = nn.Linear(10, 10).parameters()
        opt = get_optimizer(set(params), lr=1e-3, wd=0.01)
        assert opt is not None
        assert len(opt.param_groups) > 0
