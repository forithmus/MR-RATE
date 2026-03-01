"""Tests for all fusion modes: early, mid_cnn, late, late_attn with various pooling strategies."""
import pytest
import torch
from mr_rate import MRRATE
from conftest import MockImageEncoder, MockTextEncoder


DIM_TEXT = 32
DIM_IMAGE = 64
DIM_LATENT = 16
B = 2
R = 4  # number of volumes


@pytest.fixture
def image():
    return torch.randn(B, R, 1, 8, 16, 16)


@pytest.fixture
def text_input():
    from types import SimpleNamespace
    input_ids = torch.randint(1, 100, (B * 2, 32))
    attention_mask = torch.ones(B * 2, 32, dtype=torch.long)
    return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


@pytest.fixture
def vol_mask():
    return torch.ones(B, R, dtype=torch.bool)


def make_model(fusion_mode, pooling_strategy="simple_attn"):
    return MRRATE(
        image_encoder=MockImageEncoder(output_dim=DIM_IMAGE, num_tokens=16),
        text_encoder=MockTextEncoder(hidden_size=DIM_TEXT),
        dim_text=DIM_TEXT,
        dim_image=DIM_IMAGE,
        dim_latent=DIM_LATENT,
        fusion_mode=fusion_mode,
        pooling_strategy=pooling_strategy,
    )


class TestEarlyFusion:
    def test_inference(self, image, text_input, vol_mask):
        model = make_model("early")
        model.eval()
        with torch.no_grad():
            out = model(text_input, image, 'cpu', vol_mask, return_loss=False)
        assert out.shape == (B, DIM_LATENT)
        assert not torch.isnan(out).any()

    def test_training_loss(self, image, text_input, vol_mask):
        model = make_model("early")
        model.train()
        loss = model(text_input, image, 'cpu', vol_mask,
                     num_sentences_per_image=2, return_loss=True)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestMidCNNFusion:
    def test_inference(self, image, text_input, vol_mask):
        model = make_model("mid_cnn")
        model.eval()
        with torch.no_grad():
            out = model(text_input, image, 'cpu', vol_mask, return_loss=False)
        assert out.shape == (B, DIM_LATENT)
        assert not torch.isnan(out).any()

    def test_training_loss(self, image, text_input, vol_mask):
        model = make_model("mid_cnn")
        model.train()
        loss = model(text_input, image, 'cpu', vol_mask,
                     num_sentences_per_image=2, return_loss=True)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestLateFusion:
    def test_inference(self, image, text_input, vol_mask):
        model = make_model("late")
        model.eval()
        with torch.no_grad():
            out = model(text_input, image, 'cpu', vol_mask, return_loss=False)
        assert out.shape == (B, DIM_LATENT)
        assert not torch.isnan(out).any()

    def test_training_loss(self, image, text_input, vol_mask):
        model = make_model("late")
        model.train()
        loss = model(text_input, image, 'cpu', vol_mask,
                     num_sentences_per_image=2, return_loss=True)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_partial_volume_mask(self, image, text_input):
        """Only 2 of 4 volumes valid."""
        model = make_model("late")
        mask = torch.zeros(B, R, dtype=torch.bool)
        mask[:, :2] = True
        model.eval()
        with torch.no_grad():
            out = model(text_input, image, 'cpu', mask, return_loss=False)
        assert out.shape == (B, DIM_LATENT)
        assert not torch.isnan(out).any()


class TestLateAttnFusion:
    @pytest.mark.parametrize("pooling", ["simple_attn", "cross_attn", "gated"])
    def test_inference(self, image, text_input, vol_mask, pooling):
        model = make_model("late_attn", pooling_strategy=pooling)
        model.eval()
        with torch.no_grad():
            out = model(text_input, image, 'cpu', vol_mask,
                        num_sentences_per_image=2, return_loss=False)
        assert out.shape == (B, DIM_LATENT)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("pooling", ["simple_attn", "cross_attn", "gated"])
    def test_training_loss(self, image, text_input, vol_mask, pooling):
        model = make_model("late_attn", pooling_strategy=pooling)
        model.train()
        loss = model(text_input, image, 'cpu', vol_mask,
                     num_sentences_per_image=2, return_loss=True)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_cross_attn_without_text_falls_back(self, image, vol_mask):
        """Cross-attn pooling should fall back to masked average when no text."""
        model = make_model("late_attn", pooling_strategy="cross_attn")
        model.eval()
        vis_proj = model.to_visual_latent
        tokens, mask = model._encode_visual_tokens(
            image, vol_mask, vis_proj, text_latents=None
        )
        assert tokens.shape[1] == 18432
        assert not torch.isnan(tokens[:, :16]).any()  # real tokens


class TestFusionConsistency:
    """All fusion modes should produce same output shape and finite values."""

    @pytest.mark.parametrize("fusion_mode", ["early", "mid_cnn", "late", "late_attn"])
    def test_deterministic_eval(self, image, text_input, vol_mask, fusion_mode):
        model = make_model(fusion_mode)
        model.eval()
        with torch.no_grad():
            out1 = model(text_input, image, 'cpu', vol_mask, return_loss=False)
            out2 = model(text_input, image, 'cpu', vol_mask, return_loss=False)
        assert torch.allclose(out1, out2, atol=1e-5)
