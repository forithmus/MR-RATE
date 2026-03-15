"""Tests for the MRRATE main model: initialization, forward pass, loss, inference."""
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from unittest.mock import patch
from mr_rate import MRRATE, l2norm, cast_tuple, exists


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_exists_true(self):
        assert exists(0) is True
        assert exists("") is True
        assert exists([]) is True

    def test_exists_false(self):
        assert exists(None) is False

    def test_l2norm_unit_vectors(self):
        t = torch.randn(4, 16)
        normed = l2norm(t)
        norms = normed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_l2norm_preserves_direction(self):
        t = torch.tensor([[3.0, 4.0]])
        normed = l2norm(t)
        assert normed[0, 0] > 0 and normed[0, 1] > 0  # same sign

    def test_l2norm_zero_vector(self):
        t = torch.zeros(1, 16)
        normed = l2norm(t)
        assert not torch.isnan(normed).any()  # eps prevents NaN

    def test_cast_tuple_scalar(self):
        assert cast_tuple(5, 3) == (5, 5, 5)

    def test_cast_tuple_already_tuple(self):
        assert cast_tuple((1, 2, 3), 3) == (1, 2, 3)

    def test_cast_tuple_list(self):
        assert cast_tuple([1, 2], 2) == [1, 2]


# ---------------------------------------------------------------------------
# MRRATE model initialization tests
# ---------------------------------------------------------------------------
class TestMRRATEInit:
    def test_init_all_fusion_modes(self, mock_image_encoder, mock_text_encoder,
                                    dim_text, dim_image, dim_latent):
        for fusion_mode in ["early", "mid_cnn", "late", "late_attn"]:
            model = MRRATE(
                image_encoder=mock_image_encoder,
                text_encoder=mock_text_encoder,
                dim_text=dim_text,
                dim_image=dim_image,
                dim_latent=dim_latent,
                fusion_mode=fusion_mode,
            )
            assert model.fusion_mode == fusion_mode
            assert model.dim_latent == dim_latent

    def test_init_all_pooling_strategies(self, mock_image_encoder, mock_text_encoder,
                                          dim_text, dim_image, dim_latent):
        for strategy in ["simple_attn", "cross_attn", "gated"]:
            model = MRRATE(
                image_encoder=mock_image_encoder,
                text_encoder=mock_text_encoder,
                dim_text=dim_text,
                dim_image=dim_image,
                dim_latent=dim_latent,
                pooling_strategy=strategy,
            )
            assert model.pooling_strategy == strategy
            assert hasattr(model, 'recon_pool')

    def test_init_invalid_pooling_raises(self, mock_image_encoder, mock_text_encoder,
                                          dim_text, dim_image, dim_latent):
        with pytest.raises(ValueError, match="Unknown pooling_strategy"):
            MRRATE(
                image_encoder=mock_image_encoder,
                text_encoder=mock_text_encoder,
                dim_text=dim_text,
                dim_image=dim_image,
                dim_latent=dim_latent,
                pooling_strategy="invalid",
            )

    def test_init_no_image_encoder_raises(self, mock_text_encoder, dim_text, dim_latent):
        with pytest.raises(ValueError, match="image_encoder"):
            MRRATE(
                text_encoder=mock_text_encoder,
                dim_text=dim_text,
                dim_latent=dim_latent,
            )

    def test_projection_layers_created(self, mock_image_encoder, mock_text_encoder,
                                        dim_text, dim_image, dim_latent):
        model = MRRATE(
            image_encoder=mock_image_encoder,
            text_encoder=mock_text_encoder,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
        )
        assert model.to_text_latent.in_features == dim_text
        assert model.to_text_latent.out_features == dim_latent
        assert model.to_visual_latent.in_features == dim_image
        assert model.to_visual_latent.out_features == dim_latent

    def test_logit_temperature_initialized(self, mock_image_encoder, mock_text_encoder,
                                            dim_text, dim_image, dim_latent):
        model = MRRATE(
            image_encoder=mock_image_encoder,
            text_encoder=mock_text_encoder,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
        )
        expected = np.log(1 / 0.07)
        assert torch.allclose(model.logit_temperature, torch.tensor([expected]), atol=1e-4)


# ---------------------------------------------------------------------------
# Forward pass tests (inference mode, no loss)
# ---------------------------------------------------------------------------
class TestMRRATEInference:
    @pytest.fixture
    def model_early(self, mock_image_encoder, mock_text_encoder, dim_text, dim_image, dim_latent):
        return MRRATE(
            image_encoder=mock_image_encoder,
            text_encoder=mock_text_encoder,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
            fusion_mode="early",
        )

    @pytest.fixture
    def model_late_attn(self, mock_image_encoder, mock_text_encoder, dim_text, dim_image, dim_latent):
        return MRRATE(
            image_encoder=mock_image_encoder,
            text_encoder=mock_text_encoder,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
            fusion_mode="late_attn",
            pooling_strategy="simple_attn",
        )

    def test_inference_output_shape(self, model_early, dummy_image, dummy_text_input, dim_latent):
        model_early.eval()
        with torch.no_grad():
            out = model_early(
                text_input=dummy_text_input,
                image=dummy_image,
                device='cpu',
                return_loss=False,
            )
        b = dummy_image.shape[0]
        assert out.shape == (b, dim_latent)

    def test_inference_output_normalized(self, model_early, dummy_image, dummy_text_input):
        model_early.eval()
        with torch.no_grad():
            out = model_early(
                text_input=dummy_text_input,
                image=dummy_image,
                device='cpu',
                return_loss=False,
            )
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_inference_no_nan(self, model_early, dummy_image, dummy_text_input):
        model_early.eval()
        with torch.no_grad():
            out = model_early(
                text_input=dummy_text_input,
                image=dummy_image,
                device='cpu',
                return_loss=False,
            )
        assert not torch.isnan(out).any()

    def test_inference_late_attn(self, model_late_attn, dummy_image, dummy_text_input, dim_latent):
        model_late_attn.eval()
        with torch.no_grad():
            out = model_late_attn(
                text_input=dummy_text_input,
                image=dummy_image,
                device='cpu',
                return_loss=False,
            )
        b = dummy_image.shape[0]
        assert out.shape == (b, dim_latent)
        assert not torch.isnan(out).any()

    def test_default_volume_mask(self, model_early, dummy_image, dummy_text_input):
        """When real_volume_mask is None, model should create one internally."""
        model_early.eval()
        with torch.no_grad():
            out = model_early(
                text_input=dummy_text_input,
                image=dummy_image,
                device='cpu',
                real_volume_mask=None,
                return_loss=False,
            )
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Training loss tests
# ---------------------------------------------------------------------------
class TestMRRATELoss:
    @pytest.fixture
    def model(self, mock_image_encoder, mock_text_encoder, dim_text, dim_image, dim_latent):
        return MRRATE(
            image_encoder=mock_image_encoder,
            text_encoder=mock_text_encoder,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
            fusion_mode="late_attn",
            pooling_strategy="simple_attn",
        )

    def test_loss_is_scalar(self, model, dummy_image, dummy_text_input, real_volume_mask):
        model.train()
        loss = model(
            text_input=dummy_text_input,
            image=dummy_image,
            device='cpu',
            real_volume_mask=real_volume_mask,
            num_sentences_per_image=2,
            return_loss=True,
        )
        assert loss.dim() == 0  # scalar

    def test_loss_is_finite(self, model, dummy_image, dummy_text_input, real_volume_mask):
        model.train()
        loss = model(
            text_input=dummy_text_input,
            image=dummy_image,
            device='cpu',
            real_volume_mask=real_volume_mask,
            num_sentences_per_image=2,
            return_loss=True,
        )
        assert torch.isfinite(loss).all()

    def test_loss_is_positive(self, model, dummy_image, dummy_text_input, real_volume_mask):
        model.train()
        loss = model(
            text_input=dummy_text_input,
            image=dummy_image,
            device='cpu',
            real_volume_mask=real_volume_mask,
            num_sentences_per_image=2,
            return_loss=True,
        )
        assert loss.item() > 0

    def test_loss_backward(self, model, dummy_image, dummy_text_input, real_volume_mask):
        model.train()
        loss = model(
            text_input=dummy_text_input,
            image=dummy_image,
            device='cpu',
            real_volume_mask=real_volume_mask,
            num_sentences_per_image=2,
            return_loss=True,
        )
        loss.backward()
        # Check that at least some parameters got gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_loss_with_sentence_mask(self, model, dummy_image, dummy_text_input, real_volume_mask):
        model.train()
        # 2 images * 2 sentences = 4 sentences, mask out last one
        sentence_mask = torch.tensor([True, True, True, False])
        loss = model(
            text_input=dummy_text_input,
            image=dummy_image,
            device='cpu',
            real_volume_mask=real_volume_mask,
            num_sentences_per_image=2,
            sentence_mask=sentence_mask,
            return_loss=True,
        )
        assert torch.isfinite(loss).all()

    def test_loss_with_partial_volume_mask(self, model, dummy_image, dummy_text_input):
        """Loss should work when some volumes are padded (masked out)."""
        model.train()
        mask = torch.zeros(2, 4, dtype=torch.bool)
        mask[:, :2] = True  # only 2 of 4 volumes valid
        loss = model(
            text_input=dummy_text_input,
            image=dummy_image,
            device='cpu',
            real_volume_mask=mask,
            num_sentences_per_image=2,
            return_loss=True,
        )
        assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# _compute_logit_matrix tests
# ---------------------------------------------------------------------------
class TestComputeLogitMatrix:
    @pytest.fixture
    def model(self, mock_image_encoder, mock_text_encoder, dim_text, dim_image, dim_latent):
        return MRRATE(
            image_encoder=mock_image_encoder,
            text_encoder=mock_text_encoder,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
        )

    def test_output_shape(self, model, dim_latent):
        B, N_tokens, N_T = 2, 100, 4
        visual_tokens = torch.randn(B, N_tokens, dim_latent)
        text_latents = torch.randn(N_T, dim_latent)
        token_mask = torch.ones(B, N_tokens, dtype=torch.bool)
        temp = torch.tensor(14.3)

        logits = model._compute_logit_matrix(visual_tokens, text_latents, token_mask, temp)
        assert logits.shape == (N_T, B)

    def test_masked_tokens_ignored(self, model, dim_latent):
        """Padded tokens (mask=False) should not affect the output."""
        B, N_tokens, N_T = 1, 50, 2
        visual_tokens = torch.randn(B, N_tokens, dim_latent)
        text_latents = torch.randn(N_T, dim_latent)
        temp = torch.tensor(14.3)

        # All tokens valid
        mask_all = torch.ones(B, N_tokens, dtype=torch.bool)
        logits_all = model._compute_logit_matrix(visual_tokens, text_latents, mask_all, temp)

        # Add 50 zero-padded tokens with mask=False
        padded_tokens = torch.cat([visual_tokens, torch.zeros(B, 50, dim_latent)], dim=1)
        padded_mask = torch.cat([mask_all, torch.zeros(B, 50, dtype=torch.bool)], dim=1)
        logits_padded = model._compute_logit_matrix(padded_tokens, text_latents, padded_mask, temp)

        assert torch.allclose(logits_all, logits_padded, atol=1e-4)

    def test_output_clamped(self, model, dim_latent):
        B, N_tokens, N_T = 1, 10, 2
        visual_tokens = torch.randn(B, N_tokens, dim_latent) * 100
        text_latents = torch.randn(N_T, dim_latent) * 100
        token_mask = torch.ones(B, N_tokens, dtype=torch.bool)
        temp = torch.tensor(100.0)

        logits = model._compute_logit_matrix(visual_tokens, text_latents, token_mask, temp)
        assert logits.max() <= 100.0
        assert logits.min() >= -100.0

    def test_nan_inputs_handled(self, model, dim_latent):
        B, N_tokens, N_T = 1, 10, 2
        visual_tokens = torch.randn(B, N_tokens, dim_latent)
        visual_tokens[0, 0, 0] = float('nan')
        text_latents = torch.randn(N_T, dim_latent)
        token_mask = torch.ones(B, N_tokens, dtype=torch.bool)
        temp = torch.tensor(14.3)

        logits = model._compute_logit_matrix(visual_tokens, text_latents, token_mask, temp)
        assert not torch.isnan(logits).any()


# ---------------------------------------------------------------------------
# Token padding tests
# ---------------------------------------------------------------------------
class TestTokenPadding:
    @pytest.fixture
    def model(self, mock_image_encoder, mock_text_encoder, dim_text, dim_image, dim_latent):
        return MRRATE(
            image_encoder=mock_image_encoder,
            text_encoder=mock_text_encoder,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
            fusion_mode="early",
        )

    def test_padding_applied(self, model, dummy_image, real_volume_mask, dim_latent):
        """Tokens should be padded to MAX_TOKENS=18432."""
        model.eval()
        vis_proj = model.to_visual_latent
        tokens, mask = model._encode_visual_tokens(
            dummy_image, real_volume_mask, vis_proj
        )
        assert tokens.shape[1] == 18432
        assert mask.shape[1] == 18432

    def test_mask_correct(self, model, dummy_image, real_volume_mask, dim_latent):
        """Real tokens should have mask=True, padding should have mask=False."""
        model.eval()
        vis_proj = model.to_visual_latent
        tokens, mask = model._encode_visual_tokens(
            dummy_image, real_volume_mask, vis_proj
        )
        # Some tokens should be True (real) and some False (padding)
        assert mask.any()  # at least some real tokens
        assert not mask.all()  # should have padding since 16 < 18432

    def test_padding_is_zero(self, model, dummy_image, real_volume_mask, dim_latent):
        """Padded token values should be exactly zero."""
        model.eval()
        vis_proj = model.to_visual_latent
        tokens, mask = model._encode_visual_tokens(
            dummy_image, real_volume_mask, vis_proj
        )
        padded_tokens = tokens[:, ~mask[0]]
        assert (padded_tokens == 0).all()


# ---------------------------------------------------------------------------
# State dict / load tests
# ---------------------------------------------------------------------------
class TestModelSerialization:
    def test_state_dict_roundtrip(self, mock_image_encoder, mock_text_encoder,
                                   dim_text, dim_image, dim_latent):
        model1 = MRRATE(
            image_encoder=mock_image_encoder,
            text_encoder=mock_text_encoder,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
        )
        sd = model1.state_dict()
        model2 = MRRATE(
            image_encoder=mock_image_encoder,
            text_encoder=mock_text_encoder,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
        )
        model2.load_state_dict(sd)
        for (k1, v1), (k2, v2) in zip(model1.state_dict().items(), model2.state_dict().items()):
            assert k1 == k2
            assert torch.equal(v1, v2)

    def test_load_strips_module_prefix(self, mock_image_encoder, mock_text_encoder,
                                        dim_text, dim_image, dim_latent, tmp_path):
        model = MRRATE(
            image_encoder=mock_image_encoder,
            text_encoder=mock_text_encoder,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
        )
        # Save with 'module.' prefix (simulating DDP checkpoint)
        sd = {"module." + k: v for k, v in model.state_dict().items()}
        path = tmp_path / "ckpt.pt"
        torch.save(sd, path)

        model2 = MRRATE(
            image_encoder=mock_image_encoder,
            text_encoder=mock_text_encoder,
            dim_text=dim_text,
            dim_image=dim_image,
            dim_latent=dim_latent,
        )
        model2.load(str(path))
        # Should load without error
