"""
Shared fixtures for MR-RATE unit tests.

Provides lightweight mock encoders so tests run on CPU without
downloading multi-GB pretrained models.
"""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Mock Image Encoder (replaces VJEPA2Encoder for unit tests)
# ---------------------------------------------------------------------------
class MockImageEncoder(nn.Module):
    """Mimics the VJEPA2Encoder interface with tiny random weights."""

    def __init__(self, output_dim=64, num_tokens=16):
        super().__init__()
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        # Tiny CNN that maps any [B, C, D, H, W] -> [B, 3, D', H, W]
        self.temporal_cnn = nn.Conv3d(1, 3, kernel_size=1)
        # Projection to mimic transformer output -> [B, num_tokens, output_dim]
        self.proj = nn.Linear(3, output_dim)
        self.dtype = torch.float32

    def forward_cnn(self, x):
        # x: [B, C, D, H, W] -> [B, 3, D', H, W]
        if x.shape[1] != 1:
            x = x[:, :1]
        return self.temporal_cnn(x)

    def forward_transformer(self, x):
        # x: [B, 3, D', H, W] -> [B, num_tokens, output_dim]
        b = x.shape[0]
        flat = x.reshape(b, 3, -1).permute(0, 2, 1)  # [B, spatial, 3]
        # Trim or pad to fixed num_tokens
        if flat.shape[1] > self.num_tokens:
            flat = flat[:, :self.num_tokens]
        elif flat.shape[1] < self.num_tokens:
            pad = torch.zeros(b, self.num_tokens - flat.shape[1], 3, device=x.device)
            flat = torch.cat([flat, pad], dim=1)
        return self.proj(flat)

    def forward(self, x, return_encoded_tokens=True):
        x = self.forward_cnn(x)
        return self.forward_transformer(x)


# ---------------------------------------------------------------------------
# Mock Text Encoder (replaces BiomedVLP-CXR-BERT)
# ---------------------------------------------------------------------------
class MockTextEncoder(nn.Module):
    """Mimics a HuggingFace BERT-style text encoder."""

    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        b, seq_len = input_ids.shape
        # Generate deterministic-ish hidden states from input_ids
        x = input_ids.float().unsqueeze(-1)  # [B, seq_len, 1]
        hidden = self.proj(x)  # [B, seq_len, hidden_size]
        return SimpleNamespace(last_hidden_state=hidden)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def dim_image():
    return 64


@pytest.fixture
def dim_text():
    return 32


@pytest.fixture
def dim_latent():
    return 16


@pytest.fixture
def mock_image_encoder(dim_image):
    return MockImageEncoder(output_dim=dim_image, num_tokens=16)


@pytest.fixture
def mock_text_encoder(dim_text):
    return MockTextEncoder(hidden_size=dim_text)


@pytest.fixture
def dummy_image():
    """Returns a [B=2, R=4, C=1, D=8, H=16, W=16] tensor."""
    return torch.randn(2, 4, 1, 8, 16, 16)


@pytest.fixture
def dummy_text_input(dim_text):
    """Returns a mock tokenizer output with input_ids and attention_mask."""
    # 2 images * 2 sentences_per_image = 4 sentences
    input_ids = torch.randint(1, 100, (4, 32))
    attention_mask = torch.ones(4, 32, dtype=torch.long)
    return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


@pytest.fixture
def real_volume_mask():
    """Returns a [B=2, R=4] boolean mask (all volumes valid)."""
    return torch.ones(2, 4, dtype=torch.bool)
