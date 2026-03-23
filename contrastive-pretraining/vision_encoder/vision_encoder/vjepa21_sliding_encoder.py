"""
VJEPA 2.1 encoder with sliding window over depth instead of temporal CNN.
Chunks the volume along depth, runs VJEPA2.1 on each chunk, and mean-pools.
Sequential processing with gradient checkpointing per chunk.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import sys
import os


class VJEPA21SlidingEncoder(nn.Module):
    def __init__(
            self,
            checkpoint_path: str = None,
            freeze_backbone: bool = True,
            use_lora: bool = True,
            lora_r: int = 64,
            lora_alpha: int = 128,
            lora_dropout: float = 0.05,
            chunk_size: int = 64,
            input_channels: int = 1,
    ):
        super().__init__()
        assert chunk_size % 2 == 0, "chunk_size must be even (tubelet_size=2)"

        hub_dir = torch.hub.get_dir()
        repo_dir = os.path.join(hub_dir, "facebookresearch_vjepa2_main")
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        from app.vjepa_2_1.models import vision_transformer as vit_encoder

        self.encoder = vit_encoder.vit_giant_xformers(
            patch_size=16,
            img_size=(384, 384),
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )

        self.output_dim = self.encoder.embed_dim
        self.chunk_size = chunk_size

        if checkpoint_path is not None:
            print(f"Loading VJEPA 2.1 encoder weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            encoder_key = "target_encoder" if "target_encoder" in state_dict else "ema_encoder"
            encoder_state = state_dict[encoder_key]

            clean_state = {}
            for k, v in encoder_state.items():
                k = k.replace("module.", "").replace("backbone.", "")
                clean_state[k] = v

            missing, unexpected = self.encoder.load_state_dict(clean_state, strict=False)
            print(f"  Loaded {len(clean_state)} keys. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        self.encoder = self.encoder.to(torch.bfloat16)

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
                bias="none", task_type="FEATURE_EXTRACTION",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype

    def _encode_chunk(self, chunk):
        """Encode a single chunk: [1, 3, T, H, W] -> [1, N, D]"""
        pixel_values = chunk.to(dtype=self.dtype)
        if hasattr(self.encoder, 'base_model'):
            return self.encoder.base_model.model(pixel_values)
        return self.encoder(pixel_values)

    def forward(self, x, return_encoded_tokens=True):
        """
        x: [B, C, D, H, W] where B=1, C=1 (grayscale), D=depth slices

        Sequential chunk processing with gradient checkpointing.
        Running mean to avoid storing all outputs.
        """
        B, C, D, H, W = x.shape

        if C == 1:
            x = x.repeat(1, 3, 1, 1, 1)

        x = x.to(dtype=self.dtype)

        if D % self.chunk_size != 0:
            pad_d = self.chunk_size - (D % self.chunk_size)
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_d))
            D = D + pad_d

        chunks = x.split(self.chunk_size, dim=2)
        num_chunks = len(chunks)

        if not self.training:
            # Inference: batch all chunks for speed (no grad graph = fits in memory)
            chunks_batched = torch.cat(chunks, dim=0)  # [num_chunks, 3, T, H, W]
            all_tokens = self._encode_chunk(chunks_batched)  # [num_chunks, N, D]
            all_tokens = all_tokens.view(B, num_chunks, all_tokens.shape[1], all_tokens.shape[2])
            return all_tokens.mean(dim=1)

        # Training: sequential with gradient checkpointing + running mean
        pooled = None
        for i, chunk in enumerate(chunks):
            tokens = checkpoint(self._encode_chunk, chunk, use_reentrant=False)

            if pooled is None:
                pooled = tokens
            else:
                pooled = pooled + (tokens - pooled) / (i + 1)

        return pooled

    def forward_cnn(self, x):
        return x

    def forward_transformer(self, x):
        return self.forward(x)
