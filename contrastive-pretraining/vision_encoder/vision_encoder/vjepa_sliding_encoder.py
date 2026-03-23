"""
VJEPA2 encoder with sliding window over depth instead of temporal CNN.
Chunks the volume along depth, runs VJEPA on each chunk, and mean-pools.
Sequential processing with gradient checkpointing per chunk.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel
from einops import rearrange


class VJEPA2SlidingEncoder(nn.Module):
    def __init__(
            self,
            model_name: str = "facebook/vjepa2-vitg-fpc64-384",
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

        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.output_dim = self.model.config.hidden_size
        self.chunk_size = chunk_size

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["attention.key", "attention.query", "attention.value",
                                "attention.proj", "mlp.fc1", "mlp.fc2"],
                bias="none", task_type="FEATURE_EXTRACTION",
            )
            self.model = get_peft_model(self.model, lora_config)

    def _encode_chunk(self, chunk):
        """Encode a single chunk: [1, 3, T, H, W] -> [1, N, D]"""
        pixel_values = rearrange(chunk, "b c t h w -> b t c h w").to(dtype=self.model.dtype)
        return self.model.get_vision_features(pixel_values)

    def forward(self, x, return_encoded_tokens=True):
        """
        x: [B, C, D, H, W] where B=1, C=1 (grayscale), D=depth slices

        Sequential chunk processing with gradient checkpointing.
        Running mean to avoid storing all outputs.
        """
        B, C, D, H, W = x.shape

        if C == 1:
            x = x.repeat(1, 3, 1, 1, 1)

        x = x.to(dtype=self.model.dtype)

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
