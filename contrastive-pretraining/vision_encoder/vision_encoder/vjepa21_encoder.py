import torch
import torch.nn as nn
from einops import rearrange
import sys
import os


class ResidualTemporalDownsample(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(in_channels=16, out_channels=3, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=True),
        )
        self.skip = nn.Conv3d(in_channels=in_channels, out_channels=3, kernel_size=(1, 1, 1), stride=(4, 1, 1), padding=(0, 0, 0), bias=False)

    def forward(self, x):
        return self.main(x) + self.skip(x)


class VJEPA21Encoder(nn.Module):
    """
    Vision encoder using VJEPA 2.1 ViT-g backbone loaded via torch.hub.

    Key differences from VJEPA2Encoder:
    - Loaded via torch.hub.load('facebookresearch/vjepa2', ...) instead of HuggingFace
    - Has separate 2D/3D patchifiers (multi-modal tokenizer)
    - Uses deep self-supervision with hierarchical features
    - Forward expects [B, T, C, H, W] for video, [B, C, H, W] for images
    """

    def __init__(
            self,
            checkpoint_path: str = None,
            freeze_backbone: bool = True,
            use_lora: bool = True,
            lora_r: int = 64,
            lora_alpha: int = 128,
            lora_dropout: float = 0.05,
            use_temporal_cnn: bool = True,
            train_temporal_cnn: bool = True,
            input_channels: int = 1,
    ):
        super().__init__()

        # Load encoder from torch hub (without pretrained weights - we load manually)
        hub_dir = torch.hub.get_dir()
        repo_dir = os.path.join(hub_dir, "facebookresearch_vjepa2_main")

        # Add the hub repo to sys.path so we can import from it
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        from app.vjepa_2_1.models import vision_transformer as vit_encoder

        # Build the ViT-g encoder (same arch as hub's vjepa2_1_vit_giant_384)
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

        self.output_dim = self.encoder.embed_dim  # 1408

        # Load pretrained weights if checkpoint provided
        if checkpoint_path is not None:
            print(f"Loading VJEPA 2.1 encoder weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            encoder_key = "target_encoder" if "target_encoder" in state_dict else "ema_encoder"
            encoder_state = state_dict[encoder_key]

            # Clean keys: strip module. and backbone. prefixes
            clean_state = {}
            for k, v in encoder_state.items():
                k = k.replace("module.", "").replace("backbone.", "")
                clean_state[k] = v

            missing, unexpected = self.encoder.load_state_dict(clean_state, strict=False)
            print(f"  Loaded {len(clean_state)} keys. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                print(f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"  Unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        # Convert to bfloat16
        self.encoder = self.encoder.to(torch.bfloat16)

        self.use_temporal_cnn = use_temporal_cnn
        self.train_temporal_cnn = train_temporal_cnn
        self.input_channels = input_channels

        if self.use_temporal_cnn:
            self.temporal_cnn = ResidualTemporalDownsample(in_channels=self.input_channels)

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

    def forward_cnn(self, x):
        # x: [B, C, D, H, W]
        if x.shape[2] < 4:
            repeats = 4 // x.shape[2]
            x = x.repeat(1, 1, repeats, 1, 1)

        if x.shape[1] == 1 and self.input_channels == 3:
            x = x.repeat(1, 3, 1, 1, 1)

        return self.temporal_cnn(x.float())

    def forward_transformer(self, x):
        # x: [B, C, T, H, W] from CNN output
        # VJEPA2.1 encoder expects [B, C, T, H, W] for video (5D)
        pixel_values = x.to(dtype=self.dtype)
        # Call the underlying model directly to avoid PEFT injecting
        # text-model kwargs (input_ids, etc.) into VisionTransformer.forward()
        if hasattr(self.encoder, 'base_model'):
            # PEFT-wrapped: call through base_model.model (the actual VisionTransformer)
            return self.encoder.base_model.model(pixel_values)
        return self.encoder(pixel_values)

    def forward(self, x, return_encoded_tokens=True):
        x = self.forward_cnn(x)
        return self.forward_transformer(x)
