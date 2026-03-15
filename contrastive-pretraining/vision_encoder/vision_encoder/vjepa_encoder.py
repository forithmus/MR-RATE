import torch
import torch.nn as nn
from transformers import AutoVideoProcessor, AutoModel
from einops import rearrange

class ResidualTemporalDownsample(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # Use BatchNorm3d here - will be converted to SyncBatchNorm in run_train.py
        self.main = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(in_channels=16, out_channels=3, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=True),
        )
        self.skip = nn.Conv3d(in_channels=in_channels, out_channels=3, kernel_size=(1, 1, 1), stride=(4, 1, 1), padding=(0, 0, 0), bias=False)

    def forward(self, x):
        return self.main(x) + self.skip(x)

class VJEPA2Encoder(nn.Module):
    def __init__(
            self,
            model_name: str = "facebook/vjepa2-vitg-fpc64-384",
            freeze_backbone: bool = True,
            use_lora: bool = True,
            lora_r: int = 64,
            lora_alpha: int = 128,
            lora_dropout: float = 0.05,
            use_temporal_cnn: bool = True,
            train_temporal_cnn: bool = True,
            input_channels: int = 3
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.output_dim = self.model.config.hidden_size
        self.use_temporal_cnn = use_temporal_cnn
        self.train_temporal_cnn = train_temporal_cnn
        self.input_channels = input_channels

        if self.use_temporal_cnn:
            self.temporal_cnn = ResidualTemporalDownsample(in_channels=self.input_channels)

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["attention.key", "attention.query", "attention.value", "attention.proj", "mlp.fc1", "mlp.fc2"],
                bias="none", task_type="FEATURE_EXTRACTION",
            )
            self.model = get_peft_model(self.model, lora_config)

    def forward_cnn(self, x):
        # x: [B, C, D, H, W]
        
        # FIX 2: Handle X-ray inputs (Depth=1)
        # The CNN has a total stride of 4 (stride 2 * stride 2). 
        # If D < 4, Conv3d will either crash or output 0 frames.
        # We repeat the single slice 4 times so the CNN outputs exactly 1 temporal frame.
        if x.shape[2] < 4:
            # Repeat the depth dimension (dim=2)
            repeats = 4 // x.shape[2]
            # If shape is [B, C, 1, H, W], this becomes [B, C, 4, H, W]
            x = x.repeat(1, 1, repeats, 1, 1)

        # Handle Channel mismatch (Grayscale to RGB)
        if x.shape[1] == 1 and self.input_channels == 3:
            x = x.repeat(1, 3, 1, 1, 1)
            
        return self.temporal_cnn(x.float())

    def forward_transformer(self, x):
        # x: [B, 3, D', H, W]
        pixel_values = rearrange(x, "b c t h w -> b t c h w").to(dtype=self.model.dtype)
        return self.model.get_vision_features(pixel_values)

    def forward(self, x, return_encoded_tokens=True):
        x = self.forward_cnn(x)
        return self.forward_transformer(x)