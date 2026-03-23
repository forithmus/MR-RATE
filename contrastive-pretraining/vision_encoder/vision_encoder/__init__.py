from .vjepa_encoder import VJEPA2Encoder, ResidualTemporalDownsample
from .optimizer import get_optimizer

# VJEPA 2.1 and sliding encoders require torch.hub repo + timm at import time.
# Import lazily to avoid breaking environments that only use VJEPA2.
def __getattr__(name):
    if name == "VJEPA21Encoder":
        from .vjepa21_encoder import VJEPA21Encoder
        return VJEPA21Encoder
    if name == "VJEPA2SlidingEncoder":
        from .vjepa_sliding_encoder import VJEPA2SlidingEncoder
        return VJEPA2SlidingEncoder
    if name == "VJEPA21SlidingEncoder":
        from .vjepa21_sliding_encoder import VJEPA21SlidingEncoder
        return VJEPA21SlidingEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
