"""Cross-sequence attention at each atlas token.

All sequences of a study share one aligned grid, so the token at a given position of every
sequence covers the same tissue. At each position the tokens of the sequences that cover it
attend to each other (pre-norm transformer blocks over the sequence axis, a learned embedding
of each token's sequence kind added to the attention input). Every sequence token stays in the
MIL bag, updated by a residual whose output projections start at zero, so the untrained module
is the identity and the model starts from the per-sequence baseline.

The module has no dropout, so the two replayed MIL passes see identical features.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mr_dino.geometry import SEQUENCE_KINDS


class CrossSequenceBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} is not divisible by {heads} heads")
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        for layer in (self.proj, self.fc2):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, kind: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        n, s, d = x.shape
        q, k, v = self.qkv(self.norm1(x + kind)).reshape(n, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        mask = valid[:, None, None, :].expand(n, 1, s, s)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(attended.transpose(1, 2).reshape(n, s, d))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class CrossSequenceAttention(nn.Module):
    def __init__(self, dim: int, depth: int = 1, heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim, self.depth, self.heads, self.mlp_ratio = dim, depth, heads, mlp_ratio
        self.kind = nn.Embedding(len(SEQUENCE_KINDS), dim)
        nn.init.normal_(self.kind.weight, std=0.02)
        self.blocks = nn.ModuleList(CrossSequenceBlock(dim, heads, mlp_ratio) for _ in range(depth))

    def config(self) -> dict:
        return {"dim": self.dim, "depth": self.depth, "heads": self.heads, "mlp_ratio": self.mlp_ratio}

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor, kinds: torch.Tensor) -> torch.Tensor:
        """``tokens [N, S, D]`` of ``S`` sequences at ``N`` positions, ``valid [N, S]`` marks the
        sequences that cover a position (every row needs one), ``kinds [S]``. Returns ``[N, S, D]``."""
        if tokens.ndim != 3 or valid.shape != tokens.shape[:2] or kinds.shape != tokens.shape[1:2]:
            raise ValueError("Expected tokens [N,S,D], valid [N,S] and kinds [S]")
        if not bool(valid.any(1).all()):
            raise ValueError("Every position needs at least one covering sequence")
        kind = self.kind(kinds)[None]
        x = tokens
        for block in self.blocks:
            x = block(x, kind, valid)
        return x
