"""DINOv3 ViT for MR with a native 3-D patch embed and physical 3-D RoPE.

The transformer blocks are the official DINOv3 implementation pinned in
``README.md``. Only the image-specific pieces are replaced. The production
patch kernel is (2, 16, 16), so a high-resolution 64x384x384 MR crop produces
32x24x24 == 18,432 patch tokens.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Iterable

import torch
from torch import Tensor, nn

try:
    from dinov3.layers import LayerScale, Mlp, RMSNorm, SelfAttentionBlock, SwiGLUFFN
    from dinov3.utils import named_apply
except ImportError as exc:  # pragma: no cover - actionable cluster error
    raise ImportError(
        "Official DINOv3 is required. Set DINOV3_ROOT and add it to PYTHONPATH; "
        "see scripts/ctdino3d/README.md."
    ) from exc


def _triple(x: int | Iterable[int]) -> tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    out = tuple(int(v) for v in x)
    if len(out) != 3:
        raise ValueError(f"Expected a 3-tuple, got {out}")
    return out


class PatchEmbed3D(nn.Module):
    """(B,C,D,H,W) -> (B,D',H',W',embed_dim)."""

    def __init__(
        self,
        volume_size: tuple[int, int, int] = (64, 384, 384),
        patch_size: tuple[int, int, int] = (2, 16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.volume_size = _triple(volume_size)
        self.patch_size = _triple(patch_size)
        self.patches_resolution = tuple(v // p for v, p in zip(self.volume_size, self.patch_size))
        self.num_patches = math.prod(self.patches_resolution)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"PatchEmbed3D expects B,C,D,H,W, got {tuple(x.shape)}")
        x = self.proj(x)
        return x.permute(0, 2, 3, 4, 1).contiguous()

    def reset_parameters(self) -> None:
        fan_in = self.in_chans * math.prod(self.patch_size)
        bound = math.sqrt(1.0 / fan_in)
        nn.init.uniform_(self.proj.weight, -bound, bound)
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -bound, bound)


class PhysicalRopePositionEmbedding3D(nn.Module):
    """Axial 3-D RoPE whose coordinates are measured in millimetres.

    Official DINOv3 splits half of each head between its two image axes.  A
    64-dimensional head cannot be split equally over three axes, so its 32
    rotary pairs are allocated as 11 z, 11 y and 10 x pairs.  The returned
    layout remains compatible with DINOv3's ``rope_rotate_half`` operation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        patch_size: tuple[int, int, int] = (2, 16, 16),
        voxel_spacing_mm: tuple[float, float, float] = (1.0, 0.5, 0.5),
        min_period_mm: float = 6.0,
        max_period_mm: float = 1200.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2:
            raise ValueError("3-D RoPE requires an even attention head dimension")
        n_pairs = self.head_dim // 2
        base, rem = divmod(n_pairs, 3)
        self.axis_pairs = (base + (rem > 0), base + (rem > 1), base)
        self.patch_size = _triple(patch_size)
        self.voxel_spacing_mm = tuple(float(v) for v in voxel_spacing_mm)
        self.min_period_mm = float(min_period_mm)
        self.max_period_mm = float(max_period_mm)
        self.dtype = dtype
        periods = self._make_periods(device=None)
        self.register_buffer("periods", periods, persistent=True)

    def _make_periods(self, device: torch.device | None) -> Tensor:
        periods = []
        for n in self.axis_pairs:
            if n == 1:
                p = torch.tensor([self.min_period_mm], dtype=self.dtype, device=device)
            else:
                p = torch.logspace(
                    math.log10(self.min_period_mm),
                    math.log10(self.max_period_mm),
                    n,
                    dtype=self.dtype,
                    device=device,
                )
            periods.append(p)
        return torch.cat(periods)

    def _axis_coords(self, n: int, axis: int, device: torch.device) -> Tensor:
        step = self.patch_size[axis] * self.voxel_spacing_mm[axis]
        # Translation does not affect RoPE attention, but centering improves its
        # numeric range and makes differently sized physical crops comparable.
        return (torch.arange(n, device=device, dtype=self.dtype) + 0.5 - n / 2.0) * step

    def forward(self, *, D: int, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        coords = torch.stack(
            torch.meshgrid(
                self._axis_coords(D, 0, device),
                self._axis_coords(H, 1, device),
                self._axis_coords(W, 2, device),
                indexing="ij",
            ),
            dim=-1,
        ).flatten(0, 2)
        angles = []
        start = 0
        for axis, n_freq in enumerate(self.axis_pairs):
            p = self.periods[start : start + n_freq]
            angles.append(2.0 * math.pi * coords[:, axis, None] / p[None, :])
            start += n_freq
        half = torch.cat(angles, dim=-1)
        full = torch.cat((half, half), dim=-1)
        return torch.sin(full), torch.cos(full)

    def _init_weights(self) -> None:
        # FSDP2 constructs the 7B model on meta and then calls to_empty.
        # Recreate these deterministic physical constants after materializing.
        self.periods.copy_(self._make_periods(self.periods.device))


_FFN = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu64": partial(SwiGLUFFN, align_to=64),
}
_NORM = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}


def _init_weights(module: nn.Module, name: str = "") -> None:
    del name
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    elif isinstance(module, (nn.LayerNorm, RMSNorm, LayerScale)):
        module.reset_parameters()
    elif isinstance(module, PatchEmbed3D):
        module.reset_parameters()


class DinoVisionTransformer3D(nn.Module):
    """Native-volume counterpart of official DINOv3 DinoVisionTransformer."""

    def __init__(
        self,
        *,
        volume_size: tuple[int, int, int] = (64, 384, 384),
        patch_size: tuple[int, int, int] = (2, 16, 16),
        voxel_spacing_mm: tuple[float, float, float] = (1.0, 0.5, 0.5),
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = 1e-5,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        n_storage_tokens: int = 4,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = True,
        rope_min_period_mm: float = 6.0,
        rope_max_period_mm: float = 1200.0,
        rope_dtype: torch.dtype = torch.float32,
        device: Any | None = None,
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = _triple(patch_size)
        self.voxel_spacing_mm = tuple(float(v) for v in voxel_spacing_mm)
        norm_cls = _NORM[norm_layer]
        ffn_cls = _FFN[ffn_layer]
        self.patch_embed = PatchEmbed3D(volume_size, self.patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))
        self.n_storage_tokens = int(n_storage_tokens)
        if self.n_storage_tokens:
            self.storage_tokens = nn.Parameter(torch.empty(1, self.n_storage_tokens, embed_dim, device=device))
        self.rope_embed = PhysicalRopePositionEmbedding3D(
            embed_dim,
            num_heads,
            patch_size=self.patch_size,
            voxel_spacing_mm=self.voxel_spacing_mm,
            min_period_mm=rope_min_period_mm,
            max_period_mm=rope_max_period_mm,
            dtype=rope_dtype,
        )
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=drop_path_rate,
                    norm_layer=norm_cls,
                    act_layer=nn.GELU,
                    ffn_layer=ffn_cls,
                    init_values=layerscale_init,
                    mask_k_bias=mask_k_bias,
                    device=device,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_cls(embed_dim)
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.cls_norm = norm_cls(embed_dim) if untie_cls_and_patch_norms else None
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self.local_cls_norm = norm_cls(embed_dim) if untie_global_and_local_cls_norm else None
        self.head = nn.Identity()

    def init_weights(self) -> None:
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.mask_token)
        if self.n_storage_tokens:
            nn.init.normal_(self.storage_tokens, std=0.02)
        named_apply(_init_weights, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks: Tensor | None = None):
        x = self.patch_embed(x)
        B, D, H, W, _ = x.shape
        x = x.flatten(1, 3)
        if masks is not None:
            if masks.shape != x.shape[:2]:
                raise ValueError(f"Mask {tuple(masks.shape)} does not match patch grid {tuple(x.shape[:2])}")
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls = self.cls_token
        else:
            cls = self.cls_token + 0 * self.mask_token
        storage = (
            self.storage_tokens
            if self.n_storage_tokens
            else torch.empty(1, 0, self.embed_dim, device=x.device, dtype=x.dtype)
        )
        return torch.cat((cls.expand(B, -1, -1), storage.expand(B, -1, -1), x), dim=1), (D, H, W)

    def forward_features_list(self, x_list: list[Tensor], masks_list: list[Tensor | None]):
        xs, grids = [], []
        for image, masks in zip(x_list, masks_list):
            tokens, grid = self.prepare_tokens_with_masks(image, masks)
            xs.append(tokens)
            grids.append(grid)
        ropes = [
            tuple(component.to(device=x.device, dtype=x.dtype) for component in self.rope_embed(D=d, H=h, W=w))
            for x, (d, h, w) in zip(xs, grids)
        ]
        for block in self.blocks:
            xs = block(xs, ropes)
        outputs = []
        for idx, (x, masks) in enumerate(zip(xs, masks_list)):
            prefix = self.n_storage_tokens + 1
            if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                cls_reg = self.local_cls_norm(x[:, :prefix])
                patch = self.norm(x[:, prefix:])
            elif self.untie_cls_and_patch_norms:
                cls_reg = self.cls_norm(x[:, :prefix])
                patch = self.norm(x[:, prefix:])
            else:
                y = self.norm(x)
                cls_reg, patch = y[:, :prefix], y[:, prefix:]
            outputs.append(
                {
                    "x_norm_clstoken": cls_reg[:, 0],
                    "x_storage_tokens": cls_reg[:, 1:],
                    "x_norm_patchtokens": patch,
                    "x_prenorm": x,
                    "masks": masks,
                    "patch_grid": grids[idx],
                }
            )
        return outputs

    def forward_features(self, x: Tensor | list[Tensor], masks: Tensor | list[Tensor] | None = None):
        if isinstance(x, Tensor):
            return self.forward_features_list([x], [masks])[0]
        if masks is None:
            masks = [None] * len(x)
        return self.forward_features_list(x, masks)

    def forward(self, x, masks=None, is_training: bool = False):
        out = self.forward_features(x, masks)
        return out if is_training else self.head(out["x_norm_clstoken"])

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: int | list[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        tokens, grid = self.prepare_tokens_with_masks(x)
        take = set(range(len(self.blocks) - n, len(self.blocks))) if isinstance(n, int) else set(n)
        outs = []
        rope = tuple(
            component.to(device=tokens.device, dtype=tokens.dtype)
            for component in self.rope_embed(D=grid[0], H=grid[1], W=grid[2])
        )
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, rope)
            if i in take:
                outs.append(self.norm(tokens) if norm else tokens)
        prefix = self.n_storage_tokens + 1
        cls = [o[:, 0] for o in outs]
        patches = [o[:, prefix:] for o in outs]
        if reshape:
            patches = [p.reshape(p.shape[0], *grid, p.shape[-1]).permute(0, 4, 1, 2, 3) for p in patches]
        return tuple(zip(patches, cls)) if return_class_token else tuple(patches)


def vit_large_3d(**kwargs) -> DinoVisionTransformer3D:
    return DinoVisionTransformer3D(embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4, **kwargs)


def vit_hplus_3d(**kwargs) -> DinoVisionTransformer3D:
    # Exact official DINOv3 H+ transformer geometry (hub/backbones.py):
    # 1280 x 32 x 20 heads, SwiGLU ratio 6, QKV bias. Only patch/RoPE are 3-D.
    return DinoVisionTransformer3D(
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=6,
        ffn_layer="swiglu",
        qkv_bias=True,
        mask_k_bias=True,
        norm_layer="layernormbf16",
        **kwargs,
    )


def vit_7b_3d(**kwargs) -> DinoVisionTransformer3D:
    """Literal largest official DINOv3 ViT geometry, ported to 3-D MR."""
    return DinoVisionTransformer3D(
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        ffn_layer="swiglu64",
        qkv_bias=False,
        mask_k_bias=True,
        norm_layer="layernormbf16",
        untie_global_and_local_cls_norm=True,
        **kwargs,
    )
