"""Recipe fixes ported from the FORA CT DINOv3 pipeline (v13/v14, Sep 2026).

The CT pipeline started from the same DINOv3-7B recipe as this module and failed in ways we
diagnosed and fixed one by one; every piece below is what the working CT ViT-L run uses:

* cosine prototypes on both heads (``CosineLinear``): DINOv3's plain linear prototype layer
  (std 0.02) never sharpened at small learning rates, so the Sinkhorn targets stayed flat
  (iBOT target entropy 8.0/11.5 nats, top-1 0.006) for thousands of steps;
* position-binned Sinkhorn for the patch head (CAPI, arXiv 2502.08769): the patch targets are
  balanced within spatial bins of the crop instead of across all masked tokens at once;
* DINOv2 optimizer groups: layer-wise lr decay, patch-embed lr x0.2, no weight decay on biases,
  1-D parameters and tokens, and the prototype layers frozen for the first steps;
* cross-crop twin tokens: with the two global crops aligned on the patch lattice every token in
  their overlap has an exact twin, which the objective uses to stop the head-facing block from
  encoding the token's position inside the crop (measured in CT: last-block position leakage
  0.201 -> 0.094, crop-quadrant prototype fields gone).
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

TOKEN_NAMES = ("cls_token", "mask_token", "storage_tokens")


# ----------------------------------------------------------------------------- prototypes
class CosineLinear(nn.Linear):
    """Bias-free prototype layer computing cosine logits (weights L2-normalized per prototype).

    DINOv3's ``DINOHead`` already L2-normalizes the bottleneck features before ``last_layer``,
    so the logits are true cosine similarities. The parameter name stays ``last_layer.weight``,
    keeping checkpoints of the plain layer loadable."""

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, F.normalize(self.weight, dim=1, eps=1e-12))


def normalize_prototype_layer(head: nn.Module) -> None:
    layer = head.last_layer
    if not isinstance(layer, nn.Linear) or layer.bias is not None:
        raise ValueError("Expected a bias-free linear prototype layer")
    if isinstance(layer, CosineLinear):
        return
    cosine = CosineLinear(layer.in_features, layer.out_features, bias=False,
                          device=layer.weight.device, dtype=layer.weight.dtype)
    with torch.no_grad():
        cosine.weight.copy_(layer.weight)
    head.last_layer = cosine


# ----------------------------------------------------------------------------- Sinkhorn bins
def position_bin_ids(token_indices: Tensor, n_tokens: int, grid, bins) -> Tensor:
    """Spatial bin id in [0, prod(bins)) of flat token indices (modulo the crop's token count)."""
    gz, gy, gx = (int(v) for v in grid)
    if gz * gy * gx != int(n_tokens):
        raise ValueError(f"position grid {grid} does not match {n_tokens} tokens per crop")
    bz, by, bx = (int(v) for v in bins)
    if not (1 <= bz <= gz and 1 <= by <= gy and 1 <= bx <= gx):
        raise ValueError(f"position bins {bins} must fit the token grid {grid}")
    pos = token_indices % int(n_tokens)
    z = pos // (gy * gx)
    y = (pos // gx) % gy
    x = pos % gx
    return ((z * bz) // gz) * (by * bx) + ((y * by) // gy) * bx + (x * bx) // gx


def binned_sinkhorn(base, logits: Tensor, temperature: float, bin_ids: Tensor, n_bins: int,
                    iterations: int = 3) -> Tensor:
    """Run the distributed Sinkhorn ``base`` once per spatial bin; every rank visits every
    non-empty bin in the same order (bin counts are all-reduced), so collectives stay matched."""
    if bin_ids.shape[0] != logits.shape[0]:
        raise ValueError("bin ids must align with the logit rows")
    out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    counts = torch.bincount(bin_ids, minlength=n_bins).to(torch.int64).to(logits.device)
    if dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    for b in range(n_bins):
        if int(counts[b]) == 0:
            continue
        rows = (bin_ids == b).nonzero().flatten()
        q = base(logits.index_select(0, rows), temperature, iterations)
        out.index_copy_(0, rows, q)
    return out


# ----------------------------------------------------------------------------- optimizer
def _block_index(name: str):
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "blocks" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def build_param_groups(student: nn.Module, *, layerwise_decay: float, patch_embed_lr_mult: float,
                       n_blocks: int) -> list[dict]:
    """DINOv2 ``get_params_groups_with_decay`` for the 3-D student (backbone + two heads).

    lr_mult: backbone blocks[i] get decay^(n_blocks - i); patch embedding, tokens and RoPE get
    decay^(n_blocks + 1) (times ``patch_embed_lr_mult`` for the patch embedding); final norms
    get 1; heads get 1. wd_mult: 0 for biases, 1-D parameters and tokens. Parameters that share
    (lr_mult, wd_mult) are merged into one group."""
    if not 0.0 < layerwise_decay <= 1.0 or patch_embed_lr_mult <= 0 or n_blocks < 1:
        raise ValueError("Require 0 < layerwise_decay <= 1, patch_embed_lr_mult > 0, n_blocks >= 1")
    merged: dict[tuple[float, float], dict] = {}
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        clean = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        lr_mult, wd_mult = 1.0, 1.0
        if clean.startswith("backbone."):
            block = _block_index(clean)
            if block is not None:
                layer_id = block + 1
            elif "patch_embed" in clean or any(t in clean for t in TOKEN_NAMES) or "rope" in clean:
                layer_id = 0
            else:
                layer_id = n_blocks + 1
            lr_mult = float(layerwise_decay) ** (n_blocks + 1 - layer_id)
            if "patch_embed" in clean:
                lr_mult *= float(patch_embed_lr_mult)
        if param.ndim <= 1 or clean.endswith(".bias") or any(t in clean for t in TOKEN_NAMES):
            wd_mult = 0.0
        key = (round(lr_mult, 12), wd_mult)
        group = merged.setdefault(key, dict(params=[], lr_mult=float(lr_mult), wd_mult=float(wd_mult), names=[]))
        group["params"].append(param)
        group["names"].append(clean)
    groups = [merged[k] for k in sorted(merged, key=lambda k: (-k[0], -k[1]))]
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps(dict(event="param_groups", groups=[
            dict(lr_mult=g["lr_mult"], wd_mult=g["wd_mult"], n=len(g["params"]), example=g["names"][0])
            for g in groups])), flush=True)
    return groups


class GroupedAdamW(torch.optim.AdamW):
    """AdamW applying each group's ``lr_mult``/``wd_mult`` transiently inside ``step`` so the
    trainer keeps writing the plain scheduled lr / weight decay into every group."""

    @torch.no_grad()
    def step(self, closure=None):
        saved = [(g["lr"], g["weight_decay"]) for g in self.param_groups]
        for g in self.param_groups:
            g["lr"] = g["lr"] * float(g.get("lr_mult", 1.0))
            g["weight_decay"] = g["weight_decay"] * float(g.get("wd_mult", 1.0))
        try:
            return super().step(closure)
        finally:
            for g, (lr, wd) in zip(self.param_groups, saved):
                g["lr"], g["weight_decay"] = lr, wd


def make_optimizer(student: nn.Module, *, lr: float, weight_decay: float, betas, layerwise_decay: float,
                   patch_embed_lr_mult: float, n_blocks: int, fused: bool = True) -> GroupedAdamW:
    groups = build_param_groups(student, layerwise_decay=layerwise_decay,
                                patch_embed_lr_mult=patch_embed_lr_mult, n_blocks=n_blocks)
    for g in groups:
        g.pop("names")
    return GroupedAdamW(groups, lr=lr, betas=tuple(betas), weight_decay=weight_decay, fused=fused)


def cancel_last_layer_gradients(student: nn.Module) -> int:
    """DINO/DINOv2 ``freeze_last_layer``: drop the prototype-layer gradients of both heads."""
    count = 0
    for name, param in student.named_parameters():
        if "last_layer" in name and param.grad is not None:
            param.grad = None
            count += 1
    return count


# ----------------------------------------------------------------------------- twin tokens
def snap_second_start(origin, second, container_shape, crop_shape, patch):
    """Move ``second`` so that (second - origin) is a multiple of the patch on every axis,
    staying inside the container."""
    out = []
    for o, s, have, want, p in zip(origin, second, container_shape, crop_shape, patch):
        k = int(round((s - o) / p))
        lo = -(o // p)
        hi = (max(have - want, 0) - o) // p
        if hi < lo:
            hi = lo
        out.append(o + min(max(k, lo), hi) * p)
    return tuple(out)


def token_offset(g0, g1, patch):
    d = []
    for a, b, p in zip(g0, g1, patch):
        if (b - a) % p:
            raise ValueError("global crops are not aligned on the patch lattice")
        d.append((b - a) // p)
    return tuple(d)


def overlap_pairs(offset, grid) -> tuple[Tensor, Tensor]:
    """Flat token indices (crop A, crop B) of every token covering the same voxels in both crops."""
    ranges = []
    for d, g in zip(offset, grid):
        lo, hi = max(0, d), min(g, g + d)
        if hi <= lo:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        ranges.append(torch.arange(lo, hi))
    z, y, x = torch.meshgrid(*ranges, indexing="ij")
    gz, gy, gx = grid
    a = (z * gy + y) * gx + x
    b = ((z - offset[0]) * gy + (y - offset[1])) * gx + (x - offset[2])
    return a.flatten(), b.flatten()


def padded_head(head: nn.Module, tokens: Tensor, chunk: int = 4096) -> tuple[Tensor, Tensor]:
    """Run an FSDP-sharded head on a per-rank-variable token count with rank-identical call shapes.
    Returns (logits[:local], anchor); anchor keeps a zero graph edge on ranks without tokens."""
    local = int(tokens.shape[0])
    maximum = torch.tensor(local, device=tokens.device)
    if dist.is_initialized():
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    maximum = int(maximum.item())
    parts, out = [], None
    anchor = tokens.sum() * 0.0
    for start in range(0, maximum, chunk):
        stop = min(start + chunk, maximum)
        size = stop - start
        valid = max(0, min(stop, local) - start)
        x = tokens[start:start + valid]
        if valid < size:
            x = F.pad(x, (0, 0, 0, size - valid))
        out = head(x)
        if valid == 0:
            anchor = anchor + out.sum() * 0.0
            continue
        parts.append(out[:valid])
    if parts:
        logits = torch.cat(parts)
    elif out is not None:
        logits = out[:0]
    else:
        logits = tokens.new_zeros((0, head.last_layer.weight.shape[0]))
    return logits, anchor


def entropy(q: Tensor) -> Tensor:
    return -(q * q.clamp_min(1e-12).log()).sum(-1)


def target_stats(q: Tensor) -> dict[str, Tensor]:
    """Teacher-target diagnostics: entropy, mean top-1 probability, fraction of prototypes that win."""
    if q.shape[0] == 0:
        z = q.new_zeros(())
        return dict(entropy=z, top1=z, usage=z)
    return dict(entropy=entropy(q).mean(), top1=q.max(-1).values.mean(),
                usage=torch.tensor(q.argmax(-1).unique().numel() / q.shape[1], device=q.device))


def sqrt_scaled_lr(base_lr: float, global_batch: int, reference: int = 1024) -> float:
    return base_lr * math.sqrt(global_batch / reference)
