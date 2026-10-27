"""MR-RATE scalar MIL, with the MRI checkpoint's own final block and physical RoPE.

Every sequence/crop contributes to one study-level class softmax. Two replayed
passes give exact gradients without retaining every crop's transformer graph.
"""
from __future__ import annotations

import copy
import torch
from torch import nn


class LastBlockAdapter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.block = copy.deepcopy(backbone.blocks[-1]).float()
        self.norm = copy.deepcopy(backbone.norm).float()
        self.block.requires_grad_(True)
        self.norm.requires_grad_(True)
        self.rope = copy.deepcopy(backbone.rope_embed).float()
        self.prefix = 1 + backbone.n_storage_tokens
        self.dim = backbone.embed_dim
        self.rope.requires_grad_(False)

    def forward(self, tokens, grid):
        # Disable pretraining DropPath for deterministic replay. Parameters train.
        self.block.eval()
        rope = tuple(t.to(device=tokens.device, dtype=tokens.dtype)
                     for t in self.rope(D=grid[0], H=grid[1], W=grid[2]))
        return self.norm(self.block(tokens, rope))[:, self.prefix:]


def _scores(head, scores, mask):
    scores = scores.float()
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != scores.shape:
            raise ValueError("Route mask must be boolean [tokens, classes]")
        penalty = torch.nn.functional.softplus(head.route_bias_raw.float())
        scores = scores - (~mask).float() * penalty
    return scores


@torch.no_grad()
def statistics(head, factory):
    device = next(head.parameters()).device
    classes = torch.arange(head.n_classes, device=device)
    maximum = torch.full((head.n_classes,), -torch.inf, device=device)
    den = torch.zeros_like(maximum)
    num = torch.zeros_like(maximum)
    count = 0
    for tokens, mask in factory():
        scores, logits = head._chunk_outputs(tokens.float(), classes)
        scores = _scores(head, scores, mask)
        new = torch.maximum(maximum, scores.amax(0))
        old = (maximum-new).exp()
        weights = (scores-new).exp()
        den = den*old + weights.sum(0)
        num = num*old + (weights*logits.float()).sum(0)
        maximum = new
        count += len(tokens)
    if count == 0 or not bool((den > 0).all()) or not bool(torch.isfinite(num).all()):
        raise ValueError("Empty or nonfinite MIL bag")
    return maximum, den, num/den, count


@torch.no_grad()
def predict(head, factory):
    _, _, pooled, _ = statistics(head, factory)
    return (pooled*head.output_scale + head.output_bias)[None]


def backward(head, factory, target, loss_fn=None):
    """Replay deterministic features; gradients also flow into the final block."""
    if head.input_dropout.p:
        raise ValueError("Two-pass replay requires dropout=0")
    versions = tuple(p._version for p in head.parameters())
    maximum, den, pooled, count = statistics(head, factory)
    leaf = pooled.detach().requires_grad_(True)
    logits = (leaf*head.output_scale+head.output_bias)[None]
    objective = loss_fn or torch.nn.functional.binary_cross_entropy_with_logits
    loss = objective(logits, target.reshape(1,-1))
    loss.backward()
    gradient = leaf.grad.detach()
    classes = torch.arange(head.n_classes, device=pooled.device)
    seen = 0
    for tokens, mask in factory():
        scores, patch = head._chunk_outputs(tokens.float(), classes)
        scores = _scores(head, scores, mask)
        with torch.no_grad():
            attention = (scores.detach()-maximum).exp()/den
            dl = attention*gradient
            ds = dl*(patch.float()-pooled)
        torch.autograd.backward((scores,patch), (ds,dl))
        seen += len(tokens)
    if seen != count or versions != tuple(p._version for p in head.parameters()):
        raise RuntimeError("Token count or head parameters changed during replay")
    return loss.detach(), logits.detach()


@torch.no_grad()
def evidence(head, factory):
    """Yield exact positive token logit contributions; bias is excluded."""
    maximum, den, _, _ = statistics(head, factory)
    classes = torch.arange(head.n_classes, device=maximum.device)
    for tokens, mask in factory():
        scores, patch = head._chunk_outputs(tokens.float(),classes)
        attention = (_scores(head,scores,mask)-maximum).exp()/den
        yield (attention*patch.float()*head.output_scale).clamp_min(0)
