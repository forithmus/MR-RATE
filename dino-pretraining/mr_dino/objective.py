"""DINOv3 self-distillation for aligned multi-sequence MR volume views."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.layers.dino_head import DINOHead
from .model import DinoVisionTransformer3D


class SSLNetwork3D(nn.Module):
    def __init__(
        self,
        backbone: DinoVisionTransformer3D,
        dino_prototypes: int,
        ibot_prototypes: int,
        dino_hidden_dim: int = 4096,
        ibot_hidden_dim: int = 4096,
        dino_bottleneck_dim: int = 384,
        ibot_bottleneck_dim: int = 384,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dino_head = DINOHead(
            in_dim=backbone.embed_dim,
            out_dim=dino_prototypes,
            hidden_dim=dino_hidden_dim,
            bottleneck_dim=dino_bottleneck_dim,
            nlayers=3,
        )
        self.ibot_head = DINOHead(
            in_dim=backbone.embed_dim,
            out_dim=ibot_prototypes,
            hidden_dim=ibot_hidden_dim,
            bottleneck_dim=ibot_bottleneck_dim,
            nlayers=3,
        )

    def init_weights(self) -> None:
        self.backbone.init_weights()
        self.dino_head.init_weights()
        self.ibot_head.init_weights()


@torch.no_grad()
def distributed_sinkhorn(logits: Tensor, temperature: float, iterations: int = 3) -> Tensor:
    """Official DINOv3 Sinkhorn-Knopp assignment with WORLD as the group."""
    # A global additive constant cancels during Sinkhorn normalization. Remove
    # it before exponentiation so a larger masked-token batch cannot overflow
    # merely because it is more likely to contain an extreme teacher logit.
    max_logit = logits.detach().float().max()
    if dist.is_initialized():
        dist.all_reduce(max_logit, op=dist.ReduceOp.MAX)
    q = torch.exp((logits.float() - max_logit) / temperature).t()
    world = dist.get_world_size() if dist.is_initialized() else 1
    global_batch = q.shape[1] * world
    prototypes = q.shape[0]
    total = q.sum()
    if dist.is_initialized():
        dist.all_reduce(total)
    q /= total.clamp_min(1e-12)
    for _ in range(iterations):
        rows = q.sum(dim=1, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(rows)
        q /= rows.clamp_min(1e-12)
        q /= prototypes
        q /= q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q /= global_batch
    return (q * global_batch).t()


def dino_cross_entropy(
    student: Tensor,
    teacher: Tensor,
    student_temp: float,
    ignore_diagonal: bool,
) -> Tensor:
    """Return one DINO loss per sample for [views, batch, prototypes]."""
    logp = F.log_softmax(student.float() / student_temp, dim=-1)
    losses = -torch.einsum("sbk,tbk->stb", logp, teacher)
    if ignore_diagonal:
        n = min(student.shape[0], teacher.shape[0])
        view_mask = torch.ones(losses.shape[:2], dtype=torch.bool, device=losses.device)
        indices = torch.arange(n, device=losses.device)
        view_mask[indices, indices] = False
        return losses[view_mask].reshape(-1, student.shape[1]).mean(0)
    else:
        return losses.mean((0, 1))


def sampled_gram_loss(student: Tensor, target: Tensor, max_tokens: int, seed: int) -> Tensor:
    """DINOv3 Gram anchoring with bounded, shared token sampling.

    A full 18,432^2 Gram matrix is needlessly quadratic.  The same deterministic
    physical-token subset is used on student and frozen anchor, preserving the
    correlation objective while bounding high-resolution memory.
    """
    n = student.shape[1]
    if n > max_tokens:
        g = torch.Generator(device=student.device).manual_seed(seed)
        idx = torch.randperm(n, generator=g, device=student.device)[:max_tokens]
        student, target = student[:, idx], target[:, idx]
    student = F.normalize(student.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    s_gram = student @ student.transpose(-1, -2)
    t_gram = target @ target.transpose(-1, -2)
    return F.mse_loss(s_gram, t_gram, reduction="none").mean((-1, -2))


def distributed_koleo(x: Tensor, eps: float = 1e-8) -> Tensor:
    """KoLeo with detached global nearest-neighbour lookup.

    The gather occurs entirely in the forward, avoiding an autograd collective
    interleaved with DDP's gradient reductions (a known failure mode on this
    128-rank cluster). Gradients still flow through each local representation.
    """
    x = F.normalize(x.float(), dim=-1)
    if dist.is_initialized():
        gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x.detach())
        all_x = torch.cat(gathered)
        offset = dist.get_rank() * x.shape[0]
    else:
        all_x, offset = x.detach(), 0
    scores = x.detach() @ all_x.t()
    rows = torch.arange(x.shape[0], device=x.device)
    scores[rows, rows + offset] = -torch.inf
    if all_x.shape[0] == 1:
        return x.new_zeros((x.shape[0],))
    nearest = all_x[scores.argmax(dim=-1)]
    return -torch.log(torch.linalg.vector_norm(x - nearest, dim=-1).clamp_min(eps))


@dataclass
class LossWeights:
    dino: float = 1.0
    ibot: float = 1.0
    koleo: float = 0.1
    gram: float = 0.0


class DINO3DLearner(nn.Module):
    """Student, EMA teacher, optional frozen Gram anchor, and all SSL losses."""

    def __init__(
        self,
        backbone: DinoVisionTransformer3D,
        prototypes: int = 65_536,
        head_hidden_dim: int = 4096,
        dino_prototypes: int | None = None,
        ibot_prototypes: int | None = None,
        dino_head_hidden_dim: int | None = None,
        ibot_head_hidden_dim: int | None = None,
        dino_bottleneck_dim: int = 384,
        ibot_bottleneck_dim: int = 384,
        student_temperature: float = 0.1,
        loss_weights: LossWeights = LossWeights(),
        gram_max_tokens: int = 1024,
        ibot_loss_chunk_size: int = 1024,
        with_gram_anchor: bool = False,
        initialize_weights: bool = True,
    ) -> None:
        super().__init__()
        self.student = SSLNetwork3D(
            backbone,
            dino_prototypes or prototypes,
            ibot_prototypes or prototypes,
            dino_head_hidden_dim or head_hidden_dim,
            ibot_head_hidden_dim or head_hidden_dim,
            dino_bottleneck_dim,
            ibot_bottleneck_dim,
        )
        if initialize_weights:
            self.student.init_weights()
        self.teacher = copy.deepcopy(self.student).requires_grad_(False)
        self.teacher.eval()
        self.gram_anchor = copy.deepcopy(self.teacher).requires_grad_(False) if with_gram_anchor else None
        self.student_temperature = float(student_temperature)
        self.weights = loss_weights
        self.gram_max_tokens = int(gram_max_tokens)
        self.ibot_loss_chunk_size = int(ibot_loss_chunk_size)
        if self.ibot_loss_chunk_size < 1:
            raise ValueError("ibot_loss_chunk_size must be positive")

    @torch.no_grad()
    def initialize_student_teacher(self) -> None:
        """Initialize a materialized FSDP2 student, then clone its local shards."""
        self.student.init_weights()
        for target, source in zip(self.teacher.parameters(), self.student.parameters()):
            target.copy_(source)
        for target, source in zip(self.teacher.buffers(), self.student.buffers()):
            target.copy_(source)
        if self.gram_anchor is not None:
            for target, source in zip(self.gram_anchor.parameters(), self.teacher.parameters()):
                target.copy_(source)
            for target, source in zip(self.gram_anchor.buffers(), self.teacher.buffers()):
                target.copy_(source)

    @property
    def student_module(self) -> SSLNetwork3D:
        return self.student.module if hasattr(self.student, "module") else self.student

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        if self.gram_anchor is not None:
            self.gram_anchor.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        student = self.student_module
        for target, source in zip(self.teacher.parameters(), student.parameters()):
            target.lerp_(source.detach(), 1.0 - momentum)
        for target, source in zip(self.teacher.buffers(), student.buffers()):
            target.copy_(source.detach())

    @torch.no_grad()
    def reset_gram_anchor_from_teacher(self) -> None:
        if self.gram_anchor is None:
            self.gram_anchor = copy.deepcopy(self.teacher).requires_grad_(False)
        else:
            self.gram_anchor.load_state_dict(self.teacher.state_dict())
        self.gram_anchor.eval()

    def forward(self, batch: dict, teacher_temperature: float, step: int) -> tuple[Tensor, dict[str, Tensor]]:
        teacher_images = batch["teacher_global"]  # [2,B,1,D,H,W]
        student_images = batch["student_global"]
        local_images = batch["student_local"]
        masks = batch["masks"]
        mask_indices = batch["mask_indices"]
        mask_weights = batch["mask_weights"]
        n_global, batch_size = teacher_images.shape[:2]
        n_local = local_images.shape[0]
        sample_weights = batch.get("loss_weights")
        if sample_weights is None:
            sample_weights = torch.ones(batch_size, device=teacher_images.device)
        sample_weights = sample_weights.float()
        if sample_weights.shape != (batch_size,):
            raise ValueError(
                f"loss_weights must have shape ({batch_size},), got {tuple(sample_weights.shape)}"
            )

        def weighted_mean(per_sample: Tensor) -> Tensor:
            if per_sample.shape != (batch_size,):
                raise ValueError(
                    f"Expected one loss per sample ({batch_size},), got {tuple(per_sample.shape)}"
                )
            # Keep explicit sample weighting outside the minibatch denominator.
            return (per_sample * sample_weights).mean()

        tg = teacher_images.flatten(0, 1)
        sg = student_images.flatten(0, 1)
        sl = local_images.flatten(0, 1)

        with torch.no_grad():
            tout = self.teacher.backbone(tg, is_training=True)
            t_cls_logits = self.teacher.dino_head(tout["x_norm_clstoken"])
            t_patch = tout["x_norm_patchtokens"]
            t_masked = t_patch.flatten(0, 1).index_select(0, mask_indices)
            t_patch_logits = self.teacher.ibot_head(t_masked)
            t_cls_prob = distributed_sinkhorn(t_cls_logits, teacher_temperature).unflatten(0, (n_global, batch_size))
            t_patch_prob = distributed_sinkhorn(t_patch_logits, teacher_temperature)
            del t_patch_logits, t_masked

        global_out, local_out = self.student.backbone(
            [sg, sl], masks=[masks, None], is_training=True
        )
        s_g_cls = self.student.dino_head(global_out["x_norm_clstoken"]).unflatten(0, (n_global, batch_size))
        s_l_cls = self.student.dino_head(local_out["x_norm_clstoken"]).unflatten(0, (n_local, batch_size))
        s_masked = global_out["x_norm_patchtokens"].flatten(0, 1).index_select(0, mask_indices)

        dino_global = dino_cross_entropy(s_g_cls, t_cls_prob, self.student_temperature, True)
        dino_local = dino_cross_entropy(s_l_cls, t_cls_prob, self.student_temperature, False)
        # Match official relative weighting: 2 global off-diagonal terms vs
        # 2*n_local local-to-global terms.
        global_scale = 2.0 / (2.0 + 2.0 * n_local)
        local_scale = 1.0 - global_scale
        dino_each = global_scale * dino_global + local_scale * dino_local
        dino = weighted_mean(dino_each)

        # 98,304 prototypes make a full FP32 log-softmax the dominant transient
        # allocation at useful local batch sizes. Chunk only the token axis;
        # every token still sees every prototype and the mathematical loss is
        # unchanged.
        # ``ibot_head`` is FSDP-sharded.  Every rank must therefore execute the
        # same number of head forwards in the same order.  Mask sampling gives
        # each rank a different number of masked tokens; independently looping
        # to the local length desynchronizes FSDP all-gathers and deadlocks.
        # Use the distributed maximum and pad each call to an identical shape.
        local_masked = int(s_masked.shape[0])
        maximum_masked = torch.tensor(local_masked, device=s_masked.device)
        if dist.is_initialized():
            dist.all_reduce(maximum_masked, op=dist.ReduceOp.MAX)
        maximum_masked = int(maximum_masked.item())
        ibot_parts = []
        ibot_collective_anchor = s_masked.sum() * 0.0
        for start in range(0, maximum_masked, self.ibot_loss_chunk_size):
            common_stop = min(start + self.ibot_loss_chunk_size, maximum_masked)
            common_size = common_stop - start
            local_stop = min(common_stop, local_masked)
            valid = max(0, local_stop - start)
            tokens = s_masked[start:local_stop]
            if valid < common_size:
                tokens = F.pad(tokens, (0, 0, 0, common_size - valid))
            padded_logits = self.student.ibot_head(tokens)
            logits = padded_logits[:valid]
            if valid == 0:
                # The padded forward above is still required to keep FSDP
                # collective ordering identical.  Preserve a zero-valued graph
                # edge so its FSDP backward hook also runs on this rank.
                ibot_collective_anchor = (
                    ibot_collective_anchor + padded_logits.sum() * 0.0
                )
                continue
            scaled_logits = logits.float() / self.student_temperature
            targets = t_patch_prob[start:local_stop]
            # Equivalent to -sum(q * log_softmax(z)) without materializing the
            # full log-probability or elementwise-product tensors.
            target_mass = targets.sum(-1)
            expected_logit = torch.bmm(
                targets.unsqueeze(1), scaled_logits.unsqueeze(2)
            ).flatten()
            ibot_parts.append(
                torch.logsumexp(scaled_logits, dim=-1) * target_mass
                - expected_logit
            )
        ibot_each = torch.cat(ibot_parts)
        mask_counts = masks.sum(-1)
        mask_ids = torch.arange(masks.shape[0], device=masks.device).repeat_interleave(mask_counts)
        ibot_per_mask = torch.zeros(masks.shape[0], device=ibot_each.device, dtype=ibot_each.dtype)
        ibot_per_mask.scatter_add_(0, mask_ids, ibot_each * mask_weights)
        ibot_each_sample = ibot_per_mask.unflatten(0, (n_global, batch_size)).mean(0)
        ibot = weighted_mean(ibot_each_sample) + ibot_collective_anchor

        koleo_each = torch.stack(
            [
                distributed_koleo(x)
                for x in global_out["x_norm_clstoken"].unflatten(0, (n_global, batch_size))
            ]
        ).mean(0)
        koleo = weighted_mean(koleo_each)

        gram = dino.new_zeros(())
        if self.gram_anchor is not None and self.weights.gram > 0:
            with torch.no_grad():
                anchor = self.gram_anchor.backbone(tg, is_training=True)["x_norm_patchtokens"]
            gram_each = sampled_gram_loss(
                global_out["x_norm_patchtokens"], anchor, self.gram_max_tokens, seed=step
            ).unflatten(0, (n_global, batch_size)).mean(0)
            gram = weighted_mean(gram_each)

        total = (
            self.weights.dino * dino
            + self.weights.ibot * ibot
            + self.weights.koleo * koleo
            + self.weights.gram * gram
        )
        return total, {
            "loss": total.detach(),
            "dino": dino.detach(),
            "dino_global": weighted_mean(dino_global).detach(),
            "dino_local": weighted_mean(dino_local).detach(),
            "ibot": ibot.detach(),
            "koleo": koleo.detach(),
            "gram": gram.detach(),
        }
