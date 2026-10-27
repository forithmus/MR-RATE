"""Class-specific gated-attention MIL for multi-sequence MRI patch tokens.

A study contains every eligible sequence. Tokens are classified individually,
then pooled with a separate attention softmax for each diagnosis. Ragged bag
boundaries keep studies independent. Sparse aggregation streams token chunks.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    "PatchClassifier",
    "ClassifyThenAggregate",
    "SparseClassifyThenAggregate",
    "MRRATEMILHead",
    "collate_ragged_bags",
]


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


class PatchClassifier(nn.Module):
    """Shared instance classifier: Linear/GELU blocks followed by Linear."""

    def __init__(
        self,
        dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
    ) -> None:
        super().__init__()
        dim = _positive_int(dim, "dim")
        out_dim = _positive_int(out_dim, "out_dim")
        hidden_dims = tuple(
            _positive_int(hidden, f"hidden_dims[{index}]")
            for index, hidden in enumerate(hidden_dims)
        )

        layers = []
        current_dim = dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.GELU()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        return self.net(instances)


class ClassifyThenAggregate(nn.Module):
    """class-specific class-specific gated-attention MIL head.

    Parameters
    ----------
    dim:
        Width of each visual-instance embedding.
    n_classes:
        Number of independent study-level labels.
    hidden_dim:
        Width of each gated-attention projection.
    mlp_hidden_dims:
        Hidden widths in the shared instance classifier. An empty sequence
        makes the instance classifier a single linear layer.
    drop_rate:
        Dropout applied independently before the attention and classifier
        branches.
    use_gating:
        If true, attention is ``tanh(Vx) * sigmoid(Ux)``.  If false it is
        ``tanh(Vx)``.
    use_norm:
        Apply a separate LayerNorm before each branch.
    use_output_bias_scale:
        Learn a per-class affine transform of the aggregated logits.

    Notes
    -----
    ``forward(..., return_details=True)`` returns
    ``(bag_logits, class_attention, patch_logits)``.  Both detail tensors are
    flat ``[N, C]`` tensors aligned with the input instance order. For every bag
    and class, ``class_attention`` sums to one.
    """

    def __init__(
        self,
        dim: int,
        n_classes: int,
        hidden_dim: int = 512,
        mlp_hidden_dims: Sequence[int] = (384,),
        drop_rate: float = 0.0,
        init_std: float = 0.02,
        use_gating: bool = True,
        use_norm: bool = False,
        use_output_bias_scale: bool = True,
    ) -> None:
        super().__init__()
        self.dim = _positive_int(dim, "dim")
        self.n_classes = _positive_int(n_classes, "n_classes")
        hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
        if init_std <= 0.0:
            raise ValueError(f"init_std must be positive, got {init_std}")

        self.use_gating = bool(use_gating)
        self.use_output_bias_scale = bool(use_output_bias_scale)
        self.init_std = float(init_std)

        def make_norm() -> nn.Module:
            if use_norm:
                return nn.LayerNorm(self.dim, eps=1e-6)
            return nn.Identity()

        self.attention_norm = make_norm()
        self.classifier_norm = make_norm()
        self.input_dropout = nn.Dropout(drop_rate)

        self.attention_v = nn.Linear(self.dim, hidden_dim)
        self.attention_u = nn.Linear(self.dim, hidden_dim) if self.use_gating else None
        self.attention_w = nn.Linear(hidden_dim, self.n_classes)
        self.patch_classifier = PatchClassifier(
            self.dim,
            mlp_hidden_dims,
            self.n_classes,
        )

        if self.use_output_bias_scale:
            self.output_scale = nn.Parameter(torch.ones(self.n_classes))
            self.output_bias = nn.Parameter(torch.zeros(self.n_classes))
        else:
            self.register_parameter("output_scale", None)
            self.register_parameter("output_bias", None)

        self.apply(self._init_weights)

    @property
    def input_dim(self) -> int:
        """Alias useful when inspecting generic visual-instance heads."""

        return self.dim

    @property
    def num_classes(self) -> int:
        """Alias useful when inspecting generic visual-instance heads."""

        return self.n_classes

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_inputs(
        self,
        instances: torch.Tensor,
        bag_ptr: torch.Tensor,
    ) -> Tuple[int, ...]:
        if not isinstance(instances, torch.Tensor):
            raise TypeError("instances must be a torch.Tensor")
        if instances.ndim != 2 or instances.shape[1] != self.dim:
            raise ValueError(
                f"instances must have shape [N, {self.dim}], "
                f"got {tuple(instances.shape)}"
            )
        if not instances.is_floating_point():
            raise TypeError("instances must have a floating-point dtype")

        if not isinstance(bag_ptr, torch.Tensor):
            raise TypeError("bag_ptr must be a torch.Tensor")
        if bag_ptr.ndim != 1:
            raise ValueError(
                f"bag_ptr must be one-dimensional, got {tuple(bag_ptr.shape)}"
            )
        if bag_ptr.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("bag_ptr must have an integer dtype")

        bounds = tuple(int(bound) for bound in bag_ptr.detach().cpu().tolist())
        if len(bounds) < 2:
            raise ValueError("bag_ptr must contain at least a start and end boundary")
        if bounds[0] != 0:
            raise ValueError("bag_ptr must start at 0")
        if bounds[-1] != instances.shape[0]:
            raise ValueError(
                "bag_ptr must end at the number of flat instances "
                f"({instances.shape[0]}), got {bounds[-1]}"
            )
        if any(end <= start for start, end in zip(bounds[:-1], bounds[1:])):
            raise ValueError(
                "bag_ptr boundaries must be strictly increasing; "
                "MIL bags must contain at least one instance"
            )
        return bounds

    def forward(
        self,
        instances: torch.Tensor,
        bag_ptr: torch.Tensor,
        return_details: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Classify visual instances and aggregate each ragged study bag.

        Parameters
        ----------
        instances:
            Concatenated visual-instance embeddings with shape ``[N, dim]``.
        bag_ptr:
            Integer cumulative boundaries with shape ``[B + 1]``.  It must
            begin at zero, end at ``N``, and be strictly increasing.
        return_details:
            Also return flat class attention and instance-level logits.
        """

        bounds = self._validate_inputs(instances, bag_ptr)

        attention_input = self.input_dropout(self.attention_norm(instances))
        classifier_input = self.input_dropout(self.classifier_norm(instances))

        attention_hidden = torch.tanh(self.attention_v(attention_input))
        if self.attention_u is not None:
            attention_hidden = attention_hidden * torch.sigmoid(
                self.attention_u(attention_input)
            )
        attention_scores = self.attention_w(attention_hidden)
        patch_logits = self.patch_classifier(classifier_input)

        bag_logits = []
        attention_chunks = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            class_attention = F.softmax(attention_scores[start:end], dim=0)
            bag_logits.append((class_attention * patch_logits[start:end]).sum(dim=0))
            if return_details:
                attention_chunks.append(class_attention)

        output = torch.stack(bag_logits, dim=0)
        if self.output_scale is not None:
            output = output * self.output_scale + self.output_bias

        if return_details:
            return output, torch.cat(attention_chunks, dim=0), patch_logits
        return output


class SparseClassifyThenAggregate(nn.Module):
    """Class-selective token MIL with exact chunked softmax aggregation.

    Unmentioned pathology leaves are unknown, not negative.  ``class_indices``
    therefore selects only the explicitly supervised classes for the current
    report while parameters still span the complete vocabulary.  An online
    stable-softmax reduction avoids materializing ``all_tokens x classes``.
    """

    def __init__(
        self,
        dim: int,
        n_classes: int,
        hidden_dim: int = 512,
        mlp_hidden_dim: int = 384,
        drop_rate: float = 0.0,
        use_norm: bool = False,
        token_chunk_size: int = 32768,
        checkpoint_chunks: bool = True,
    ) -> None:
        super().__init__()
        self.dim = _positive_int(dim, "dim")
        self.n_classes = _positive_int(n_classes, "n_classes")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.mlp_hidden_dim = _positive_int(mlp_hidden_dim, "mlp_hidden_dim")
        self.token_chunk_size = _positive_int(token_chunk_size, "token_chunk_size")
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError("drop_rate must be in [0, 1)")
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self.attention_norm = (
            nn.LayerNorm(self.dim, eps=1e-6) if use_norm else nn.Identity()
        )
        self.classifier_norm = (
            nn.LayerNorm(self.dim, eps=1e-6) if use_norm else nn.Identity()
        )
        self.input_dropout = nn.Dropout(drop_rate)
        self.attention_v = nn.Linear(self.dim, self.hidden_dim)
        self.attention_u = nn.Linear(self.dim, self.hidden_dim)
        self.attention_w = nn.Linear(self.hidden_dim, self.n_classes)
        self.patch_hidden = nn.Linear(self.dim, self.mlp_hidden_dim)
        self.patch_out = nn.Linear(self.mlp_hidden_dim, self.n_classes)
        self.output_scale = nn.Parameter(torch.ones(self.n_classes))
        self.output_bias = nn.Parameter(torch.zeros(self.n_classes))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _chunk_outputs(
        self, tokens: torch.Tensor, class_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_scores = self._chunk_attention_scores(tokens, class_indices)
        classifier_input = self.input_dropout(self.classifier_norm(tokens))
        patch_hidden = F.gelu(self.patch_hidden(classifier_input))
        patch_logits = F.linear(
            patch_hidden,
            self.patch_out.weight.index_select(0, class_indices),
            self.patch_out.bias.index_select(0, class_indices),
        )
        return attention_scores, patch_logits

    def _chunk_attention_scores(
        self, tokens: torch.Tensor, class_indices: torch.Tensor
    ) -> torch.Tensor:
        """Class-attention scores without the unused patch-logit branch."""
        attention_input = self.input_dropout(self.attention_norm(tokens))
        attention_hidden = torch.tanh(self.attention_v(attention_input))
        attention_hidden = attention_hidden * torch.sigmoid(
            self.attention_u(attention_input)
        )
        attention_scores = F.linear(
            attention_hidden,
            self.attention_w.weight.index_select(0, class_indices),
            self.attention_w.bias.index_select(0, class_indices),
        )
        return attention_scores

    def forward(
        self, tokens: torch.Tensor, class_indices: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.dim or tokens.shape[0] == 0:
            raise ValueError(
                f"tokens must be nonempty [N,{self.dim}], got {tuple(tokens.shape)}"
            )
        if class_indices.ndim != 1 or class_indices.numel() == 0:
            raise ValueError("class_indices must be a nonempty vector")
        class_indices = class_indices.to(device=tokens.device, dtype=torch.long)
        if (
            int(class_indices.min()) < 0
            or int(class_indices.max()) >= self.n_classes
            or class_indices.unique().numel() != class_indices.numel()
        ):
            raise ValueError("class_indices are out of range or duplicated")

        from torch.utils.checkpoint import checkpoint

        running_max = torch.full(
            (class_indices.numel(),),
            -torch.inf,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        running_denominator = torch.zeros_like(running_max)
        running_numerator = torch.zeros_like(running_max)
        for start in range(0, tokens.shape[0], self.token_chunk_size):
            chunk = tokens[start : start + self.token_chunk_size]
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                scores, patch_logits = checkpoint(
                    self._chunk_outputs,
                    chunk,
                    class_indices,
                    use_reentrant=False,
                )
            else:
                scores, patch_logits = self._chunk_outputs(chunk, class_indices)
            chunk_max = scores.max(dim=0).values
            new_max = torch.maximum(running_max, chunk_max)
            old_scale = torch.exp(running_max - new_max)
            weights = torch.exp(scores - new_max.unsqueeze(0))
            running_numerator = (
                running_numerator * old_scale
                + (weights * patch_logits).sum(dim=0)
            )
            running_denominator = (
                running_denominator * old_scale + weights.sum(dim=0)
            )
            running_max = new_max
        logits = running_numerator / running_denominator.clamp_min(
            torch.finfo(running_denominator.dtype).tiny
        )
        logits = (
            logits * self.output_scale.index_select(0, class_indices)
            + self.output_bias.index_select(0, class_indices)
        )
        return logits.unsqueeze(0)

    def forward_with_evidence(
        self, tokens: torch.Tensor, class_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bag logits and exact class-attention-pooled token features.

        The evidence vector for class ``c`` is

        ``sum_t softmax(attention_score[t, c]) * tokens[t]``.

        It is accumulated in the same online-softmax pass as the class logit,
        so every token contributes without materializing an ``N x C x D``
        tensor.  This method is intended for frozen-MIL conditioning; the
        ordinary :meth:`forward` path remains unchanged for MIL training.
        """
        if tokens.ndim != 2 or tokens.shape[1] != self.dim or tokens.shape[0] == 0:
            raise ValueError(
                f"tokens must be nonempty [N,{self.dim}], got {tuple(tokens.shape)}"
            )
        if class_indices.ndim != 1 or class_indices.numel() == 0:
            raise ValueError("class_indices must be a nonempty vector")
        class_indices = class_indices.to(device=tokens.device, dtype=torch.long)
        if (
            int(class_indices.min()) < 0
            or int(class_indices.max()) >= self.n_classes
            or class_indices.unique().numel() != class_indices.numel()
        ):
            raise ValueError("class_indices are out of range or duplicated")

        running_max = torch.full(
            (class_indices.numel(),),
            -torch.inf,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        running_denominator = torch.zeros_like(running_max)
        running_logit_numerator = torch.zeros_like(running_max)
        # Evidence accumulation is small (C x D) and benefits from fp32 when
        # a study contains hundreds of thousands of uncapped image tokens.
        running_evidence_numerator = torch.zeros(
            (class_indices.numel(), self.dim),
            dtype=torch.float32,
            device=tokens.device,
        )
        for start in range(0, tokens.shape[0], self.token_chunk_size):
            chunk = tokens[start : start + self.token_chunk_size]
            scores, patch_logits = self._chunk_outputs(chunk, class_indices)
            chunk_max = scores.max(dim=0).values
            new_max = torch.maximum(running_max, chunk_max)
            old_scale = torch.exp(running_max - new_max)
            weights = torch.exp(scores - new_max.unsqueeze(0))
            running_logit_numerator = (
                running_logit_numerator * old_scale
                + (weights * patch_logits).sum(dim=0)
            )
            running_evidence_numerator = (
                running_evidence_numerator * old_scale.float().unsqueeze(1)
                + weights.float().transpose(0, 1) @ chunk.float()
            )
            running_denominator = (
                running_denominator * old_scale + weights.sum(dim=0)
            )
            running_max = new_max

        denominator = running_denominator.clamp_min(
            torch.finfo(running_denominator.dtype).tiny
        )
        logits = running_logit_numerator / denominator
        logits = (
            logits * self.output_scale.index_select(0, class_indices)
            + self.output_bias.index_select(0, class_indices)
        )
        evidence = (
            running_evidence_numerator / denominator.float().unsqueeze(1)
        ).to(tokens.dtype)
        return logits.unsqueeze(0), evidence.unsqueeze(0)


# MRI head alias.
MRRATEMILHead = ClassifyThenAggregate


def collate_ragged_bags(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate ``(visual_instances, target)`` samples without padding.

    Returns ``(flat_features, targets, bag_ptr)``.  Boundaries stay on CPU,
    matching PyTorch's usual collation metadata; the head accepts boundaries on
    either CPU or the feature device.
    """

    if len(batch) == 0:
        raise ValueError("cannot collate an empty batch")

    bags, targets = zip(*batch)
    first = bags[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 2:
        raise ValueError("each visual-instance bag must be a rank-2 tensor [M, D]")
    feature_dim = first.shape[1]

    for index, bag in enumerate(bags):
        if not isinstance(bag, torch.Tensor) or bag.ndim != 2:
            raise ValueError(f"bag {index} must be a rank-2 tensor [M, D]")
        if bag.shape[0] == 0:
            raise ValueError(f"bag {index} is empty")
        if bag.shape[1] != feature_dim:
            raise ValueError(
                f"bag {index} has feature width {bag.shape[1]}, "
                f"expected {feature_dim}"
            )
        if not bag.is_floating_point():
            raise TypeError(f"bag {index} must have a floating-point dtype")

    if not all(isinstance(target, torch.Tensor) for target in targets):
        raise TypeError("each target must be a torch.Tensor")

    lengths = torch.tensor([bag.shape[0] for bag in bags], dtype=torch.long)
    bag_ptr = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    return torch.cat(bags, dim=0), torch.stack(targets, dim=0), bag_ptr
