"""MRI modalities as input channels.

A multi-channel input stacks one acquisition per modality, in the fixed order of
``MODALITIES``, followed by one availability plane per modality (1 where the modality is
present in the input, 0 where it is missing or dropped). Missing image channels are zero.
All sequences of a study live on the same aligned grid, so channels are voxel-aligned.
"""
from __future__ import annotations

import random

import torch

from .geometry import SEQUENCE_KINDS, sequence_kind

MODALITIES = ("t1", "t2", "flair", "swi")
# Kinds that are read into a modality channel. Contrast-enhanced T1 shares the T1 channel.
KIND_TO_CHANNEL = {
    SEQUENCE_KINDS.index("t1"): 0,
    SEQUENCE_KINDS.index("t1ce"): 0,
    SEQUENCE_KINDS.index("t2"): 1,
    SEQUENCE_KINDS.index("flair"): 2,
    SEQUENCE_KINDS.index("swi"): 3,
}


def n_input_channels(modalities=MODALITIES) -> int:
    return 2 * len(modalities)


def modality_groups(image_paths) -> list[list[int]]:
    """Indices of the acquisitions of each modality; other sequences are not used."""
    groups = [[] for _ in MODALITIES]
    for index, path in enumerate(image_paths):
        channel = KIND_TO_CHANNEL.get(sequence_kind(path))
        if channel is not None:
            groups[channel].append(index)
    return groups


def stack_channels(volumes_by_channel, shape=None) -> torch.Tensor:
    """``[2M, D, H, W]`` from one volume (or ``None``) per modality."""
    present = [v for v in volumes_by_channel if v is not None]
    if not present:
        raise ValueError("At least one modality must be present")
    shape = tuple(present[0].shape) if shape is None else tuple(shape)
    dtype = present[0].dtype
    images, planes = [], []
    for volume in volumes_by_channel:
        if volume is None:
            images.append(torch.zeros(shape, dtype=dtype))
            planes.append(torch.zeros(shape, dtype=dtype))
        else:
            if tuple(volume.shape) != shape:
                raise ValueError(f"Modality volumes must share one grid, got {tuple(volume.shape)} and {shape}")
            images.append(volume)
            planes.append(torch.ones(shape, dtype=dtype))
    return torch.stack(images + planes)


def drop_channels(stack: torch.Tensor, keep) -> torch.Tensor:
    """Zero the image channel and availability plane of every modality not in ``keep``."""
    m = stack.shape[0] // 2
    out = stack.clone()
    for channel in range(m):
        if channel not in keep:
            out[channel].zero_()
            out[m + channel].zero_()
    return out


def subset_code(keep) -> int:
    return sum(1 << int(c) for c in keep)


def random_subset(available, rng: random.Random, full_probability: float, keep_probability: float = 0.5):
    """The full set with ``full_probability``, otherwise a random nonempty subset."""
    available = sorted(available)
    if not available:
        raise ValueError("No modality available")
    if len(available) == 1 or rng.random() < full_probability:
        return tuple(available)
    while True:
        keep = tuple(c for c in available if rng.random() < keep_probability)
        if keep:
            return keep


def acquisition_stacks(groups) -> list[list[int | None]]:
    """Every acquisition of every modality in at least one stack.

    Stack ``k`` uses acquisition ``k mod n_m`` of modality ``m``; the number of stacks is the
    largest acquisition count of any modality.
    """
    count = max((len(g) for g in groups), default=0)
    if count == 0:
        raise ValueError("Study has no acquisition of any input modality")
    return [[g[k % len(g)] if g else None for g in groups] for k in range(count)]
